import jenkspy
import json
import numpy as np
import pm4py
from pathlib import Path
from pm4py.objects.log.obj import EventLog
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from typing import Dict, List, Optional, Any, Tuple

import core_utils as utils
from models import AnalysisConfig

logger = utils.get_logger(__name__)

RNG_RETRY = 2

class Discretizer:
    """Pre-computes Jenks Natural Breaks boundaries for numerical attributes.

    Uses a 3-stage pipeline robust to skewed distributions:
      Stage 1 — short-circuit for few unique values.
      Stage 2 — KDE-based dominant region extraction.
      Stage 3 — Jenks Natural Breaks on residuals with GVF-based k selection.

    Must be fit() before any DT training so that all downstream components
    (decision_mining, effect_analyzer, correlation_miner) share the same
    fixed interval vocabulary.

    Attributes:
        boundaries: Mapping from sanitized attribute name to sorted list of
            split-point floats. E.g. {"crp": [10.0, 97.5]}.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        self.config = config or AnalysisConfig()
        self.boundaries: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        log: EventLog,
        numeric_attributes: List[str],
        cache_path: Optional[Path] = None,
        force: bool = False,
    ) -> None:
        """Compute boundaries for every numeric attribute in the log.

        Args:
            log: A pm4py EventLog object (full log, not just train split).
            numeric_attributes: Sanitized names of attributes typed as numerical.
            cache_path: Optional path to a JSON cache file. When provided and the
                file exists (and force=False), boundaries are loaded from it instead
                of being recomputed. After computation the boundaries are saved there.
            force: If True, ignore an existing cache and recompute from scratch.
        """
        if not numeric_attributes:
            logger.info("Discretizer: no numeric attributes to process.")
            return

        if cache_path is not None and not force and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.boundaries = {k: v for k, v in cached.items() if k in numeric_attributes}
            missing = [a for a in numeric_attributes if a not in self.boundaries]
            logger.info(
                "Discretizer: boundaries loaded from cache %s (%d attrs)",
                cache_path, len(self.boundaries),
            )
            if missing:
                logger.warning(
                    "Discretizer: %d attribute(s) in numeric_attributes not found in cache "
                    "(cache may be stale or attribute was skipped at build time): %s",
                    len(missing), missing,
                )
            return

        df = pm4py.convert_to_dataframe(log)

        for attr in numeric_attributes:
            col = self._find_column(df, attr)
            if col is None:
                logger.debug("Discretizer: column for attribute '%s' not found in log, skipping.", attr)
                continue

            values = df[col].dropna().to_numpy(dtype=float, na_value=np.nan)
            values = values[~np.isnan(values)]

            if len(values) < 2:
                logger.warning(
                    "Discretizer: '%s' skipped — only %d non-null value(s) found in log "
                    "(attribute is too sparse to discretize).",
                    attr, len(values),
                )
                continue

            n_unique = len(np.unique(values))
            if n_unique < 2:
                logger.warning(
                    "Discretizer: '%s' skipped — all %d non-null values are identical "
                    "(constant attribute, no split point possible).",
                    attr, len(values),
                )
                continue

            bounds = self._find_best_boundaries(attr, values, n_unique)
            if bounds:
                self.boundaries[attr] = bounds
                logger.info("Discretizer: '%s' → %d bins, boundaries=%s", attr, len(bounds) + 1, bounds)
            else:
                logger.info("Discretizer: '%s' not discretized (GVF below threshold or no meaningful structure).", attr)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(self.boundaries), encoding="utf-8")
            logger.info("Discretizer: boundaries saved to cache %s", cache_path)

    def transform_value(self, attr: str, value: Any) -> str:
        """Convert a numeric value to its interval label.

        Args:
            attr: Sanitized attribute name.
            value: Raw numeric value.

        Returns:
            Interval label string, e.g. "lte_10_0", "gte_10_0_lte_97_5", "gte_97_5".
            Falls back to str(value) if attr has no boundaries or value is None.
        """
        if attr not in self.boundaries or value is None:
            return str(value)
        return utils.discretize_value(attr, value, self.boundaries)

    def all_bin_labels(self, attr: str) -> List[str]:
        """Return every possible interval label for a discretized attribute.

        Args:
            attr: Sanitized attribute name.

        Returns:
            Ordered list of bin label strings covering (-inf, +inf).
            Returns [] if the attribute has no boundaries.
        """
        thresholds = self.boundaries.get(attr)
        if not thresholds:
            return []
        return Discretizer.labels_from_splits(thresholds)

    @staticmethod
    def labels_from_splits(splits: List[float]) -> List[str]:
        """Return all categorical interval labels for a given list of split points.

        Mirrors the label format produced by core_utils.discretize_value.

        Args:
            splits: Boundary floats (order does not matter).

        Returns:
            Ordered list of label strings covering (-inf, +inf).
            E.g. splits=[10.0, 50.0] → ["lte_10_0", "gte_10_0_lte_50_0", "gte_50_0"].
            Returns [] for an empty splits list.
        """
        if not splits:
            return []
        thresholds = sorted(splits)

        def _fmt(t: float) -> str:
            return str(t).replace('.', '_').replace('-', 'neg')

        labels: List[str] = [f"lte_{_fmt(thresholds[0])}"]
        for i in range(len(thresholds) - 1):
            labels.append(f"gte_{_fmt(thresholds[i])}_lte_{_fmt(thresholds[i + 1])}")
        labels.append(f"gte_{_fmt(thresholds[-1])}")
        return labels

    # ------------------------------------------------------------------
    # Private orchestrator
    # ------------------------------------------------------------------

    def _find_best_boundaries(
        self, attr: str, values: np.ndarray, n_unique: int
    ) -> List[float]:
        """3-stage pipeline: returns sorted boundary floats or [] if quality fails.

        Args:
            attr: Attribute name (used for logging).
            values: 1-D array of non-null numeric values.
            n_unique: Number of unique values in the array.

        Returns:
            Sorted list of boundary floats, or [] if no valid discretization found.
        """
        cfg = self.config

        # --- Stage 1: few unique values ---
        unique_vals = np.unique(values)
        if len(unique_vals) <= cfg.kmeans_max_k:
            boundaries = self._midpoints_between(unique_vals)
            logger.debug(
                "Discretizer: '%s' Stage 1 — %d unique values → %d boundaries",
                attr, len(unique_vals), len(boundaries),
            )
            return boundaries

        # --- Stage 2: dominant region extraction via KDE ---
        dominant_centers, residuals = self._extract_dominant_regions(attr, values, cfg)

        if dominant_centers is None:  # KDE failed — use equal-frequency fallback
            return self._equal_frequency_boundaries(values, cfg.kmeans_max_k)

        # --- Stage 3: Jenks Natural Breaks on residuals ---
        residual_centers = self._cluster_residuals(attr, residuals, cfg, n_dominant=len(dominant_centers))

        # Merge all bin centers, compute boundaries as midpoints
        all_centers = sorted(dominant_centers + residual_centers)
        if len(all_centers) < 2:
            return []

        boundaries = self._midpoints_between(np.array(all_centers))
        if not boundaries:
            return []

        return boundaries

    # ------------------------------------------------------------------
    # Stage 2 — dominant region extraction
    # ------------------------------------------------------------------

    def _extract_dominant_regions(
        self, attr: str, values: np.ndarray, cfg: AnalysisConfig
    ) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
        """Identify KDE peak regions whose mass exceeds dominance_threshold.

        Args:
            attr: Attribute name (for logging).
            values: Full value array.
            cfg: Analysis config.

        Returns:
            (dominant_centers, residuals) where dominant_centers is a list of
            mean values for each dominant region and residuals is the sub-array
            of values not covered by any dominant region.
            Returns (None, None) when KDE fails — orchestrator must use equal-frequency fallback.
        """
        try:
            kde = gaussian_kde(values, bw_method="silverman")
        except np.linalg.LinAlgError:
            logger.warning(
                "Discretizer: '%s' KDE failed (singular matrix) — falling back to equal-frequency.",
                attr,
            )
            return None, None

        grid = np.linspace(values.min(), values.max(), self.config.kde_grid_points)
        density = kde(grid)

        peak_idx = argrelextrema(density, np.greater, order=self.config.kde_extrema_order)[0]
        valley_idx = argrelextrema(density, np.less, order=self.config.kde_extrema_order)[0]

        # Sentinel valleys at the extremes
        valley_positions = np.concatenate([[grid[0]], grid[valley_idx], [grid[-1]]])

        dominant_centers: List[float] = []
        dominant_mask = np.zeros(len(values), dtype=bool)

        for pi in peak_idx:
            peak_pos = grid[pi]
            left = valley_positions[valley_positions <= peak_pos].max()
            right = valley_positions[valley_positions >= peak_pos].min()

            in_region = (values >= left) & (values <= right)
            mass = in_region.mean()

            if mass >= cfg.dominance_threshold:
                center = float(values[in_region].mean())
                dominant_centers.append(center)
                dominant_mask |= in_region
                logger.debug(
                    "Discretizer: '%s' dominant region [%.4f, %.4f] mass=%.1f%% center=%.4f",
                    attr, left, right, mass * 100, center,
                )

        residuals = values[~dominant_mask]
        logger.debug(
            "Discretizer: '%s' Stage 2 — %d dominant region(s), %d residual points",
            attr, len(dominant_centers), len(residuals),
        )
        return dominant_centers, residuals

    # ------------------------------------------------------------------
    # Stage 3 — Jenks Natural Breaks on residuals
    # ------------------------------------------------------------------

    def _cluster_residuals(
            self,
            attr: str,
            residuals: np.ndarray,
            cfg: AnalysisConfig,
            n_dominant: int,
    ) -> List[float]:
        """Jenks Natural Breaks on residual values with GVF-based k selection.

        Args:
            attr: Attribute name (for logging).
            residuals: Values not covered by any dominant region.
            cfg: Analysis config.
            n_dominant: Number of dominant bins already found (caps max_k).

        Returns:
            List of bin center floats for the residuals.
        """
        if len(residuals) < cfg.min_residual_points or len(np.unique(residuals)) < 2:
            if len(residuals) > 0:
                logger.debug(
                    "Discretizer: '%s' Stage 3 — too few residuals (%d), single bin.",
                    attr, len(residuals),
                )
                return [float(residuals.mean())]
            return []

        max_k_residual = max(2, cfg.kmeans_max_k - n_dominant)
        n_unique_res = len(np.unique(residuals))
        k_max = min(max_k_residual, n_unique_res)

        # Global best across all retries
        best_k = 1
        best_breaks: Optional[List[float]] = None
        best_gvf: float = 0.0
        best_found = False

        retry_sample_loops = RNG_RETRY if len(residuals) > cfg.jenks_sample_size else 1
        for i in range(retry_sample_loops):
            if best_found:
                break

            # Stratified sample for Jenks DP (O(n²k) — too slow on large arrays)
            if len(residuals) > cfg.jenks_sample_size:
                rng = np.random.default_rng(42 + i)
                idx = rng.choice(len(residuals), size=cfg.jenks_sample_size, replace=False)
                sample = np.sort(residuals[idx])
                logger.debug(
                    "Discretizer: '%s' Stage 3 — sampling %d/%d points for Jenks (seed=%d)",
                    attr, cfg.jenks_sample_size, len(residuals), 42 + i,
                )
            else:
                sample = residuals

            # Per-retry state for marginal gain early stop
            prev_gvf_local: Optional[float] = None

            for k in range(2, k_max + 1):
                try:
                    breaks = jenkspy.jenks_breaks(sample.tolist(), n_classes=k)
                except Exception as exc:
                    logger.warning(
                        "Discretizer: '%s' Stage 3 — jenkspy failed at k=%d (%s) — stopping k search.",
                        attr, k, exc,
                    )
                    break

                gvf = self._gvf(residuals, breaks)
                logger.debug("Discretizer: '%s' Stage 3 k=%d GVF=%.4f (retry %d)", attr, k, gvf, i)

                # Early stop a: excellent fit reached
                if gvf >= cfg.gvf_target:
                    best_k = k
                    best_breaks = breaks
                    best_gvf = gvf
                    best_found = True
                    break

                # Early stop b: marginal gain too small within this retry — stop searching higher k
                if prev_gvf_local is not None and (gvf - prev_gvf_local) < cfg.min_gvf_improvement:
                    break

                # Update global best if this k is better than anything found so far
                if gvf > best_gvf:
                    best_k = k
                    best_breaks = breaks
                    best_gvf = gvf

                prev_gvf_local = gvf

        # Quality gate: best partition must explain enough variance
        if best_k == 1 or best_breaks is None or best_gvf < cfg.min_gvf_threshold:
            logger.debug(
                "Discretizer: '%s' Stage 3 — best GVF (%.4f at k=%d) below threshold %.2f → single bin.",
                attr, best_gvf, best_k, cfg.min_gvf_threshold,
            )
            return [float(residuals.mean())]

        return self._jenks_bin_centers(residuals, best_breaks)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gvf(values: np.ndarray, breaks: List[float]) -> float:
        """Goodness of Variance Fit: 1 - WCSS/TSS.

        Args:
            values: 1-D array of all residual values (not just the sample).
            breaks: Jenks breaks [min_val, b1, ..., max_val] — k+1 elements for k bins.

        Returns:
            GVF in [0.0, 1.0]. Returns 1.0 for constant arrays (TSS=0).
        """
        mean = float(values.mean())
        tss = float(np.sum((values - mean) ** 2))
        if tss == 0.0:
            return 1.0
        wcss = 0.0
        for i in range(len(breaks) - 1):
            # First bin is closed on both sides; subsequent bins are half-open (b_i, b_{i+1}]
            if i == 0:
                mask = (values >= breaks[0]) & (values <= breaks[1])
            else:
                mask = (values > breaks[i]) & (values <= breaks[i + 1])
            if mask.any():
                bin_vals = values[mask]
                wcss += float(np.sum((bin_vals - float(bin_vals.mean())) ** 2))
        return 1.0 - wcss / tss

    @staticmethod
    def _jenks_bin_centers(values: np.ndarray, breaks: List[float]) -> List[float]:
        """Return the mean of values in each Jenks bin as the bin center.

        Args:
            values: Full residual array (all points, not just the sample).
            breaks: Jenks breaks [min, b1, ..., max] for k bins.

        Returns:
            Sorted list of k center floats (one per non-empty bin).
        """
        centers = []
        for i in range(len(breaks) - 1):
            if i == 0:
                mask = (values >= breaks[0]) & (values <= breaks[1])
            else:
                mask = (values > breaks[i]) & (values <= breaks[i + 1])
            if mask.any():
                centers.append(float(values[mask].mean()))
        return sorted(centers)

    @staticmethod
    def _midpoints_between(sorted_vals: np.ndarray) -> List[float]:
        """Return midpoints between consecutive sorted values."""
        return [
            round((float(sorted_vals[i]) + float(sorted_vals[i + 1])) / 2, 4)
            for i in range(len(sorted_vals) - 1)
        ]

    @staticmethod
    def _equal_frequency_boundaries(values: np.ndarray, n_bins: int) -> List[float]:
        """Equal-frequency (quantile) binning fallback."""
        quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        return [round(float(np.percentile(values, q)), 4) for q in quantiles]

    def _find_column(self, df: Any, attr: str) -> Optional[str]:
        """Return the DataFrame column name matching a sanitized attribute name."""
        for col in df.columns:
            if col == attr:
                return col
        return None
