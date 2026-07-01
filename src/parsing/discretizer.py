import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List, Optional, Any, Tuple

import core_utils as utils
from models import AnalysisConfig

logger = utils.get_logger(__name__)

KMEANS_RETRY = 2


class Discretizer:
    """Pre-computes k-means++ boundaries for numerical attributes.

    Uses a 3-stage pipeline robust to skewed distributions:
      Stage 1 — short-circuit for few unique values.
      Stage 2 — KDE-based dominant region extraction.
      Stage 3 — KMeans on residuals with silhouette-only validation.

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

    def fit(self, log: EventLog, numeric_attributes: List[str]) -> None:
        """Compute boundaries for every numeric attribute in the log.

        Args:
            log: A pm4py EventLog object (full log, not just train split).
            numeric_attributes: Sanitized names of attributes typed as numerical.
        """
        if not numeric_attributes:
            logger.info("Discretizer: no numeric attributes to process.")
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
                logger.debug("Discretizer: '%s' has fewer than 2 non-null values, skipping.", attr)
                continue

            n_unique = len(np.unique(values))
            if n_unique < 2:
                logger.debug("Discretizer: '%s' is constant (1 unique value), skipping.", attr)
                continue

            bounds = self._find_best_boundaries(attr, values, n_unique)
            if bounds:
                self.boundaries[attr] = bounds
                logger.info("Discretizer: '%s' → %d bins, boundaries=%s", attr, len(bounds) + 1, bounds)
            else:
                logger.info("Discretizer: '%s' not discretized (silhouette below threshold or no valid k).", attr)

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

        # --- Stage 3: KMeans on residuals ---
        residual_centers = self._cluster_residuals(attr, residuals, cfg, n_dominant=len(dominant_centers))

        # Merge all bin centers, compute boundaries as midpoints
        all_centers = sorted(dominant_centers + residual_centers)
        if len(all_centers) < 2:
            return []

        boundaries = self._midpoints_between(np.array(all_centers))
        if not boundaries:
            return []

        # Final silhouette gate on the full value set
        labels = self._assign_labels(values, boundaries)
        if len(np.unique(labels)) < 2:
            return []

        score = silhouette_score(values.reshape(-1, 1), labels)
        if score < cfg.kmeans_silhouette_threshold:
            logger.debug(
                "Discretizer: '%s' final silhouette %.4f below threshold %.4f — discarding.",
                attr, score, cfg.kmeans_silhouette_threshold,
            )
            return []

        logger.debug("Discretizer: '%s' final silhouette=%.4f, %d boundaries", attr, score, len(boundaries))
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

        grid = np.linspace(values.min(), values.max(), 1000)
        density = kde(grid)

        peak_idx = argrelextrema(density, np.greater, order=5)[0]
        valley_idx = argrelextrema(density, np.less, order=5)[0]

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
    # Stage 3 — KMeans on residuals
    # ------------------------------------------------------------------

    def _cluster_residuals(
        self,
        attr: str,
        residuals: np.ndarray,
        cfg: AnalysisConfig,
        n_dominant: int,
    ) -> List[float]:
        """KMeans on residual values with silhouette-only k selection.

        Args:
            attr: Attribute name (for logging).
            residuals: Values not covered by any dominant region.
            cfg: Analysis config.
            n_dominant: Number of dominant bins already found (caps max_k).

        Returns:
            List of cluster center floats for the residuals.
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

        # k=1 baseline: all residuals in one group, silhouette=0 by convention
        best_score = 0.0
        best_centers: List[float] = [float(residuals.mean())]
        logger.debug("Discretizer: '%s' Stage 3 k=1 (baseline, silhouette=0.0000)", attr)

        for k in range(2, min(max_k_residual, n_unique_res) + 1):
            for _ in range(KMEANS_RETRY):
                km = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=cfg.kmeans_n_init,
                    random_state=42,
                )
                labels = km.fit_predict(residuals.reshape(-1, 1))
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(residuals.reshape(-1, 1), labels)
                logger.debug("Discretizer: '%s' Stage 3 k=%d silhouette=%.4f", attr, k, score)
                if score > best_score:
                    best_score = score
                    best_centers = sorted(km.cluster_centers_.flatten().tolist())

        return best_centers

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _midpoints_between(sorted_vals: np.ndarray) -> List[float]:
        """Return midpoints between consecutive sorted values."""
        return [
            round((float(sorted_vals[i]) + float(sorted_vals[i + 1])) / 2, 4)
            for i in range(len(sorted_vals) - 1)
        ]

    @staticmethod
    def _assign_labels(values: np.ndarray, boundaries: List[float]) -> np.ndarray:
        """Assign bin index to each value given sorted boundary list."""
        return np.digitize(values, boundaries)

    @staticmethod
    def _equal_frequency_boundaries(values: np.ndarray, n_bins: int) -> List[float]:
        """Equal-frequency (quantile) binning fallback."""
        quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        return [round(float(np.percentile(values, q)), 4) for q in quantiles]

    def _find_column(self, df: Any, attr: str) -> Optional[str]:
        """Return the DataFrame column name matching a sanitized attribute name."""
        for col in df.columns:
            if utils.sanitize_name(col) == attr:
                return col
        return None
