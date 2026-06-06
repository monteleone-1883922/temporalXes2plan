import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List, Optional, Any

import core_utils as utils
from models import AnalysisConfig

logger = utils.get_logger(__name__)


class Discretizer:
    """Pre-computes k-means++ boundaries for numerical attributes.

    Must be fit() before any DT training so that all downstream components
    (decision_mining, effect_analyzer, correlation_miner) share the same
    fixed interval vocabulary.

    Attributes:
        boundaries: Mapping from sanitized attribute name to sorted list of
            split-point floats. E.g. {"crp": [10.0, 97.5]}.
            Attributes with insufficient variance or that fail the silhouette
            threshold are absent from this dict.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        self.config = config or AnalysisConfig()
        self.boundaries: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, log: EventLog, numeric_attributes: List[str]) -> None:
        """Compute k-means++ boundaries for every numeric attribute in the log.

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
                logger.debug(f"Discretizer: column for attribute '{attr}' not found in log, skipping.")
                continue

            values = df[col].dropna().to_numpy(dtype=float, na_value=np.nan)
            values = values[~np.isnan(values)]

            if len(values) < 2:
                logger.debug(f"Discretizer: '{attr}' has fewer than 2 non-null values, skipping.")
                continue

            n_unique = len(np.unique(values))
            if n_unique < 2:
                logger.debug(f"Discretizer: '{attr}' is constant (1 unique value), skipping.")
                continue

            bounds = self._find_best_boundaries(attr, values, n_unique)
            if bounds:
                self.boundaries[attr] = bounds
                logger.info(f"Discretizer: '{attr}' → {len(bounds)+1} bins, boundaries={bounds}")
            else:
                logger.info(f"Discretizer: '{attr}' not discretized (silhouette below threshold or no valid k).")

    def transform_value(self, attr: str, value: Any) -> str:
        """Convert a numeric value to its interval label.

        Produces the same format as core_utils.discretize_value so that the
        rest of the pipeline (correlation_miner, pddl_helper) remains unchanged.

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
    # Private helpers
    # ------------------------------------------------------------------

    def _find_column(self, df: Any, attr: str) -> Optional[str]:
        """Return the DataFrame column name matching a sanitized attribute name."""
        for col in df.columns:
            if utils.sanitize_name(col) == attr:
                return col
        return None

    def _find_best_boundaries(
        self, attr: str, values: np.ndarray, n_unique: int
    ) -> List[float]:
        """Run k-means++ for k in [2, max_k] and return boundaries for best k.

        Selects k via silhouette score. Enforces minimum cluster size.
        Returns an empty list when no k meets the quality threshold.

        Args:
            attr: Attribute name (used only for logging).
            values: 1-D array of non-null numeric values.
            n_unique: Number of unique values in the array.

        Returns:
            Sorted list of boundary floats (len = best_k - 1), or [] if none qualify.
        """
        cfg = self.config
        effective_max_k = min(cfg.kmeans_max_k, n_unique - 1)

        best_score = -np.inf
        best_boundaries: List[float] = []

        for k in range(2, effective_max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=cfg.kmeans_n_init,
                random_state=42,
            )
            labels = kmeans.fit_predict(values.reshape(-1, 1))

            # Guard: reject k if any cluster is too small
            cluster_fractions = np.bincount(labels) / len(labels)
            if np.any(cluster_fractions < cfg.kmeans_min_cluster_fraction):
                #TODO maybe we can remove outliers
                logger.debug(
                    f"Discretizer: '{attr}' k={k} rejected — "
                    f"min cluster fraction {cluster_fractions.min():.3f} < {cfg.kmeans_min_cluster_fraction}"
                )
                continue

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(values.reshape(-1, 1), labels)
            logger.debug(f"Discretizer: '{attr}' k={k} silhouette={score:.4f}")

            if score > best_score:
                best_score = score
                centroids = sorted(kmeans.cluster_centers_.flatten())
                best_boundaries = [
                    round((centroids[i] + centroids[i + 1]) / 2, 4)
                    for i in range(len(centroids) - 1)
                ]

        if best_score < cfg.kmeans_silhouette_threshold:
            return []

        return best_boundaries
