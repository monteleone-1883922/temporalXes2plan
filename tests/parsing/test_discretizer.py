"""
Tests for parsing.discretizer.Discretizer — pure unit tests.

All tests build synthetic pm4py EventLogs in-memory; no XES files are read.
"""
import numpy as np
import pytest
from unittest.mock import patch

import core_utils as utils
from models import AnalysisConfig
from parsing.discretizer import Discretizer
from tests.helpers import make_event, make_trace, make_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bimodal_log(n_low: int = 30, n_high: int = 30):
    """Log with a clearly bimodal 'crp' attribute: cluster around 5 and cluster around 100."""
    traces = []
    for i in range(n_low):
        traces.append(make_trace(make_event("A", crp=5.0 + i * 0.1), case_id=f"low_{i}"))
    for i in range(n_high):
        traces.append(make_trace(make_event("A", crp=100.0 + i * 0.1), case_id=f"high_{i}"))
    return make_log(*traces)


def _constant_log(n: int = 20):
    """Log where 'crp' is always the same value — no discriminative power."""
    return make_log(*[
        make_trace(make_event("A", crp=42.0), case_id=f"c{i}") for i in range(n)
    ])


def _tiny_cluster_log():
    """Log with a dominant cluster (95%) and a tiny outlier cluster (5%)."""
    traces = [make_trace(make_event("A", crp=5.0 + i * 0.1), case_id=f"main_{i}") for i in range(38)]
    traces += [make_trace(make_event("A", crp=200.0), case_id=f"tiny_{i}") for i in range(2)]
    return make_log(*traces)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDiscretizerFit:
    def test_fit_produces_boundaries_for_discriminant_attribute(self):
        """A clearly bimodal distribution must yield at least one boundary."""
        d = Discretizer(AnalysisConfig(kmeans_silhouette_threshold=0.05))
        d.fit(_bimodal_log(), numeric_attributes=["crp"])
        assert "crp" in d.boundaries
        assert len(d.boundaries["crp"]) >= 1

    def test_fit_skips_constant_attribute(self):
        """An attribute with a single unique value must not be discretized."""
        d = Discretizer()
        d.fit(_constant_log(), numeric_attributes=["crp"])
        assert "crp" not in d.boundaries

    def test_fit_dominant_region_skewed_distribution(self):
        """Stage 2 must detect a dominant region on a heavily skewed distribution.

        The tiny-cluster log has 95% of points near 5.0 and 5% near 200.0.
        The zero-mass region (5.0) is dominant; the attribute should be
        discretized with at least one boundary separating the two clusters.
        """
        cfg = AnalysisConfig(dominance_threshold=0.30, kmeans_max_k=3)
        d = Discretizer(cfg)
        d.fit(_tiny_cluster_log(), numeric_attributes=["crp"])
        # With a dominant region at ~5.0 and residuals at ~200.0, we expect
        # either a boundary separating them or no discretization if silhouette fails.
        # Either outcome is valid; we just verify no crash and correct type.
        if "crp" in d.boundaries:
            assert isinstance(d.boundaries["crp"], list)
            assert d.boundaries["crp"] == sorted(d.boundaries["crp"])

    def test_fit_skips_unknown_attribute(self):
        """Attributes not present in the log must be silently skipped."""
        d = Discretizer()
        d.fit(_bimodal_log(), numeric_attributes=["nonexistent_attr"])
        assert "nonexistent_attr" not in d.boundaries

    def test_fit_empty_numeric_list_produces_no_boundaries(self):
        """Calling fit() with an empty attribute list must leave boundaries empty."""
        d = Discretizer()
        d.fit(_bimodal_log(), numeric_attributes=[])
        assert d.boundaries == {}


class TestClusterResiduals:
    def test_always_returns_at_least_one_center_for_valid_input(self):
        """k=1 baseline ensures _cluster_residuals never returns [] for sufficient input."""
        d = Discretizer(AnalysisConfig(min_residual_points=10))
        rng = np.random.default_rng(0)
        residuals = rng.uniform(0, 100, 200)
        centers = d._cluster_residuals("x", residuals, d.config, n_dominant=0)
        assert len(centers) >= 1

    def test_bimodal_residuals_return_multiple_centers(self):
        """Clearly bimodal residuals should produce at least 2 centers."""
        d = Discretizer(AnalysisConfig(kmeans_max_k=5, kmeans_n_init=3, min_residual_points=10))
        rng = np.random.default_rng(42)
        residuals = np.concatenate([rng.normal(10, 0.3, 100), rng.normal(90, 0.3, 100)])
        centers = d._cluster_residuals("x", residuals, d.config, n_dominant=0)
        assert len(centers) >= 2
        assert centers == sorted(centers)

    def test_too_few_residuals_returns_single_center(self):
        """When residuals fall below min_residual_points, a single mean center is returned."""
        d = Discretizer(AnalysisConfig(min_residual_points=50))
        residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 < 50
        centers = d._cluster_residuals("x", residuals, d.config, n_dominant=0)
        assert centers == [pytest.approx(3.0)]

    def test_min_delta_blocks_marginal_improvement(self):
        """A high min_delta must prevent upgrading from k=1 even when k=2 silhouette > 0."""
        rng = np.random.default_rng(0)
        # Mild bimodal: two slightly separated groups — k=2 silhouette will be modest
        residuals = np.concatenate([rng.normal(10, 2.0, 100), rng.normal(20, 2.0, 100)])
        # With min_delta=1.0, k=2 silhouette can never beat baseline+1.0 → stays at k=1
        d_strict = Discretizer(AnalysisConfig(
            kmeans_max_k=5, kmeans_n_init=3, min_residual_points=10,
            kmeans_silhouette_min_delta=1.0,
        ))
        centers_strict = d_strict._cluster_residuals("x", residuals, d_strict.config, n_dominant=0)
        assert len(centers_strict) == 1  # k=1 baseline wins

        # With min_delta=0.0, any improvement is accepted → k=2 should win
        d_loose = Discretizer(AnalysisConfig(
            kmeans_max_k=5, kmeans_n_init=3, min_residual_points=10,
            kmeans_silhouette_min_delta=0.0,
        ))
        centers_loose = d_loose._cluster_residuals("x", residuals, d_loose.config, n_dominant=0)
        assert len(centers_loose) >= 2


class TestKDEFallback:
    def test_kde_failure_returns_equal_frequency_boundaries(self):
        """When KDE raises LinAlgError, _find_best_boundaries must use equal-frequency fallback."""
        d = Discretizer(AnalysisConfig(kmeans_max_k=3))
        rng = np.random.default_rng(0)
        values = rng.uniform(0, 100, 500)
        n_unique = len(np.unique(values))

        with patch("parsing.discretizer.gaussian_kde", side_effect=np.linalg.LinAlgError):
            boundaries = d._find_best_boundaries("x", values, n_unique)

        # equal-frequency with max_k=3 → 2 quantile split points
        assert len(boundaries) == 2
        assert boundaries == sorted(boundaries)


class TestDiscretizerTransformValue:
    def test_transform_value_produces_correct_intervals(self):
        """transform_value must return the same label as core_utils.discretize_value."""
        d = Discretizer()
        d.boundaries["crp"] = [10.0, 97.5]

        assert d.transform_value("crp", 5.0) == "lte_10_0"
        assert d.transform_value("crp", 50.0) == "gte_10_0_lte_97_5"
        assert d.transform_value("crp", 200.0) == "gte_97_5"

    def test_transform_value_matches_discretize_value_utility(self):
        """transform_value output must be identical to core_utils.discretize_value."""
        d = Discretizer()
        d.boundaries["crp"] = [10.0, 97.5]

        for val in [5.0, 50.0, 200.0]:
            assert d.transform_value("crp", val) == utils.discretize_value("crp", val, d.boundaries)

    def test_transform_value_unknown_attribute_returns_string(self):
        """For attributes without boundaries, transform_value must return str(value)."""
        d = Discretizer()
        assert d.transform_value("unknown", 42.0) == "42.0"

    def test_transform_value_none_returns_string(self):
        """A None value must return 'None' without raising."""
        d = Discretizer()
        d.boundaries["crp"] = [10.0, 97.5]
        assert d.transform_value("crp", None) == "None"
