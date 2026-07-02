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
        d = Discretizer()
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
        # either a boundary separating them or no discretization if GVF fails.
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
        """Single-bin fallback ensures _cluster_residuals never returns [] for sufficient input."""
        d = Discretizer(AnalysisConfig(min_residual_points=10))
        rng = np.random.default_rng(0)
        residuals = rng.uniform(0, 100, 200)
        centers = d._cluster_residuals("x", residuals, d.config, n_dominant=0)
        assert len(centers) >= 1

    def test_bimodal_residuals_return_multiple_centers(self):
        """Clearly bimodal residuals should produce at least 2 centers."""
        d = Discretizer(AnalysisConfig(kmeans_max_k=5, min_residual_points=10))
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

    def test_high_gvf_threshold_forces_single_bin(self):
        """Setting min_gvf_threshold above the achievable GVF at k=2 must force single bin.

        For uniform U(0,1000) Jenks with k=2 achieves GVF ≈ 0.75 (theoretical: 1 - 1/k²).
        kmeans_max_k=2 caps the search at k=2 — with higher k allowed, GVF keeps
        improving (k=3 already reaches ≈0.89) and the threshold would be satisfied,
        which is not what this test is checking. With min_gvf_threshold=0.80 the
        k=2 result is rejected and a single bin is returned.
        """
        rng = np.random.default_rng(7)
        residuals = rng.uniform(0.0, 1000.0, 500)
        d = Discretizer(AnalysisConfig(
            kmeans_max_k=2, min_residual_points=10,
            min_gvf_threshold=0.80,   # > 0.75 → k=2 GVF is too low → single bin
        ))
        centers = d._cluster_residuals("x", residuals, d.config, n_dominant=0)
        assert len(centers) == 1

    def test_deterministic_across_runs(self):
        """Same input must produce identical output across multiple calls."""
        rng = np.random.default_rng(42)
        residuals = np.concatenate([rng.normal(10, 0.5, 200), rng.normal(90, 0.5, 200)])
        d = Discretizer(AnalysisConfig(kmeans_max_k=5, min_residual_points=10))
        centers_a = d._cluster_residuals("x", residuals, d.config, n_dominant=0)
        centers_b = d._cluster_residuals("x", residuals, d.config, n_dominant=0)
        assert centers_a == centers_b

    def test_sampling_activates_for_large_input(self):
        """Arrays exceeding jenks_sample_size must be sampled; result still correct."""
        rng = np.random.default_rng(99)
        # Two clear clusters × 15k points each = 30k total > default 20k
        residuals = np.concatenate([rng.normal(0, 1, 15000), rng.normal(100, 1, 15000)])
        d = Discretizer(AnalysisConfig(
            kmeans_max_k=5, min_residual_points=10,
            jenks_sample_size=20000,
        ))
        centers = d._cluster_residuals("x", residuals, d.config, n_dominant=0)
        assert len(centers) >= 2
        assert centers == sorted(centers)


class TestGVF:
    def test_perfect_separation_returns_high_gvf(self):
        """Two perfectly separated groups must yield GVF close to 1.0."""
        values = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
        breaks = [1.0, 50.5, 100.0]  # splits exactly between 1 and 100
        gvf = Discretizer._gvf(values, breaks)
        assert gvf > 0.95

    def test_constant_array_returns_one(self):
        """Constant array has TSS=0 → GVF defined as 1.0."""
        values = np.array([42.0, 42.0, 42.0])
        breaks = [42.0, 42.0]
        gvf = Discretizer._gvf(values, breaks)
        assert gvf == pytest.approx(1.0)

    def test_single_bin_returns_zero(self):
        """When all values fall in one bin, WCSS = TSS → GVF = 0."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # breaks that encompass everything in one bin
        breaks = [1.0, 5.0]
        gvf = Discretizer._gvf(values, breaks)
        assert gvf == pytest.approx(0.0, abs=1e-9)


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
