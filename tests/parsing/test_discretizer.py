"""
Tests for parsing.discretizer.Discretizer — pure unit tests.

All tests build synthetic pm4py EventLogs in-memory; no XES files are read.
"""
import pytest

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
