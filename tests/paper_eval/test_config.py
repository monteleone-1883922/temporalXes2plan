"""The per-log attribute lists in paper_eval.config.LOGS must be consistent."""
import pytest

from models import AnalysisConfig
from paper_eval.config import LOGS


@pytest.mark.parametrize("spec", LOGS.values(), ids=lambda s: s.name)
def test_ignore_and_keep_lists_do_not_overlap(spec):
    assert not set(spec.ignore_attributes) & set(spec.keep_attributes)


@pytest.mark.parametrize("spec", LOGS.values(), ids=lambda s: s.name)
def test_keep_does_not_name_a_default_ignored_attribute(spec):
    # keep only protects from the automatic filters; AnalysisConfig raises otherwise
    assert not set(spec.keep_attributes) & set(AnalysisConfig().ignored_attributes)


def test_sepsis_keeps_diagnose_from_the_noise_filter():
    assert "Diagnose" in LOGS["sepsis"].keep_attributes


class TestPaperDefaults:
    """The defaults follow the paper (§4.1 POWL, §5.1 split without spanning cases)."""

    def test_discovery_uses_powl(self):
        from paper_eval.config import PaperEvalConfig
        assert PaperEvalConfig().discovery_algorithm == "powl"

    def test_cases_spanning_the_split_point_are_dropped_from_train(self):
        from paper_eval.config import PaperEvalConfig
        assert PaperEvalConfig().drop_spanning_train is True

    def test_powl_is_a_discovery_algorithm_the_pipeline_supports(self):
        from paper_eval.config import PaperEvalConfig
        from parsing.model_discoverer import ModelDiscoverer
        assert PaperEvalConfig().discovery_algorithm in ModelDiscoverer.DISCOVERY_ALGORITHMS

    def test_unit_costs_is_a_switch(self):
        from paper_eval.config import PaperEvalConfig
        assert isinstance(PaperEvalConfig().unit_costs, bool)
        assert PaperEvalConfig(unit_costs=True).unit_costs is True
        assert PaperEvalConfig(unit_costs=False).unit_costs is False
