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


def test_factory_is_a_local_log_without_its_event_attributes():
    spec = LOGS["factory"]
    assert spec.local and spec.has_lifecycle
    assert set(spec.ignore_attributes) == {"Activity", "Resource", "elementId", "resourceCost", "resourceId"}
    # only the factory is provided by hand
    assert [s.name for s in LOGS.values() if s.local] == ["factory"]


def test_rq1_run_parameters_decided_on_2026_11_09():
    from paper_eval.config import PaperEvalConfig
    cfg = PaperEvalConfig()
    assert cfg.plan_timeout_s == 120 and cfg.plan_memory_mb == 12288  # 2 minutes, 12 GB
    assert cfg.search_n_trials == 100 and cfg.unit_costs is True and cfg.seed == 42


class TestPaperDefaults:
    """The defaults follow the paper (§4.1 POWL, §5.1 split without spanning cases)."""

    def test_discovery_uses_powl(self):
        from paper_eval.config import PaperEvalConfig
        assert PaperEvalConfig().discovery_algorithm == "powl"

    def test_dropping_spanning_cases_from_train_is_a_switch_and_off_by_default(self):
        # off since 2026-11-09: on BPIC13 it would leave 99 training cases of 993
        from paper_eval.config import PaperEvalConfig
        assert PaperEvalConfig().drop_spanning_train is False
        assert PaperEvalConfig(drop_spanning_train=True).drop_spanning_train is True

    def test_powl_is_a_discovery_algorithm_the_pipeline_supports(self):
        from paper_eval.config import PaperEvalConfig
        from parsing.model_discoverer import ModelDiscoverer
        assert PaperEvalConfig().discovery_algorithm in ModelDiscoverer.DISCOVERY_ALGORITHMS

    def test_unit_costs_is_a_switch(self):
        from paper_eval.config import PaperEvalConfig
        assert isinstance(PaperEvalConfig().unit_costs, bool)
        assert PaperEvalConfig(unit_costs=True).unit_costs is True
        assert PaperEvalConfig(unit_costs=False).unit_costs is False


def test_only_bpic13_incidents_reads_its_lifecycle_as_a_status():
    # its lifecycle:transition holds sub-statuses; closed problems already has them in concept:name
    assert [n for n, spec in LOGS.items() if spec.status_as_activity] == ["bpic13_i"]


def test_replay_time_limit_is_2_min_30_in_paper_eval_and_off_in_the_library_default():
    from models import AnalysisConfig
    from paper_eval.config import PaperEvalConfig
    assert AnalysisConfig().replay_time_limit_s is None       # thesis and web: no limit
    assert PaperEvalConfig().replay_time_limit_s == 150.0     # paper_eval: 2 min 30 s
