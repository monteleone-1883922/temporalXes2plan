"""
Tests for parsing.temporal_extractor.

Covers the three duration extraction strategies and the data model validation.
All event logs are built in-memory via conftest helpers — no XES files are loaded.

Strategy overview:
  - extract_from_lifecycle_pn:          reads FiringStep.duration_seconds from PetriNetLog.
  - estimate_from_inter_event_times_pn: gaps between consecutive labeled PetriNetLog steps.
  - from_external:                      accepts user-supplied min/max bounds directly.
  - extract:                            combines all three with priority external > lifecycle > inter_event.
"""
import math
import logging
from datetime import datetime, timedelta, timezone

import pytest

from pm4py import PetriNet, Marking

from tests.helpers import make_event, make_trace, make_log, _transition, ts
from parsing.temporal_extractor import (
    ExternalDuration,
    TemporalExtractor,
    _stats_to_duration,
)
from models import ActionDurationStats, FiringStep, TraceExecution, PetriNetLog


def _step_with_duration(activity: str, duration: float) -> FiringStep:
    """FiringStep with a pre-set lifecycle duration (simulates PetriNetLogBuilder output)."""
    t = _transition(f"t_{activity}", activity)
    return FiringStep(
        transition=t,
        activity_name=activity,
        is_tau=False,
        from_places=set(),
        attributes={},
        duration_seconds=duration,
    )


# ===========================================================================
# ExternalDuration — input validation
# ===========================================================================

class TestExternalDuration:
    def test_valid_bounds_are_accepted(self):
        """min < max is the normal case and must not raise."""
        ext = ExternalDuration(min_duration=10.0, max_duration=100.0)
        assert ext.min_duration == 10.0
        assert ext.max_duration == 100.0

    def test_equal_bounds_are_accepted(self):
        """min == max represents a fixed-duration action and is valid."""
        ext = ExternalDuration(min_duration=30.0, max_duration=30.0)
        assert ext.min_duration == ext.max_duration

    def test_inverted_bounds_raise_value_error(self):
        """min > max is physically impossible and must raise ValueError."""
        with pytest.raises(ValueError, match="min_duration"):
            ExternalDuration(min_duration=200.0, max_duration=50.0)


# ===========================================================================
# _stats_to_duration — the core formula max(obs_min, mean-std) / min(obs_max, mean+std)
# ===========================================================================

class TestStatsToDuration:
    def test_single_observation_yields_point_interval(self):
        """
        With one sample, std_dev is 0.  The formula collapses to
        effective_min == effective_max == the single observed value.
        """
        stats = _stats_to_duration([60.0], source="lifecycle")
        assert stats.mean == pytest.approx(60.0)
        assert stats.std_dev == pytest.approx(0.0)
        assert stats.effective_min == pytest.approx(60.0)
        assert stats.effective_max == pytest.approx(60.0)
        assert stats.count == 1

    def test_uniform_observations_same_as_single(self):
        """All identical values produce the same point interval as a single observation."""
        stats = _stats_to_duration([120.0, 120.0, 120.0], source="inter_event")
        assert stats.std_dev == pytest.approx(0.0)
        assert stats.effective_min == pytest.approx(stats.effective_max)

    def test_typical_case_applies_formula_correctly(self):
        """
        For a spread distribution the effective bounds must equal
        max(obs_min, mean - std) and min(obs_max, mean + std).
        """
        durations = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = _stats_to_duration(durations, source="lifecycle")

        expected_mean = 30.0
        expected_std = math.sqrt(
            sum((d - expected_mean) ** 2 for d in durations) / len(durations)
        )
        assert stats.mean == pytest.approx(expected_mean)
        assert stats.std_dev == pytest.approx(expected_std)
        assert stats.effective_min == pytest.approx(
            max(min(durations), expected_mean - expected_std)
        )
        assert stats.effective_max == pytest.approx(
            min(max(durations), expected_mean + expected_std)
        )

    def test_effective_bounds_never_inverted(self):
        """effective_min must always be <= effective_max for any input."""
        for durations in [[1.0], [0.0, 100.0], [5.0, 5.0, 5.0, 100.0]]:
            stats = _stats_to_duration(durations, source="lifecycle")
            assert stats.effective_min <= stats.effective_max

    def test_observed_min_max_stored_correctly(self):
        """observed_min and observed_max must reflect the actual extremes in the data."""
        stats = _stats_to_duration([3.0, 7.0, 15.0], source="lifecycle")
        assert stats.observed_min == pytest.approx(3.0)
        assert stats.observed_max == pytest.approx(15.0)

    def test_source_label_is_preserved(self):
        """The source string passed in must be stored verbatim on the returned object."""
        assert _stats_to_duration([60.0], source="lifecycle").source == "lifecycle"
        assert _stats_to_duration([60.0], source="inter_event").source == "inter_event"

    def test_count_matches_number_of_observations(self):
        """count must equal the length of the durations list passed in."""
        assert _stats_to_duration([1.0, 2.0, 3.0, 4.0], source="lifecycle").count == 4


# ===========================================================================
# TemporalExtractor.extract_from_lifecycle_pn — Case 1
# ===========================================================================

class TestExtractFromLifecyclePn:
    def test_step_with_duration_is_collected(self):
        """A FiringStep with duration_seconds=300 must produce a lifecycle entry."""
        ex = TraceExecution("c1", [_step_with_duration("visit", 300.0)])
        result = TemporalExtractor().extract_from_lifecycle_pn(_pn_log(ex))
        assert "visit" in result
        assert result["visit"].source == "lifecycle"
        assert result["visit"].count == 1
        assert result["visit"].mean == pytest.approx(300.0)

    def test_step_without_duration_is_skipped(self):
        """A FiringStep with duration_seconds=None must not appear in the result."""
        t = _transition("t_a", "a")
        step = FiringStep(
            transition=t, activity_name="a", is_tau=False,
            from_places=set(), attributes={}, duration_seconds=None,
        )
        ex = TraceExecution("c1", [step])
        result = TemporalExtractor().extract_from_lifecycle_pn(_pn_log(ex))
        assert "a" not in result

    def test_tau_step_is_always_skipped(self):
        """Tau steps never carry lifecycle durations."""
        step = _pn_step("tau_1", 0, is_tau=True)
        ex = TraceExecution("c1", [step])
        assert TemporalExtractor().extract_from_lifecycle_pn(_pn_log(ex)) == {}

    def test_multiple_executions_accumulate(self):
        """Durations from different executions for the same activity are aggregated."""
        ex1 = TraceExecution("c1", [_step_with_duration("triage", 60.0)])
        ex2 = TraceExecution("c2", [_step_with_duration("triage", 120.0)])
        result = TemporalExtractor().extract_from_lifecycle_pn(_pn_log(ex1, ex2))
        assert result["triage"].count == 2
        assert result["triage"].mean == pytest.approx(90.0)

    def test_source_is_lifecycle(self):
        """All entries returned must carry source='lifecycle'."""
        ex = TraceExecution("c1", [_step_with_duration("x", 100.0)])
        result = TemporalExtractor().extract_from_lifecycle_pn(_pn_log(ex))
        assert result["x"].source == "lifecycle"

    def test_empty_pn_log_returns_empty_dict(self):
        assert TemporalExtractor().extract_from_lifecycle_pn(_pn_log()) == {}

    def test_mixed_steps_some_with_some_without_duration(self):
        """Only steps with a non-None duration contribute — others are silently skipped."""
        steps = [
            _step_with_duration("scan", 200.0),
            FiringStep(
                transition=_transition("t_b", "blood_test"),
                activity_name="blood_test",
                is_tau=False,
                from_places=set(),
                attributes={},
                duration_seconds=None,
            ),
        ]
        ex = TraceExecution("c1", steps)
        result = TemporalExtractor().extract_from_lifecycle_pn(_pn_log(ex))
        assert "scan" in result
        assert "blood_test" not in result


# ===========================================================================
# TemporalExtractor.from_external — Case 2
# ===========================================================================

class TestFromExternal:
    def test_valid_entry_sets_effective_bounds_and_leaves_stats_none(self):
        """
        A valid ExternalDuration must be converted to an ActionDurationStats
        with effective_min/max matching the input and all statistical fields None.
        """
        ext = {"surgery": ExternalDuration(min_duration=600.0, max_duration=3600.0)}
        result = TemporalExtractor().from_external(ext)

        assert "surgery" in result
        stats = result["surgery"]
        assert stats.effective_min == pytest.approx(600.0)
        assert stats.effective_max == pytest.approx(3600.0)
        assert stats.source == "external"
        # Statistical fields must be absent for externally supplied data
        assert stats.mean is None
        assert stats.std_dev is None
        assert stats.observed_min is None
        assert stats.observed_max is None
        assert stats.count is None

    def test_invalid_entry_is_skipped_with_warning(self, caplog):
        """
        An ExternalDuration with min > max is invalid.  It must be skipped and
        a WARNING must be logged.  (ExternalDuration.__post_init__ may catch it
        at construction time; the method also re-validates defensively.)
        """
        # Build the object by patching the values after construction to bypass
        # __post_init__, simulating a tampered object reaching from_external.
        bad = ExternalDuration(10.0, 100.0)
        object.__setattr__(bad, "min_duration", 999.0)  # min > max after construction

        with caplog.at_level(logging.WARNING, logger="parsing.temporal_extractor"):
            result = TemporalExtractor().from_external({"bad_act": bad})

        assert "bad_act" not in result
        assert any("skip" in msg.lower() for msg in caplog.messages)

    def test_partial_coverage_only_specified_activities_returned(self):
        """
        External durations may cover only a subset of process activities.
        The result must contain exactly the specified activities, no more.
        """
        ext = {
            "registration": ExternalDuration(30.0, 120.0),
            "triage":       ExternalDuration(60.0, 300.0),
        }
        result = TemporalExtractor().from_external(ext)
        assert set(result.keys()) == {"registration", "triage"}

    def test_empty_input_returns_empty_dict(self):
        """Passing an empty dict must return an empty dict without errors."""
        assert TemporalExtractor().from_external({}) == {}


# ===========================================================================
# TemporalExtractor.extract — priority and fallback logic
# ===========================================================================

class TestExtract:
    def test_external_overrides_lifecycle_for_same_activity(self):
        """
        If an activity appears both in external_durations and has lifecycle duration
        in the pn_log, the external bounds must win.
        """
        ex = TraceExecution("c1", [_step_with_duration("scan", 200.0)])
        ext = {"scan": ExternalDuration(min_duration=50.0, max_duration=150.0)}
        result = TemporalExtractor().extract(
            external_durations=ext,
            petri_net_log=_pn_log(ex),
        )
        assert result["scan"].source == "external"
        assert result["scan"].effective_min == pytest.approx(50.0)
        assert result["scan"].effective_max == pytest.approx(150.0)

    def test_lifecycle_preferred_over_inter_event(self):
        """
        An activity with both lifecycle duration and inter-event gap must use
        the lifecycle source (higher confidence).
        """
        # exam has duration_seconds (lifecycle) AND a timestamp (inter-event gap to discharge)
        exam_step = FiringStep(
            transition=_transition("t_exam", "exam"),
            activity_name="exam",
            is_tau=False,
            from_places=set(),
            attributes={"time:timestamp": ts(0)},
            duration_seconds=300.0,
        )
        discharge_step = _pn_step("discharge", 50)
        ex = TraceExecution("c1", [exam_step, discharge_step])
        result = TemporalExtractor().extract(petri_net_log=_pn_log(ex))
        assert result["exam"].source == "lifecycle"
        assert result["exam"].mean == pytest.approx(300.0)

    def test_inter_event_used_when_no_lifecycle_data_and_fallback_enabled(self):
        """
        An activity with no duration_seconds falls back to inter_event estimation
        from petri_net_log when fallback_to_inter_event=True (the default).
        """
        ex = TraceExecution("c1", [_pn_step("triage", 0), _pn_step("blood_test", 90)])
        result = TemporalExtractor().extract(petri_net_log=_pn_log(ex))
        assert "triage" in result
        assert result["triage"].source == "inter_event"

    def test_fallback_disabled_excludes_inter_event_only_activities(self):
        """
        With fallback_to_inter_event=False, activities with only inter-event data
        must be absent from the result.
        """
        ex = TraceExecution("c1", [_pn_step("triage", 0), _pn_step("blood_test", 90)])
        result = TemporalExtractor().extract(
            fallback_to_inter_event=False,
            petri_net_log=_pn_log(ex),
        )
        assert "triage" not in result

    def test_external_only_activity_appears_even_without_pn_log(self):
        """
        An activity in external_durations that has no pn_log data must still appear.
        """
        ext = {"virtual_activity": ExternalDuration(10.0, 20.0)}
        result = TemporalExtractor().extract(external_durations=ext)
        assert "virtual_activity" in result
        assert result["virtual_activity"].source == "external"

    def test_result_contains_union_of_all_strategy_outputs(self):
        """
        extract() must return data for every activity covered by at least one strategy.
        scan: lifecycle (duration_seconds set), triage: inter-event only, lab: external.
        """
        scan_step = FiringStep(
            transition=_transition("t_scan", "scan"),
            activity_name="scan",
            is_tau=False,
            from_places=set(),
            attributes={"time:timestamp": ts(0)},
            duration_seconds=100.0,
        )
        triage_step = _pn_step("triage", 100)
        discharge_step = _pn_step("discharge", 200)
        ex = TraceExecution("c1", [scan_step, triage_step, discharge_step])
        ext = {"lab": ExternalDuration(30.0, 60.0)}
        result = TemporalExtractor().extract(
            external_durations=ext,
            petri_net_log=_pn_log(ex),
        )
        assert result["scan"].source == "lifecycle"
        assert result["lab"].source == "external"
        assert result["triage"].source == "inter_event"


# ===========================================================================
# Helpers for PetriNetLog-based tests
# ===========================================================================

def _pn_step(activity: str, offset: int, is_tau: bool = False) -> FiringStep:
    """FiringStep with a time:timestamp at base + offset seconds."""
    t = _transition(f"t_{activity}", None if is_tau else activity)
    attrs = {} if is_tau else {"time:timestamp": ts(offset)}
    return FiringStep(
        transition=t,
        activity_name=activity,
        is_tau=is_tau,
        from_places=set(),
        attributes=attrs,
    )


def _pn_log(*executions: TraceExecution) -> PetriNetLog:
    return PetriNetLog(
        executions=list(executions),
        net=PetriNet("test"),
        initial_marking=Marking(),
        final_marking=Marking(),
    )


# ===========================================================================
# TemporalExtractor.estimate_from_inter_event_times_pn
# ===========================================================================

class TestEstimateFromInterEventTimesPn:
    def test_consecutive_labeled_steps_produce_correct_gap(self):
        """A[@0] → B[@60]: A contributes a 60s gap, B has no successor."""
        ex = TraceExecution("c1", [_pn_step("a", 0), _pn_step("b", 60)])
        result = TemporalExtractor().estimate_from_inter_event_times_pn(_pn_log(ex))

        assert "a" in result
        assert result["a"].mean == pytest.approx(60.0)
        assert "b" not in result

    def test_tau_steps_between_labeled_steps_are_skipped(self):
        """A[@0] → tau → B[@60]: tau is invisible, gap for A must be 60s."""
        ex = TraceExecution("c1", [
            _pn_step("a", 0),
            _pn_step("tau_1", 0, is_tau=True),
            _pn_step("b", 60),
        ])
        result = TemporalExtractor().estimate_from_inter_event_times_pn(_pn_log(ex))

        assert result["a"].mean == pytest.approx(60.0)
        assert "tau_1" not in result

    def test_last_labeled_step_excluded(self):
        """The last labeled step in every execution has no successor and is omitted."""
        ex = TraceExecution("c1", [_pn_step("a", 0), _pn_step("b", 60)])
        result = TemporalExtractor().estimate_from_inter_event_times_pn(_pn_log(ex))

        assert "b" not in result

    def test_multiple_executions_accumulate(self):
        """Two executions each contributing 60s gap for A → mean stays 60s, count 2."""
        ex1 = TraceExecution("c1", [_pn_step("a", 0), _pn_step("b", 60)])
        ex2 = TraceExecution("c2", [_pn_step("a", 0), _pn_step("b", 60)])
        result = TemporalExtractor().estimate_from_inter_event_times_pn(_pn_log(ex1, ex2))

        assert result["a"].count == 2
        assert result["a"].mean == pytest.approx(60.0)

    def test_negative_gap_is_discarded_with_warning(self, caplog):
        """ts[i+1] < ts[i] → skip with warning."""
        ex = TraceExecution("c1", [_pn_step("x", 500), _pn_step("y", 100)])
        with caplog.at_level(logging.WARNING, logger="parsing.temporal_extractor"):
            result = TemporalExtractor().estimate_from_inter_event_times_pn(_pn_log(ex))

        assert "x" not in result
        assert any("negative" in msg.lower() for msg in caplog.messages)

    def test_missing_timestamp_is_skipped_with_warning(self, caplog):
        """Step with no time:timestamp in attributes → skipped with warning."""
        t = _transition("t_a", "a")
        step_no_ts = FiringStep(
            transition=t, activity_name="a", is_tau=False, from_places=set(), attributes={}
        )
        step_b = _pn_step("b", 60)
        ex = TraceExecution("c1", [step_no_ts, step_b])

        with caplog.at_level(logging.WARNING, logger="parsing.temporal_extractor"):
            result = TemporalExtractor().estimate_from_inter_event_times_pn(_pn_log(ex))

        assert "a" not in result
        assert any("skipped" in msg.lower() for msg in caplog.messages)

    def test_source_label_is_inter_event(self):
        """All entries must carry source='inter_event'."""
        ex = TraceExecution("c1", [_pn_step("a", 0), _pn_step("b", 90)])
        result = TemporalExtractor().estimate_from_inter_event_times_pn(_pn_log(ex))

        assert result["a"].source == "inter_event"

    def test_empty_pn_log_returns_empty_dict(self):
        result = TemporalExtractor().estimate_from_inter_event_times_pn(_pn_log())
        assert result == {}


# ===========================================================================
# TemporalExtractor.extract — petri_net_log parameter
# ===========================================================================

class TestExtractWithPetriNetLog:
    def test_pn_log_used_for_inter_event_when_provided(self):
        """When petri_net_log is provided, inter-event gaps come from it."""
        ex = TraceExecution("c1", [_pn_step("triage", 0), _pn_step("discharge", 90)])
        result = TemporalExtractor().extract(petri_net_log=_pn_log(ex))

        assert "triage" in result
        assert result["triage"].source == "inter_event"
        assert result["triage"].mean == pytest.approx(90.0)

    def test_lifecycle_still_wins_over_pn_inter_event(self):
        """
        When a step has duration_seconds (lifecycle), it wins over the inter-event
        gap derived from the same step's timestamp.
        """
        scan_step = FiringStep(
            transition=_transition("t_scan", "scan"),
            activity_name="scan",
            is_tau=False,
            from_places=set(),
            attributes={"time:timestamp": ts(0)},
            duration_seconds=200.0,
        )
        discharge_step = _pn_step("discharge", 50)
        ex = TraceExecution("c1", [scan_step, discharge_step])
        result = TemporalExtractor().extract(petri_net_log=_pn_log(ex))

        assert result["scan"].source == "lifecycle"
        assert result["scan"].mean == pytest.approx(200.0)
