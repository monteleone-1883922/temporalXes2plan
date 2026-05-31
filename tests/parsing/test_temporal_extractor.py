"""
Tests for parsing.temporal_extractor.

Covers the three duration extraction strategies and the data model validation.
All event logs are built in-memory via conftest helpers — no XES files are loaded.

Strategy overview:
  - extract_from_lifecycle:       pairs start/complete lifecycle events per activity.
  - estimate_from_inter_event_times: uses the gap between consecutive complete events.
  - from_external:                accepts user-supplied min/max bounds directly.
  - extract:                      combines all three with priority external > lifecycle > inter_event.
"""
import math
import logging

import pytest

from tests.helpers import make_event, make_trace, make_log
from parsing.temporal_extractor import (
    ActionDurationStats,
    ExternalDuration,
    TemporalExtractor,
    _stats_to_duration,
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
# TemporalExtractor.extract_from_lifecycle — Case 1
# ===========================================================================

class TestExtractFromLifecycle:
    def test_single_start_complete_pair_returns_correct_delta(self):
        """
        One start at t=0 and one complete at t=300 for activity 'visit'.
        The extractor must report a single 300-second observation.
        """
        log = make_log(
            make_trace(
                make_event("visit", lifecycle="start", offset=0),
                make_event("visit", lifecycle="complete", offset=300),
            )
        )
        result = TemporalExtractor(log).extract_from_lifecycle()
        assert "visit" in result
        stats = result["visit"]
        assert stats.source == "lifecycle"
        assert stats.count == 1
        assert stats.mean == pytest.approx(300.0)

    def test_multiple_pairs_aggregate_correctly(self):
        """
        Three traces each with a start/complete pair for 'test' at durations
        60, 120, 180 seconds.  Mean must be 120, stats must cover all three.
        """
        log = make_log(
            make_trace(
                make_event("test", lifecycle="start", offset=0),
                make_event("test", lifecycle="complete", offset=60),
                case_id="c1",
            ),
            make_trace(
                make_event("test", lifecycle="start", offset=0),
                make_event("test", lifecycle="complete", offset=120),
                case_id="c2",
            ),
            make_trace(
                make_event("test", lifecycle="start", offset=0),
                make_event("test", lifecycle="complete", offset=180),
                case_id="c3",
            ),
        )
        stats = TemporalExtractor(log).extract_from_lifecycle()["test"]
        assert stats.count == 3
        assert stats.mean == pytest.approx(120.0)
        assert stats.observed_min == pytest.approx(60.0)
        assert stats.observed_max == pytest.approx(180.0)

    def test_fifo_matching_for_repeated_activity_in_same_case(self):
        """
        When the same activity appears twice in a single case (start1, start2,
        complete1, complete2) the FIFO rule must pair start1 with complete1.
        Both durations must be captured, not just the second pair.
        """
        log = make_log(
            make_trace(
                make_event("scan", lifecycle="start", offset=0),
                make_event("scan", lifecycle="start", offset=10),
                make_event("scan", lifecycle="complete", offset=100),   # pairs with start@0 → 100s
                make_event("scan", lifecycle="complete", offset=120),   # pairs with start@10 → 110s
            )
        )
        stats = TemporalExtractor(log).extract_from_lifecycle()["scan"]
        assert stats.count == 2

    def test_complete_without_matching_start_is_skipped(self, caplog):
        """
        A 'complete' event that has no preceding 'start' must be discarded.
        A WARNING about orphan completes must appear in the logs.
        """
        log = make_log(
            make_trace(make_event("orphan", lifecycle="complete", offset=100))
        )
        with caplog.at_level(logging.WARNING, logger="parsing.temporal_extractor"):
            result = TemporalExtractor(log).extract_from_lifecycle()
        assert "orphan" not in result
        assert any("orphan" in msg or "no matching start" in msg.lower() for msg in caplog.messages)

    def test_negative_delta_is_discarded_with_warning(self, caplog):
        """
        When complete timestamp < start timestamp (data quality issue), the pair
        must be discarded with a WARNING log entry.
        """
        log = make_log(
            make_trace(
                make_event("check", lifecycle="start", offset=500),
                make_event("check", lifecycle="complete", offset=100),  # complete before start
            )
        )
        with caplog.at_level(logging.WARNING, logger="parsing.temporal_extractor"):
            result = TemporalExtractor(log).extract_from_lifecycle()
        assert "check" not in result
        assert any("negative" in msg.lower() for msg in caplog.messages)

    def test_event_missing_activity_name_is_counted_as_skipped(self, caplog):
        """
        An event without a 'concept:name' attribute must be skipped and the
        total count of skipped events must be logged at WARNING level.
        """
        bad_event = make_event("", lifecycle="start", offset=0)   # sanitize_name("") → ""
        good_event = make_event("valid", lifecycle="complete", offset=100)
        log = make_log(make_trace(bad_event, good_event))

        with caplog.at_level(logging.WARNING, logger="parsing.temporal_extractor"):
            TemporalExtractor(log).extract_from_lifecycle()
        assert any("skipped" in msg.lower() for msg in caplog.messages)

    def test_log_without_lifecycle_returns_empty_dict(self):
        """
        A log with no lifecycle:transition attributes at all means no start/complete
        pairs can be found — the result must be an empty dict.
        """
        log = make_log(
            make_trace(
                make_event("A", offset=0),
                make_event("B", offset=60),
            )
        )
        assert TemporalExtractor(log).extract_from_lifecycle() == {}

    def test_mixed_timezone_aware_and_naive_timestamps_do_not_raise(self):
        """
        pm4py logs from different XES sources may have timezone-aware or naive
        datetimes.  Mixing them in the same log must not raise TypeError.
        The _delta_seconds helper normalises both sides before subtraction.
        """
        log = make_log(
            make_trace(
                make_event("task", lifecycle="start", offset=0, tz_aware=True),
                make_event("task", lifecycle="complete", offset=200, tz_aware=False),
            )
        )
        # Should not raise
        result = TemporalExtractor(log).extract_from_lifecycle()
        assert "task" in result


# ===========================================================================
# TemporalExtractor.estimate_from_inter_event_times — Case 3
# ===========================================================================

class TestEstimateFromInterEventTimes:
    def test_three_event_trace_produces_two_gaps(self):
        """
        A trace [A@0, B@60, C@180] produces two inter-event gaps:
          A → 60 s  (B starts 60 s after A)
          B → 120 s (C starts 120 s after B)
        The last event C gets no gap (no successor).
        """
        log = make_log(
            make_trace(
                make_event("A", offset=0),
                make_event("B", offset=60),
                make_event("C", offset=180),
            )
        )
        result = TemporalExtractor(log).estimate_from_inter_event_times()
        assert result["a"].mean == pytest.approx(60.0)
        assert result["b"].mean == pytest.approx(120.0)
        assert "c" not in result   # last event — no successor

    def test_single_event_trace_produces_no_data(self):
        """A trace with only one event has no successor, so no gaps can be computed."""
        log = make_log(make_trace(make_event("alone", offset=0)))
        result = TemporalExtractor(log).estimate_from_inter_event_times()
        assert "alone" not in result

    def test_start_events_are_filtered_out_on_lifecycle_log(self):
        """
        When the log contains start/complete pairs (full_lifecycle_log), 'start'
        events must be excluded before computing inter-event gaps to avoid
        spurious near-zero gaps between start:X and complete:X.

        Net: start:A@0, complete:A@300, start:B@300, complete:B@600.
        After filtering 'start' events: [complete:A@300, complete:B@600].
        Expected single gap for A: 300 s (B - A complete timestamps).
        """
        log = make_log(
            make_trace(
                make_event("A", lifecycle="start", offset=0),
                make_event("A", lifecycle="complete", offset=300),
                make_event("B", lifecycle="start", offset=300),
                make_event("B", lifecycle="complete", offset=600),
            )
        )
        result = TemporalExtractor(log).estimate_from_inter_event_times()
        # After filtering starts: [A@300, B@600] → A contributes 300 s gap
        assert "a" in result
        assert result["a"].mean == pytest.approx(300.0)

    def test_negative_gap_is_discarded_with_warning(self, caplog):
        """
        Events out of chronological order (t[i+1] < t[i]) produce a negative gap.
        These must be discarded and a WARNING must be emitted.
        """
        log = make_log(
            make_trace(
                make_event("X", offset=500),
                make_event("Y", offset=100),  # before X — negative gap
            )
        )
        with caplog.at_level(logging.WARNING, logger="parsing.temporal_extractor"):
            result = TemporalExtractor(log).estimate_from_inter_event_times()
        assert "x" not in result
        assert any("negative" in msg.lower() for msg in caplog.messages)

    def test_event_missing_timestamp_is_counted_as_skipped(self, caplog):
        """
        An event with no time:timestamp must be skipped and the count logged
        at WARNING level.
        """
        bad = make_event("Z", offset=0)
        del bad["time:timestamp"]           # remove the timestamp key entirely
        good = make_event("W", offset=60)
        log = make_log(make_trace(bad, good))

        with caplog.at_level(logging.WARNING, logger="parsing.temporal_extractor"):
            TemporalExtractor(log).estimate_from_inter_event_times()
        assert any("skipped" in msg.lower() for msg in caplog.messages)

    def test_source_label_is_inter_event(self):
        """All entries returned must carry source='inter_event'."""
        log = make_log(
            make_trace(make_event("P", offset=0), make_event("Q", offset=90))
        )
        result = TemporalExtractor(log).estimate_from_inter_event_times()
        assert result["p"].source == "inter_event"


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
        result = TemporalExtractor(make_log()).from_external(ext)

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
            result = TemporalExtractor(make_log()).from_external({"bad_act": bad})

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
        result = TemporalExtractor(make_log()).from_external(ext)
        assert set(result.keys()) == {"registration", "triage"}

    def test_empty_input_returns_empty_dict(self):
        """Passing an empty dict must return an empty dict without errors."""
        assert TemporalExtractor(make_log()).from_external({}) == {}


# ===========================================================================
# TemporalExtractor.extract — priority and fallback logic
# ===========================================================================

class TestExtract:
    def _lifecycle_log(self) -> object:
        """Helper: log with one lifecycle pair for 'scan' (200 s)."""
        return make_log(
            make_trace(
                make_event("scan", lifecycle="start", offset=0),
                make_event("scan", lifecycle="complete", offset=200),
            )
        )

    def _inter_event_log(self) -> object:
        """Helper: complete-only log for 'triage' → 'blood_test' (90 s gap)."""
        return make_log(
            make_trace(make_event("triage", offset=0), make_event("blood_test", offset=90))
        )

    def test_external_overrides_lifecycle_for_same_activity(self):
        """
        If an activity appears both in external_durations and in the lifecycle log,
        the external bounds must win.
        """
        log = self._lifecycle_log()
        ext = {"scan": ExternalDuration(min_duration=50.0, max_duration=150.0)}
        result = TemporalExtractor(log).extract(external_durations=ext)

        assert result["scan"].source == "external"
        assert result["scan"].effective_min == pytest.approx(50.0)
        assert result["scan"].effective_max == pytest.approx(150.0)

    def test_lifecycle_preferred_over_inter_event(self):
        """
        An activity with both lifecycle pairs and inter-event data must use
        the lifecycle source (higher confidence).
        """
        log = make_log(
            make_trace(
                make_event("exam", lifecycle="start", offset=0),
                make_event("exam", lifecycle="complete", offset=300),
                make_event("exam", offset=300),         # also present as complete-only event
                make_event("followup", offset=400),
            )
        )
        result = TemporalExtractor(log).extract()
        assert result["exam"].source == "lifecycle"

    def test_inter_event_used_when_no_lifecycle_data_and_fallback_enabled(self):
        """
        An activity with no lifecycle pairs falls back to inter_event estimation
        when fallback_to_inter_event=True (the default).
        """
        log = self._inter_event_log()
        result = TemporalExtractor(log).extract()
        # 'triage' has a gap (to blood_test) but no lifecycle data
        assert "triage" in result
        assert result["triage"].source == "inter_event"

    def test_fallback_disabled_excludes_inter_event_only_activities(self):
        """
        With fallback_to_inter_event=False, activities that only have inter-event
        data must be absent from the result.
        """
        log = self._inter_event_log()
        result = TemporalExtractor(log).extract(fallback_to_inter_event=False)
        # No lifecycle pairs → 'triage' must not appear
        assert "triage" not in result

    def test_external_only_activity_appears_even_without_log_data(self):
        """
        An activity specified in external_durations that does not exist in the log
        must still appear in the result (the user explicitly provided bounds for it).
        """
        log = make_log()   # empty log — no events
        ext = {"virtual_activity": ExternalDuration(10.0, 20.0)}
        result = TemporalExtractor(log).extract(external_durations=ext)
        assert "virtual_activity" in result
        assert result["virtual_activity"].source == "external"

    def test_result_contains_union_of_all_strategy_outputs(self):
        """
        extract() must return data for every activity covered by at least one strategy.
        """
        # lifecycle data for 'scan', inter-event data for 'triage'
        log = make_log(
            make_trace(
                make_event("scan", lifecycle="start", offset=0),
                make_event("scan", lifecycle="complete", offset=100),
                make_event("triage", offset=100),
                make_event("discharge", offset=200),
            )
        )
        ext = {"lab": ExternalDuration(30.0, 60.0)}
        result = TemporalExtractor(log).extract(external_durations=ext)

        assert result["scan"].source == "lifecycle"
        assert result["lab"].source == "external"
        # 'triage' comes from inter_event (gap to discharge)
        assert result["triage"].source == "inter_event"
