"""End-to-end test for network_search.runner — docs/network_improvement_loop_plan.md
§9 phases 4/5: validates the full chain Parser -> DomainBuilder ->
TraceReplayer -> score on a real (synthetic) log, and the Successive
Halving promotion mechanism (phase 5).

Marked @pytest.mark.integration (skipped by default, run with -m integration)
since it runs the real discovery/mining pipeline several times — same
convention as tests/parsing/test_parser.py, whose synthetic XOR log this
test reuses.
"""
import json
import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import optuna
import pm4py
import pytest
from pm4py.objects.log.obj import Event, EventLog, Trace

import core_utils as utils
from models import AnalysisConfig
from network_search.runner import TrialRecord, find_best_config, save_trial_records
from network_search.scoring import ScoreWeights, TrialMetrics, combine

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Synthetic log — same shape as tests/parsing/test_parser.py's _make_xor_log:
# a clear XOR split driven by the 'risk' attribute.
# ---------------------------------------------------------------------------

def _ts(offset_seconds: int = 0) -> datetime:
    base = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_seconds)


def _event(name: str, offset: int = 0, **attrs) -> Event:
    e = Event()
    e["concept:name"] = name
    e["time:timestamp"] = _ts(offset)
    for k, v in attrs.items():
        e[k] = v
    return e


def _trace(case_id: str, events: List[Event]) -> Trace:
    t = Trace()
    t.attributes["concept:name"] = case_id
    for e in events:
        t.append(e)
    return t


def _make_xor_log() -> EventLog:
    log = EventLog()
    for i in range(60):
        log.append(_trace(f"high_{i}", [
            _event("Register", 0, risk="high"),
            _event("Approve", 10, outcome="approved"),
            _event("Close", 20, status="done"),
        ]))
    for i in range(60):
        log.append(_trace(f"low_{i}", [
            _event("Register", 0, risk="low"),
            _event("Reject", 10, outcome="rejected"),
            _event("Close", 20, status="done"),
        ]))
    return log


@pytest.fixture(scope="module")
def xes_path(tmp_path_factory) -> str:
    tmp = tmp_path_factory.mktemp("runner_integration")
    path = str(tmp / "xor_log.xes")
    pm4py.write_xes(_make_xor_log(), path)
    return path


@pytest.fixture(scope="module")
def split_result(xes_path):
    with utils.split(xes_path, test_pct=0.2, seed=0) as result:
        yield result


@pytest.fixture(scope="module")
def search_result(split_result) -> Tuple[AnalysisConfig, List[TrialRecord]]:
    return find_best_config(
        training_log_path=str(split_result.train_path),
        test_traces=split_result.test_cases,
        n_trials=3,
    )


@pytest.fixture(scope="module")
def best_config(search_result) -> AnalysisConfig:
    return search_result[0]


@pytest.fixture(scope="module")
def records(search_result) -> List[TrialRecord]:
    return search_result[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRunnerEndToEnd:
    def test_returns_one_record_per_trial(self, records):
        assert len(records) == 3

    def test_every_record_is_promoted(self, records):
        # With only 3 trials and the default min_promoted=15, the round
        # never has enough trials to demote anyone -- see
        # TestSuccessiveHalvingPromotion below for actual demotion.
        # stopped_early legitimately varies per trial (phase 6, see
        # TestSuccessiveHalvingPromotion.test_early_stopping_only_cuts_off_losing_trials
        # for the correctness property).
        assert all(r.promoted_to_rung2 for r in records)

    def test_every_record_has_a_finite_score(self, records):
        for r in records:
            assert r.score is not None
            assert math.isfinite(r.score)

    def test_reproducibility_score_is_a_valid_fraction(self, records):
        for r in records:
            assert r.metrics.reproducibility_score is not None
            assert 0.0 <= r.metrics.reproducibility_score <= 1.0

    def test_n_test_replayed_is_bounded_by_the_test_set_size(self, records):
        # test_pct=0.2 on 120 traces -> 24 test traces. Phase 6's early
        # stopping can cut a promoted trial's replay short, so
        # n_test_replayed is <= 24 (and == 24 exactly when stopped_early is
        # False, since then nothing interrupted the full replay).
        for r in records:
            assert 0 < r.metrics.n_test_replayed <= 24
            if not r.stopped_early:
                assert r.metrics.n_test_replayed == 24

    def test_coverage_is_a_valid_fraction(self, records):
        for r in records:
            assert 0.0 <= r.metrics.coverage <= 1.0

    def test_trial_numbers_are_distinct(self, records):
        assert len({r.trial_number for r in records}) == len(records)

    def test_configs_vary_across_trials(self, records):
        # Sanity check that suggest_config is actually wired to Optuna here
        # (not silently returning a constant default config every time).
        assert len({r.config.probability_min_samples for r in records}) > 1 or \
               len({r.config.xor_prune_threshold for r in records}) > 1

    def test_best_config_is_the_highest_scoring_trial(self, best_config, records):
        best_record = max(records, key=lambda r: r.score)
        assert best_config == best_record.config

    def test_best_config_is_an_analysis_config(self, best_config):
        assert isinstance(best_config, AnalysisConfig)


@pytest.mark.integration
class TestSaveTrialRecords:
    def test_writes_one_json_line_per_record(self, records, tmp_path):
        out_path = tmp_path / "network_search_trials.jsonl"
        save_trial_records(records, str(out_path))
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(records)

    def test_each_line_is_valid_json_with_expected_keys(self, records, tmp_path):
        out_path = tmp_path / "network_search_trials.jsonl"
        save_trial_records(records, str(out_path))
        for line in out_path.read_text(encoding="utf-8").splitlines():
            parsed = json.loads(line)
            assert {"trial_number", "config", "metrics", "score",
                    "promoted_to_rung2", "stopped_early"} <= parsed.keys()

    def test_ignored_attributes_set_serialized_as_sorted_list(self, records, tmp_path):
        out_path = tmp_path / "network_search_trials.jsonl"
        save_trial_records(records, str(out_path))
        first = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
        ignored = first["config"]["ignored_attributes"]
        assert isinstance(ignored, list)
        assert ignored == sorted(ignored)


# ---------------------------------------------------------------------------
# Successive Halving (phase 5) — a round big enough, with a low enough
# min_promoted, to actually demote some trials rather than trivially
# promoting everyone.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sh_search_result(split_result) -> Tuple[AnalysisConfig, List[TrialRecord]]:
    return find_best_config(
        training_log_path=str(split_result.train_path),
        test_traces=split_result.test_cases,
        n_trials=10,
        round_size=10, promotion_fraction=0.3, min_promoted=2,
    )


@pytest.fixture(scope="module")
def sh_records(sh_search_result) -> List[TrialRecord]:
    return sh_search_result[1]


@pytest.mark.integration
class TestSuccessiveHalvingPromotion:
    def test_returns_one_record_per_trial(self, sh_records):
        assert len(sh_records) == 10

    def test_promotes_the_expected_count(self, sh_records):
        # round_size=10, promotion_fraction=0.3 -> round(10*0.3) = 3,
        # above min_promoted=2, so exactly 3 trials should be promoted.
        promoted = [r for r in sh_records if r.promoted_to_rung2]
        assert len(promoted) == 3

    def test_discarded_trials_have_no_reproducibility_and_were_not_replayed(self, sh_records):
        discarded = [r for r in sh_records if not r.promoted_to_rung2]
        assert len(discarded) == 7
        for r in discarded:
            assert r.metrics.reproducibility_score is None
            assert r.metrics.n_test_replayed == 0

    def test_promoted_trials_were_replayed_at_least_partially(self, sh_records):
        # test_pct=0.2 on 120 traces -> 24 test traces. Phase 6's exact
        # early stopping can legitimately cut a promoted trial's replay
        # short (n_test_replayed < 24) once it can no longer beat the best
        # score found so far -- see test_early_stopping_never_discards_a_
        # trial_that_could_have_won for the correctness property that makes
        # this safe.
        promoted = [r for r in sh_records if r.promoted_to_rung2]
        for r in promoted:
            assert r.metrics.reproducibility_score is not None
            assert 0 < r.metrics.n_test_replayed <= 24
            if not r.stopped_early:
                assert r.metrics.n_test_replayed == 24

    def test_early_stopping_never_discards_a_trial_that_could_have_won(self, sh_records):
        # Exact-bound property (docs/network_improvement_loop.md §5.4.2):
        # for a stopped_early trial, its best-possible completion (every
        # remaining test trace succeeding) must not exceed the highest
        # score any other trial in this search actually achieved --
        # otherwise the bound would have wrongly cut off a potential winner.
        best_actual = max(r.score for r in sh_records if r.score is not None)
        for r in sh_records:
            if not r.stopped_early:
                continue
            n_total = 24
            n_success = round(r.metrics.reproducibility_score * n_total)
            n_remaining = n_total - r.metrics.n_test_replayed
            best_possible_repro = (n_success + n_remaining) / n_total
            best_possible = combine(
                TrialMetrics(r.metrics.fallback_score, r.metrics.duplication_penalty,
                             best_possible_repro, n_total, r.metrics.coverage),
                ScoreWeights(),
            )
            assert best_possible <= best_actual + 1e-9

    def test_every_record_still_has_a_finite_score(self, sh_records):
        for r in sh_records:
            assert r.score is not None
            assert math.isfinite(r.score)

    def test_promoted_trials_are_the_best_by_partial_score(self, sh_records):
        # Every promoted trial's fallback_score/duplication_penalty-only
        # partial score must be >= every discarded trial's -- that is what
        # "promote the top fraction" means (docs/network_improvement_loop_plan.md §7.c).
        def partial(r):
            m = TrialMetrics(r.metrics.fallback_score, r.metrics.duplication_penalty, None, 0, r.metrics.coverage)
            return combine(m, ScoreWeights())

        promoted = [r for r in sh_records if r.promoted_to_rung2]
        discarded = [r for r in sh_records if not r.promoted_to_rung2]
        min_promoted_partial = min(partial(r) for r in promoted)
        max_discarded_partial = max(partial(r) for r in discarded)
        assert min_promoted_partial >= max_discarded_partial

    def test_best_config_can_come_from_a_promoted_trial_only(self, sh_search_result, sh_records):
        best_config, _ = sh_search_result
        best_record = max(sh_records, key=lambda r: r.score)
        assert best_config == best_record.config
        assert best_record.promoted_to_rung2


# ---------------------------------------------------------------------------
# base_config propagation (GUI optimizer toggle: fields the search doesn't
# tune must come from the caller's base config, not AnalysisConfig()'s
# hardcoded default) -- see search_space.py's TestBaseConfigOverride for the
# suggest_config()-level test; this checks find_best_config() threads it
# through its ask/tell loop correctly.
# ---------------------------------------------------------------------------

CUSTOM_BASE_CONFIG = AnalysisConfig(
    replay_engine="alignments",
    snapshot_dir="custom_snapshot_dir",
    ignored_attributes={"custom_attr"},
)


@pytest.fixture(scope="module")
def base_config_search_result(split_result) -> Tuple[AnalysisConfig, List[TrialRecord]]:
    return find_best_config(
        training_log_path=str(split_result.train_path),
        test_traces=split_result.test_cases,
        n_trials=3,
        base_config=CUSTOM_BASE_CONFIG,
    )


@pytest.mark.integration
class TestBaseConfigPropagation:
    def test_every_trial_config_carries_base_config_untouched_fields(self, base_config_search_result):
        _, records = base_config_search_result
        for record in records:
            assert record.config.replay_engine == CUSTOM_BASE_CONFIG.replay_engine
            assert record.config.snapshot_dir == CUSTOM_BASE_CONFIG.snapshot_dir
            assert record.config.ignored_attributes == CUSTOM_BASE_CONFIG.ignored_attributes

    def test_best_config_also_carries_base_config_untouched_fields(self, base_config_search_result):
        best_config, _ = base_config_search_result
        assert best_config.replay_engine == CUSTOM_BASE_CONFIG.replay_engine
        assert best_config.snapshot_dir == CUSTOM_BASE_CONFIG.snapshot_dir
