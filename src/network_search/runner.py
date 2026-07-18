"""Search-loop core: find_best_config().

Performs **no I/O of any kind** — it never loads or splits an event log.
`find_best_config()` takes `training_log_path` (a plain path string handed
straight to `Parser`, which always needs one — that's intrinsic to the
pipeline, not something this module adds on top) and `test_traces`
(already-loaded pm4py `Trace` objects) as parameters. The caller is
responsible for loading the log, validating it, splitting it (via
`core_utils.split()`), and computing `original_activity_names` (via
`core_utils.activity_names()`) before calling in.

Also deliberately does *not* build the final network, persist
current.json, or expose a CLI — docs/network_improvement_loop_plan.md §0
(decided once this module reached phase 4 end-to-end): network_search is
not a standalone tool, it runs on every full-log import at the same point
the network is normally built with a fixed AnalysisConfig, i.e. from
Pipeline.run() (src/pipeline.py) — the one entry point already shared by
the CLI (src/main.py) and the web import flow
(web/api.py::_pipeline_thread). Pipeline.run() loads/splits the log, calls
find_best_config() to get the winning config, then builds the final
network on the *complete* log with its own existing steps
(Parse/Encode/Serialize/Persist) — the split handed to find_best_config()
exists only to score candidates during the search, not to hold anything
back from the final network.

This is phase 5 of docs/network_improvement_loop_plan.md §9: Successive
Halving is now built in — every round evaluates rung 1 (screening) for all
its trials, promotes only the best fraction to rung 2 (full replay), and
tells Optuna a partial score for the rest so TPE still learns from them
without paying for their replay (docs/network_improvement_loop_plan.md §7
points c-e).

Phase 6 (docs/network_improvement_loop.md §5.4.2, plan §7 point f) adds
exact early stopping *inside* rung 2: replay stops the moment
max_reachable_score() proves a trial cannot beat the best score found so
far, no matter how the remaining test traces replay.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, List, Optional, Set, Tuple

import optuna

import core_utils as utils
from encoding.domain_builder import DomainBuilder
from encoding.prepared_input import PreparedDomainInput
from models import AnalysisConfig, ParseResult
from parsing.xes_parser import (
    Parser,
    PreloadedLogAndNet,
    PreloadedReplay,
    load_and_discover,
    preload_replay,
)

from network_search.metrics import (
    compute_coverage,
    compute_duplication_score,
    compute_xor_score,
    compute_effect_score,
)
from network_search.scoring import ScoreWeights, TrialMetrics, combine, max_reachable_score
from network_search.search_space import suggest_config
from replay.trace_replayer import TraceReplayer

logger = utils.get_logger(__name__)


@dataclasses.dataclass
class TrialRecord:
    """One trial's full outcome — docs/network_improvement_loop_plan.md §3.

    For a trial not promoted to rung 2 (promoted_to_rung2=False), metrics
    carries reproducibility_score=None and n_test_replayed=0 — score is
    still set, to the rung-1-only partial score used to keep TPE informed
    (docs/network_improvement_loop_plan.md §7 point e). stopped_early is
    True when rung 2's replay was cut short by the exact early-stopping
    bound (docs/network_improvement_loop.md §5.4.2) — it can only be True
    when promoted_to_rung2 is also True.
    """

    trial_number: int
    config: AnalysisConfig
    metrics: TrialMetrics
    score: Optional[float]
    promoted_to_rung2: bool
    stopped_early: bool


@dataclasses.dataclass
class _Rung1Result:
    """Everything rung 1 produces for one trial — enough to either stop here
    (discarded) or continue into rung 2 (promoted) without recomputing
    anything already built."""

    trial: optuna.Trial
    config: AnalysisConfig
    parse_result: ParseResult
    prepared: PreparedDomainInput
    xor_score: float
    effect_score: float
    duplication_score: float
    coverage: float
    partial_score: float


def _evaluate_rung1(
    trial: optuna.Trial,
    training_log_path: str,
    weights: ScoreWeights,
    coverage_percentage: float,
    discovery_algorithm: str,
    original_activity_names: Optional[Set[str]],
    preloaded: PreloadedLogAndNet,
    preloaded_replay: PreloadedReplay,
    base_config: Optional[AnalysisConfig] = None,
) -> _Rung1Result:
    """Cheap phase: build the network, compute what's knowable without replay.

    partial_score uses combine() with reproducibility_score=None (treated
    as 0.0 by combine(), docs/network_improvement_loop.md §5.4.1) — it is
    both the ranking key for promotion and, for trials that end up
    discarded, the value reported to Optuna and stored as TrialRecord.score.
    """
    config = suggest_config(trial, base_config=base_config)
    parser = Parser(
        log_path=training_log_path,
        coverage_percentage=coverage_percentage,
        discovery_algorithm=discovery_algorithm,
        config=config,
        preloaded=preloaded,
        preloaded_replay=preloaded_replay,
    )
    parse_result = parser.parse_result
    prepared = DomainBuilder().build_prepared_input(parse_result, config=config)

    xor_score = compute_xor_score(parser.xor_screening, weights)
    effect_score = compute_effect_score(parser.effect_screening, weights)
    duplication_score = compute_duplication_score(prepared.transitions)
    coverage = (
        compute_coverage(parse_result, original_activity_names)
        if original_activity_names is not None else 1.0
    )
    partial_metrics = TrialMetrics(
        xor_score=xor_score,
        effect_score=effect_score,
        duplication_score=duplication_score,
        reproducibility_score=None,
        n_test_replayed=0,
        coverage=coverage,
        full_replayability_score=None,
    )
    partial_score = combine(partial_metrics, weights)

    return _Rung1Result(
        trial=trial,
        config=config,
        parse_result=parse_result,
        prepared=prepared,
        xor_score=xor_score,
        effect_score=effect_score,
        duplication_score=duplication_score,
        coverage=coverage,
        partial_score=partial_score,
    )


def _discarded_record(rung1: _Rung1Result) -> TrialRecord:
    """Not promoted to rung 2 — score stays the rung-1-only partial_score."""
    metrics = TrialMetrics(
        xor_score=rung1.xor_score,
        effect_score=rung1.effect_score,
        duplication_score=rung1.duplication_score,
        reproducibility_score=None,
        n_test_replayed=0,
        coverage=rung1.coverage,
        full_replayability_score=None,
    )

    logger.debug(
        "Trial %d: discarded at rung 1 (partial_score=%.4f, xor=%.4f, effect=%.4f, "
        "duplication=%.4f, coverage=%.4f)",
        rung1.trial.number, rung1.partial_score, rung1.xor_score, rung1.effect_score,
        rung1.duplication_score, rung1.coverage,
    )

    return TrialRecord(
        trial_number=rung1.trial.number,
        config=rung1.config,
        metrics=metrics,
        score=rung1.partial_score,
        promoted_to_rung2=False,
        stopped_early=False,
    )


def _evaluate_rung2(
    rung1: _Rung1Result,
    test_traces: List[Any],
    weights: ScoreWeights,
    best_score_so_far: float,
) -> TrialRecord:
    """Expensive phase, only for promoted trials: replay test traces one at a
    time, stopping early (docs/network_improvement_loop.md §5.4.2) the
    instant max_reachable_score() proves this trial cannot beat
    best_score_so_far even if every remaining trace replayed successfully —
    an exact bound, not a statistical estimate, so it never discards a trial
    that would have gone on to win.

    Each test trace is replayed with a single call to
    replay_evaluation_split(trace, n_prefix=0) rather than two separate
    replay passes. n_prefix=0 makes the cut point the very start of the
    trace (cut_idx=0 unconditionally), so the walked tail is the entire
    trace starting from the true initial marking/empty attributes —
    semantically identical to replay_full_trace's loose check
    (outcome.reached_end) — while additionally verifying
    outcome.matches_expected_final_state (the real log-observed final
    attribute assignment) in the same guarded/backtracking walk. This is
    the "full replayability" signal: reached_end AND
    matches_expected_final_state (mirrors evaluation.trace_sampler's
    identical boolean combination for is_replayable).
    """
    replayer = TraceReplayer(rung1.prepared, rung1.parse_result.petri_net_model, rung1.config)

    n_success = 0
    n_full_success = 0
    n_processed = 0
    stopped_early = False
    for trace in test_traces:
        outcome = replayer.replay_evaluation_split(trace, n_prefix=0)
        n_processed += 1
        if outcome.reached_end:
            n_success += 1
            if outcome.matches_expected_final_state:
                n_full_success += 1
        bound = max_reachable_score(
            n_success, n_full_success, n_processed, len(test_traces),
            rung1.xor_score, rung1.effect_score, rung1.duplication_score, weights,
        )
        if bound <= best_score_so_far:
            stopped_early = True
            break

    # n_success / len(test_traces) (not / n_processed) even when interrupted
    # — this trial is already known to lose, see
    # docs/network_improvement_loop_plan.md §7's note; the value recorded
    # here is indicative/for debugging only, nothing downstream depends on it.
    reproducibility_score = n_success / len(test_traces) if test_traces else 0.0
    full_replayability_score = n_full_success / len(test_traces) if test_traces else 0.0

    metrics = TrialMetrics(
        xor_score=rung1.xor_score,
        effect_score=rung1.effect_score,
        duplication_score=rung1.duplication_score,
        reproducibility_score=reproducibility_score,
        n_test_replayed=n_processed,
        coverage=rung1.coverage,
        full_replayability_score=full_replayability_score,
    )
    score = combine(metrics, weights)

    logger.info(
        "Trial %d: score=%.4f (xor=%.4f, effect=%.4f, duplication=%.4f, reproducibility=%.4f, "
        "full_replayability=%.4f, coverage=%.4f, stopped_early=%s, n_test_replayed=%d/%d)",
        rung1.trial.number, score, rung1.xor_score, rung1.effect_score, rung1.duplication_score,
        reproducibility_score, full_replayability_score, rung1.coverage, stopped_early,
        n_processed, len(test_traces),
    )

    return TrialRecord(
        trial_number=rung1.trial.number,
        config=rung1.config,
        metrics=metrics,
        score=score,
        promoted_to_rung2=True,
        stopped_early=stopped_early,
    )


def find_best_config(
    training_log_path: str,
    test_traces: List[Any],
    n_trials: int,
    original_activity_names: Optional[Set[str]] = None,
    weights: Optional[ScoreWeights] = None,
    coverage_percentage: float = 0.8,
    discovery_algorithm: str = "inductive",
    round_size: int = 20,
    promotion_fraction: float = 0.35,
    min_promoted: int = 7,
    base_config: Optional[AnalysisConfig] = None,
) -> Tuple[AnalysisConfig, List[TrialRecord]]:
    """Successive Halving loop (phase 5) — docs/network_improvement_loop_plan.md §7.

    Runs n_trials in rounds of at most round_size: every trial in a round
    runs rung 1 (screening) first; only the best `promotion_fraction`
    fraction (never fewer than min_promoted, capped at the round's size)
    continue to rung 2 (full replay), which stops early (phase 6, see
    _evaluate_rung2) the moment it can no longer beat the best complete
    score found so far across the whole search. Discarded trials still get
    a study.tell() with their rung-1-only partial score, so TPE keeps
    learning from them without paying for their replay.

    Uses Optuna's ask()/tell() API rather than study.optimize() precisely
    because promotion decisions need every round's trials evaluated at
    rung 1 *before* any of them is told — study.optimize() calls the
    objective (and reports its result) one trial at a time, which doesn't
    allow that.

    training_log_path/test_traces/original_activity_names come from the
    caller's own train/test split (core_utils.split()) — see the module
    docstring: this function does no I/O of its own. That split exists
    only to score candidates — the caller (Pipeline.run(), §0) is expected
    to build the actual final network on the complete log with the
    returned config, not on this training split.

    Returns (best_config, every TrialRecord in trial order) — best_config is
    picked by .score, ties broken by whichever trial ran first.

    base_config: fields Optuna doesn't tune (see search_space.py's
    docstring) are taken from this config as-is for every trial instead of
    AnalysisConfig()'s hardcoded defaults. Defaults to None (equivalent to
    AnalysisConfig()).
    """
    weights = weights or ScoreWeights()

    preloaded = load_and_discover(training_log_path, coverage_percentage, discovery_algorithm)
    preloaded_replay = preload_replay(preloaded, base_config)

    records: List[TrialRecord] = []
    study = optuna.create_study(direction="maximize")
    best_score_so_far = float("-inf")

    remaining = n_trials
    round_num = 0
    while remaining > 0:
        round_num += 1
        this_round_size = min(round_size, remaining)
        completed_before = n_trials - remaining
        logger.info(
            "Round %d: trials %d-%d of %d",
            round_num, completed_before + 1, completed_before + this_round_size, n_trials,
        )
        trials = [study.ask() for _ in range(this_round_size)]

        rung1_results = []
        for i, trial in enumerate(trials, start=1):
            logger.info(
                "Trial %d (round %d, %d/%d): starting rung 1 evaluation",
                trial.number, round_num, i, this_round_size,
            )
            rung1_results.append(
                _evaluate_rung1(
                    trial, training_log_path, weights,
                    coverage_percentage, discovery_algorithm, original_activity_names,
                    preloaded, preloaded_replay, base_config=base_config,
                )
            )
        # Best partial_score first -- ties broken by ask() order (stable sort).
        rung1_results.sort(key=lambda r: r.partial_score, reverse=True)

        n_promote = min(
            max(round(this_round_size * promotion_fraction), min_promoted),
            this_round_size,
        )
        for i, rung1 in enumerate(rung1_results):
            if i < n_promote:
                record = _evaluate_rung2(rung1, test_traces, weights, best_score_so_far)
                best_score_so_far = max(best_score_so_far, record.score)
            else:
                record = _discarded_record(rung1)
            records.append(record)
            study.tell(rung1.trial, record.score)

        remaining -= this_round_size
        logger.info(
            "Round %d done: %d/%d promoted to rung 2, best_score_so_far=%.4f",
            round_num, n_promote, this_round_size, best_score_so_far,
        )

    best_record = max(records, key=lambda r: r.score)
    logger.info(
        "Search complete: %d trials evaluated, best trial #%d score=%.4f",
        len(records), best_record.trial_number, best_record.score,
    )
    return best_record.config, records

#TODO is to remove?
def save_trial_records(records: List[TrialRecord], path: str) -> None:
    """Write every TrialRecord as one JSON line — docs/network_improvement_loop_plan.md §8.

    Called by Pipeline.run() (§0), not by this module — network_search
    itself has no persistence of its own.
    """
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dataclasses.asdict(record), default=_json_default))
            f.write("\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
