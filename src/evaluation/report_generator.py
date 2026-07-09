"""Write per-log results and cross-log summary to disk.

Per-log results are written to ``<output_dir>/<log_id>/result.json``.
The cross-log summary is written atomically (write to .tmp, then rename)
to ``<output_dir>/summary.json`` and ``<output_dir>/summary.csv``.

Typical usage::

    from evaluation.report_generator import LogResult, QueryResult, write_log_result, write_cross_log_summary

    write_log_result(log_id, log_result, output_dir)
    write_cross_log_summary(all_results, output_dir, cost_weight=0.001)
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Metrics and metadata for a single planner query (Q1, Q2, or Q3).

    Attributes:
        query_id: Unique identifier, e.g. "<log_id>_<trace_id>_Q1".
        query_type: "Q1", "Q2", or "Q3".
        trace_id: Case ID of the source trace.
        prefix_ratio: Fraction of events used as prefix.
        n_prefix_events: Number of events in the prefix.
        attempts: Number of planner invocations (including retries).
        solvability: Final solvability string from the planner.
        planner_duration_s: Wall-clock time of the last planner call, in seconds.
        metrics: Dict produced by q*_metrics().
        validation: Dict produced by ValidationReport (or None if plan not found).
    """

    query_id: str
    query_type: str
    trace_id: str
    prefix_ratio: float
    n_prefix_events: int
    attempts: int
    solvability: str
    planner_duration_s: Optional[float]
    metrics: Dict[str, Any]
    validation: Optional[Dict[str, Any]]
    is_replayable: bool = True


@dataclass
class LogResult:
    """All results for a single event log.

    Attributes:
        log_id: Event Log ID from the metadata CSV.
        log_name: Human-readable name from the metadata CSV.
        log_fmt: "xes" or "csv".
        n_train_cases: Number of traces in the training set.
        n_test_cases: Number of traces in the test set.
        n_activities: Number of distinct activities in the domain.
        pipeline_ok: False if parsing/domain-building failed.
        pipeline_error: Error message when pipeline_ok is False.
        queries: List of QueryResult objects.
        used_optimizer: True when network_search's optimizer picked the
            AnalysisConfig for this log instead of the fixed CLI config.
        search_summary: {"n_trials", "best_score", "best_trial_number"} when
            used_optimizer is True, else None.
    """

    log_id: str
    log_name: str
    log_fmt: str
    n_train_cases: int
    n_test_cases: int
    n_activities: int
    pipeline_ok: bool
    pipeline_error: Optional[str]
    queries: List[QueryResult] = field(default_factory=list)
    used_optimizer: bool = False
    search_summary: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_log_result(
    log_id: str,
    log_result: LogResult,
    output_dir: Path,
) -> Path:
    """Serialise a LogResult to ``<output_dir>/<log_id>/result.json``.

    Args:
        log_id: Event Log ID (used as directory name).
        log_result: LogResult to serialise.
        output_dir: Root results directory.

    Returns:
        Path to the written file.
    """
    log_dir = Path(output_dir) / log_id
    log_dir.mkdir(parents=True, exist_ok=True)
    dest = log_dir / "result.json"
    payload = _log_result_to_dict(log_result)
    _atomic_write_json(dest, payload)
    return dest


def write_cross_log_summary(
    all_results: List[LogResult],
    output_dir: Path,
    cost_weight: float = 0.001,
) -> None:
    """Write ``summary.json`` and ``summary.csv`` to output_dir.

    Both files are written atomically (to a .tmp sibling, then renamed) so
    that a crash mid-write never leaves a partially written file.

    Args:
        all_results: All LogResult objects collected so far.
        output_dir: Root results directory.
        cost_weight: α value recorded in the summary for reference.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_summary(all_results, cost_weight)
    _atomic_write_json(output_dir / "summary.json", summary)
    _write_summary_csv(output_dir / "summary.csv", summary)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _log_result_to_dict(lr: LogResult) -> Dict[str, Any]:
    return {
        "log_id": lr.log_id,
        "log_name": lr.log_name,
        "log_fmt": lr.log_fmt,
        "n_train_cases": lr.n_train_cases,
        "n_test_cases": lr.n_test_cases,
        "n_activities": lr.n_activities,
        "pipeline_ok": lr.pipeline_ok,
        "pipeline_error": lr.pipeline_error,
        "replayability": _replayability_stats(lr.queries),
        "queries": [_query_result_to_dict(q) for q in lr.queries],
        "used_optimizer": lr.used_optimizer,
        "search_summary": lr.search_summary,
    }


def _query_result_to_dict(qr: QueryResult) -> Dict[str, Any]:
    return {
        "query_id": qr.query_id,
        "query_type": qr.query_type,
        "trace_id": qr.trace_id,
        "prefix_ratio": qr.prefix_ratio,
        "n_prefix_events": qr.n_prefix_events,
        "attempts": qr.attempts,
        "solvability": qr.solvability,
        "planner_duration_s": qr.planner_duration_s,
        "metrics": qr.metrics,
        "validation": qr.validation,
        "is_replayable": qr.is_replayable,
    }


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    all_results: List[LogResult],
    cost_weight: float,
) -> Dict[str, Any]:
    n_logs = len(all_results)
    n_logs_ok = sum(1 for r in all_results if r.pipeline_ok)
    n_queries_total = sum(len(r.queries) for r in all_results)

    q1_solved: List[float] = []
    q1_within: List[float] = []
    q1_plan_time: List[float] = []
    q2_solved: List[float] = []
    q2_within: List[float] = []
    q2_plan_time: List[float] = []
    q3_solved: List[float] = []
    q3_correct: List[float] = []
    q3_within: List[float] = []
    q3_plan_time: List[float] = []

    # Replayability cross-check: solved despite the trace not being
    # replayable, and not-solved despite the trace being replayable —
    # accumulated per query type across logs (mean of per-log ratios,
    # same convention as solved_ratio_mean).
    solved_given_not_replayable: Dict[str, List[float]] = {"Q1": [], "Q2": [], "Q3": []}
    not_solved_given_replayable: Dict[str, List[float]] = {"Q1": [], "Q2": [], "Q3": []}

    per_log_rows: List[Dict[str, Any]] = []

    for lr in all_results:
        log_q1 = [q for q in lr.queries if q.query_type == "Q1"]
        log_q2 = [q for q in lr.queries if q.query_type == "Q2"]
        log_q3 = [q for q in lr.queries if q.query_type == "Q3"]

        q1_sr = _ratio(log_q1, "solved")
        q1_wr = _ratio_metric(log_q1, "within_budget")
        q2_sr = _ratio(log_q2, "solved")
        q2_wr = _ratio_metric(log_q2, "within_budget")
        q3_cr = _ratio_metric(log_q3, "correct")

        replay_stats = _replayability_stats(lr.queries)
        per_log_rows.append({
            "log_id": lr.log_id,
            "log_name": lr.log_name,
            "log_fmt": lr.log_fmt,
            "n_test_cases": lr.n_test_cases,
            "q1_solved_ratio": q1_sr,
            "q2_solved_ratio": q2_sr,
            "q2_within_budget_ratio": q2_wr,
            "q3_correct_ratio": q3_cr,
            "n_traces_replayable": replay_stats["n_traces_replayable"],
            "n_traces_not_replayable": replay_stats["n_traces_not_replayable"],
            "pct_traces_replayable": replay_stats["pct_traces_replayable"],
            "q1_solved_replayable": replay_stats["q1_solved_replayable"],
            "q1_solved_not_replayable": replay_stats["q1_solved_not_replayable"],
            "q2_solved_replayable": replay_stats["q2_solved_replayable"],
            "q2_solved_not_replayable": replay_stats["q2_solved_not_replayable"],
            "q3_solved_replayable": replay_stats["q3_solved_replayable"],
            "q3_solved_not_replayable": replay_stats["q3_solved_not_replayable"],
        })

        for qtype in ("Q1", "Q2", "Q3"):
            key = qtype.lower()
            snr = replay_stats[f"{key}_solved_not_replayable"]
            if snr is not None:
                solved_given_not_replayable[qtype].append(snr)
            sr = replay_stats[f"{key}_solved_replayable"]
            if sr is not None:
                not_solved_given_replayable[qtype].append(1 - sr)

        # Accumulate for global means.
        if q1_sr is not None:
            q1_solved.append(q1_sr)
        if q1_wr is not None:
            q1_within.append(q1_wr)
        for q in log_q1:
            v = q.metrics.get("plan_time_s")
            if v is not None:
                q1_plan_time.append(v)

        if q2_sr is not None:
            q2_solved.append(q2_sr)
        if q2_wr is not None:
            q2_within.append(q2_wr)
        for q in log_q2:
            v = q.metrics.get("plan_time_s")
            if v is not None:
                q2_plan_time.append(v)

        q3_sr = _ratio(log_q3, "solved")
        if q3_sr is not None:
            q3_solved.append(q3_sr)
        if q3_cr is not None:
            q3_correct.append(q3_cr)
        q3_wr = _ratio_metric(log_q3, "within_budget")
        if q3_wr is not None:
            q3_within.append(q3_wr)
        for q in log_q3:
            v = q.metrics.get("plan_time_s")
            if v is not None:
                q3_plan_time.append(v)

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_logs": n_logs,
        "n_logs_ok": n_logs_ok,
        "n_queries_total": n_queries_total,
        "cost_weight": cost_weight,
        "q1": {
            "solved_ratio_mean": _mean(q1_solved),
            "within_budget_ratio_mean": _mean(q1_within),
            "plan_time_mean": _mean(q1_plan_time),
            "plan_time_std": _std(q1_plan_time),
            "solved_given_not_replayable_mean": _mean(solved_given_not_replayable["Q1"]),
            "not_solved_given_replayable_mean": _mean(not_solved_given_replayable["Q1"]),
        },
        "q2": {
            "solved_ratio_mean": _mean(q2_solved),
            "within_budget_ratio_mean": _mean(q2_within),
            "plan_time_mean": _mean(q2_plan_time),
            "plan_time_std": _std(q2_plan_time),
            "solved_given_not_replayable_mean": _mean(solved_given_not_replayable["Q2"]),
            "not_solved_given_replayable_mean": _mean(not_solved_given_replayable["Q2"]),
        },
        "q3": {
            "solved_ratio_mean": _mean(q3_solved),
            "correct_ratio_mean": _mean(q3_correct),
            "within_budget_ratio_mean": _mean(q3_within),
            "plan_time_mean": _mean(q3_plan_time),
            "plan_time_std": _std(q3_plan_time),
            "solved_given_not_replayable_mean": _mean(solved_given_not_replayable["Q3"]),
            "not_solved_given_replayable_mean": _mean(not_solved_given_replayable["Q3"]),
        },
        "per_log": per_log_rows,
    }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_summary_csv(dest: Path, summary: Dict[str, Any]) -> None:
    per_log = summary.get("per_log", [])
    if not per_log:
        dest.write_text("", encoding="utf-8")
        return

    columns = list(per_log[0].keys())
    tid = threading.get_ident()
    tmp = dest.with_name(f"{dest.stem}.{os.getpid()}_{tid}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(per_log)
    tmp.replace(dest)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _atomic_write_json(dest: Path, payload: Any) -> None:
    tid = threading.get_ident()
    tmp = dest.with_name(f"{dest.stem}.{os.getpid()}_{tid}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)


# ---------------------------------------------------------------------------
# Replayability helpers
# ---------------------------------------------------------------------------

def _solved_ratio(queries: List[QueryResult]) -> Optional[float]:
    """Fraction of queries where solvability == 'solved'. None if list is empty."""
    if not queries:
        return None
    return sum(1 for q in queries if q.solvability == "solved") / len(queries)


def _replayability_stats(queries: List[QueryResult]) -> Dict[str, Any]:
    """Compute replayability breakdown and per-query-type solved rates.

    Counts unique traces by trace_id; all queries for the same trace share the
    same is_replayable value — we take it from the first query seen for that trace.
    """
    trace_replayable: Dict[str, bool] = {}
    for q in queries:
        if q.trace_id not in trace_replayable:
            trace_replayable[q.trace_id] = q.is_replayable

    n_rep = sum(1 for v in trace_replayable.values() if v)
    n_not = sum(1 for v in trace_replayable.values() if not v)
    n_total = n_rep + n_not

    stats: Dict[str, Any] = {
        "n_traces_replayable": n_rep,
        "n_traces_not_replayable": n_not,
        "pct_traces_replayable": round(n_rep / n_total, 4) if n_total > 0 else None,
    }

    for qtype in ("Q1", "Q2", "Q3"):
        qtype_queries = [q for q in queries if q.query_type == qtype]
        rep_q = [q for q in qtype_queries if q.is_replayable]
        not_q = [q for q in qtype_queries if not q.is_replayable]
        key = qtype.lower()
        stats[f"{key}_solved_replayable"] = _solved_ratio(rep_q)
        stats[f"{key}_solved_not_replayable"] = _solved_ratio(not_q)

    return stats


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _std(values: List[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def _ratio(queries: List[QueryResult], bool_key: str) -> Optional[float]:
    """Compute the fraction of queries where metrics[bool_key] is True."""
    vals = [q.metrics.get(bool_key) for q in queries if q.metrics.get(bool_key) is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def _ratio_metric(queries: List[QueryResult], metric_key: str) -> Optional[float]:
    """Compute the fraction of queries where metrics[metric_key] is True."""
    vals = [q.metrics.get(metric_key) for q in queries
            if q.metrics.get(metric_key) is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)
