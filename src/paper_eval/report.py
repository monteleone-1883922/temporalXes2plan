"""Phase "report": summarise plan_results.jsonl into the RQ1 table.

For each goal ("completion", "deadline"), overall and per prefix bucket, it
reports how many instances ended in each status and, for the SOLVED ones only,
the mean, variance, median, minimum and maximum of the grounding time, the
planning time and the total time.

Timeouts, out-of-memory aborts, unsolvable problems, errors and skipped
instances are only COUNTED: they never enter the averages, so they cannot
bias them (a timeout has no real planning time). Every instance weighs 1, as
the instances are already deduplicated.

The variance is the sample variance (n - 1); it is None for fewer than two values.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from paper_eval.plan import RESULTS_FILE, read_results

SUMMARY_JSON = "rq1_summary.json"
SUMMARY_MD = "rq1_summary.md"
# The times that go into the means, for solved instances.
TIME_FIELDS = ("grounding_s", "planning_s", "wall_s")


def describe(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    """n, mean, sample variance, median, min and max of the non-missing values."""
    xs = [float(v) for v in values if v is not None]
    n = len(xs)
    return {
        "n": n,
        "mean": statistics.fmean(xs) if n else None,
        "variance": statistics.variance(xs) if n > 1 else None,
        "median": statistics.median(xs) if n else None,
        "min": min(xs) if n else None,
        "max": max(xs) if n else None,
    }


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts by status and, for the solved rows, the time statistics."""
    solved = [r for r in rows if r["status"] == "solved"]
    return {
        "n_instances": len(rows),
        "n_cases": sum(r.get("n_cases") or 0 for r in rows),
        "by_status": dict(sorted(Counter(r["status"] for r in rows).items())),
        "solved": {field: describe(r[field] for r in solved) for field in TIME_FIELDS},
    }


def _bucket_order(rows: List[Dict[str, Any]]) -> List[Optional[str]]:
    """Buckets from the shortest prefixes to the longest (None last)."""
    first_k: Dict[Optional[str], int] = {}
    for r in rows:
        first_k[r["bucket"]] = min(r["k"], first_k.get(r["bucket"], r["k"]))
    return sorted(first_k, key=lambda b: (b is None, first_k[b]))


def summarize_rq1(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """{goal: {"overall": group, "by_bucket": {bucket: group}}} for the rows."""
    by_goal: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_goal[r["goal"]].append(r)
    summary: Dict[str, Any] = {}
    for goal in sorted(by_goal):
        goal_rows = by_goal[goal]
        by_bucket = {
            str(b): summarize_group([r for r in goal_rows if r["bucket"] == b])
            for b in _bucket_order(goal_rows)
        }
        summary[goal] = {"overall": summarize_group(goal_rows), "by_bucket": by_bucket}
    return summary


def _fmt(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:.3f}"


def _count(group: Dict[str, Any], prefix: str) -> int:
    return sum(n for status, n in group["by_status"].items() if status.startswith(prefix))


def to_markdown(summary: Dict[str, Any]) -> str:
    """One table per goal; times in seconds, over the solved instances only."""
    header = (
        "| bucket | instances | solved | timeout | out of memory | unsolvable | error | skipped "
        "| grounding mean | grounding var | planning mean | planning var | total mean | total var |"
    )
    rule = "|" + "---|" * 14
    out: List[str] = []
    for goal, blocks in summary.items():
        out += [f"### Goal: {goal}", "", header, rule]
        rows = [("all", blocks["overall"])] + list(blocks["by_bucket"].items())
        for name, g in rows:
            s = g["by_status"]
            t = g["solved"]
            out.append(
                f"| {name} | {g['n_instances']} | {s.get('solved', 0)} | {s.get('timeout', 0)} "
                f"| {s.get('out_of_memory', 0)} | {_count(g, 'unsolvable')} | {s.get('error', 0)} "
                f"| {_count(g, 'skipped')} "
                f"| {_fmt(t['grounding_s']['mean'])} | {_fmt(t['grounding_s']['variance'])} "
                f"| {_fmt(t['planning_s']['mean'])} | {_fmt(t['planning_s']['variance'])} "
                f"| {_fmt(t['wall_s']['mean'])} | {_fmt(t['wall_s']['variance'])} |"
            )
        out.append("")
    return "\n".join(out)


def run_report(model_dir: Path) -> Dict[str, Any]:
    """Read model_dir/plan_results.jsonl, write rq1_summary.json and rq1_summary.md."""
    rows = read_results(Path(model_dir) / RESULTS_FILE)
    if not rows:
        raise FileNotFoundError(f"no rows in {Path(model_dir) / RESULTS_FILE}: run the 'plan' phase first")
    summary = summarize_rq1(rows)
    (Path(model_dir) / SUMMARY_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path(model_dir) / SUMMARY_MD).write_text(to_markdown(summary), encoding="utf-8")
    return summary
