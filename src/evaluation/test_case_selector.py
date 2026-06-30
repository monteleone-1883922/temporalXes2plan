"""Stratified train/test split for event log evaluation.

Splits an event log into a training set (written to a temporary XES file) and a
list of test traces, stratified by trace-length decile so that the training set
preserves the length distribution of the original log.

Typical usage::

    from evaluation.test_case_selector import split

    with split("logs/sample.xes", log_fmt="xes", n_test_cases=20, seed=42) as s:
        # s.train_path is a temporary XES file ready for EvalAPI.parse()
        parse_result = api.parse(str(s.train_path))
        for trace in s.test_cases:
            ...
    # s.train_path has been deleted here
"""

from __future__ import annotations
from parsing.csv_loader import csv_to_event_log

import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pm4py


@dataclass
class TrainTestSplit:
    """Output of a stratified train/test split.

    Attributes:
        train_path: Path to a temporary XES file containing the training traces.
            Deleted automatically when used as a context manager.
        test_cases: List of pm4py Trace objects for evaluation.
        n_train: Number of traces in the training set.
        n_test: Number of traces in the test set.
    """

    train_path: Path
    test_cases: List[Any]
    n_train: int
    n_test: int

    def __enter__(self) -> "TrainTestSplit":
        return self

    def __exit__(self, *_: Any) -> None:
        if self.train_path.exists():
            self.train_path.unlink()


def split(
    log_path: str,
    log_fmt: str = "xes",
    n_test_cases: int = 20,
    seed: int = 42,
    mapping: Optional[dict] = None,
    min_test_trace_length: int = 3,
) -> TrainTestSplit:
    """Load an event log and produce a stratified train/test split.

    Test cases are selected via stratified sampling across trace-length deciles
    so the test set covers the full length distribution.  Traces shorter than
    *min_test_trace_length* are excluded from the test candidate pool (they
    cannot produce a meaningful prefix for evaluation) but are always kept in
    the training set.  The remaining traces are written to a temporary XES file.

    Args:
        log_path: Path to the XES or CSV event log.
        log_fmt: Format of the log file — "xes" or "csv".
        n_test_cases: Target number of test traces.  If the eligible pool has
            fewer traces than *n_test_cases*, all eligible traces become test
            cases.
        seed: Random seed for reproducibility.
        mapping: Column mapping required when log_fmt="csv".  Keys: case_id,
            activity, timestamp (optional), lifecycle (optional).
        min_test_trace_length: Minimum number of events a trace must have to be
            eligible as a test case.  Shorter traces are kept only in the
            training set.

    Returns:
        TrainTestSplit with train_path, test_cases, n_train, n_test.
    """
    log = _load_log(log_path, log_fmt, mapping)
    traces = list(log)

    # Separate eligible test candidates from traces that are always in train.
    eligible_indices = [i for i, t in enumerate(traces) if len(t) >= min_test_trace_length]

    rng = random.Random(seed)
    eligible_traces = [traces[i] for i in eligible_indices]
    sampled_local = _stratified_sample_indices(eligible_traces, n_test_cases, rng)
    # Map back to original indices.
    test_indices = [eligible_indices[j] for j in sampled_local]
    test_set = set(test_indices)

    test_cases = [traces[i] for i in test_indices]
    train_traces = [traces[i] for i in range(len(traces)) if i not in test_set]

    train_log = pm4py.objects.log.obj.EventLog(
        train_traces,
        attributes=log.attributes,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".xes", delete=False)
    tmp.close()
    train_path = Path(tmp.name)
    pm4py.write_xes(train_log, str(train_path))

    return TrainTestSplit(
        train_path=train_path,
        test_cases=test_cases,
        n_train=len(train_traces),
        n_test=len(test_cases),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_log(log_path: str, log_fmt: str, mapping: Optional[dict]) -> Any:
    """Load an XES or CSV event log via pm4py.

    Args:
        log_path: File path.
        log_fmt: "xes" or "csv".
        mapping: Column mapping for CSV files.

    Returns:
        pm4py EventLog object.
    """
    if log_fmt.lower() == "csv":

        return csv_to_event_log(log_path, mapping or {})
    return pm4py.objects.log.importer.xes.importer.apply(log_path)


def _stratified_sample_indices(
    traces: List[Any],
    n_test_cases: int,
    rng: random.Random,
) -> List[int]:
    """Select test-set indices via proportional stratified sampling on trace length deciles.

    Traces are binned into up to 10 decile buckets sorted by length.  Each
    bucket contributes a number of test cases proportional to its size, so the
    test set reflects the actual length distribution of the candidate pool.
    Leftover quota from rounding is assigned to the largest buckets first.

    Args:
        traces: Candidate traces (already filtered for minimum length).
        n_test_cases: Target total number of test traces.
        rng: Seeded Random instance for reproducibility.

    Returns:
        Sorted list of selected trace indices (into the *traces* list).
    """
    n_total = len(traces)
    if n_total == 0:
        return []

    n_select = min(n_test_cases, n_total)

    lengths = [len(t) for t in traces]
    indexed = sorted(enumerate(lengths), key=lambda x: x[1])

    n_buckets = min(10, n_total)
    buckets: List[List[int]] = [[] for _ in range(n_buckets)]
    for rank, (orig_idx, _) in enumerate(indexed):
        bucket_idx = min(rank * n_buckets // n_total, n_buckets - 1)
        buckets[bucket_idx].append(orig_idx)

    # Proportional allocation: each bucket gets floor(n_select * size / n_total).
    raw_alloc = [n_select * len(b) / n_total for b in buckets]
    alloc = [int(a) for a in raw_alloc]
    remainder = n_select - sum(alloc)

    # Distribute leftover slots to buckets with largest fractional parts.
    fracs = sorted(
        range(n_buckets),
        key=lambda i: raw_alloc[i] - alloc[i],
        reverse=True,
    )
    for i in range(remainder):
        alloc[fracs[i]] += 1

    selected: List[int] = []
    for bucket, take in zip(buckets, alloc):
        take = min(take, len(bucket))
        if take > 0:
            selected.extend(rng.sample(bucket, take))

    return sorted(selected)
