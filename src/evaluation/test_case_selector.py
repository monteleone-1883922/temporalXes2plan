"""Stratified train/test split, specialized for plan evaluation.

Builds on the general-purpose primitives in core_utils (_load_log,
_stratified_sample_indices, _build_train_test_split, TrainTestSplit) and
adds two things specific to evaluation's own purpose, not needed by a
general train/test split (e.g. network_search's, see core_utils.split):

- min_test_cases/max_test_cases: evaluation runs want a predictable number
  of Q1/Q2/Q3 queries regardless of log size.
- min_test_trace_length: a test trace needs enough events to produce a
  meaningful prefix (evaluation.trace_sampler.sample_prefix); traces below
  this are still used for training, just never sampled as a test case.

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

import random
from typing import Optional

from core_utils import TrainTestSplit, _build_train_test_split, _load_log, _stratified_sample_indices

__all__ = ["TrainTestSplit", "split"]


def split(
    log_path: str,
    log_fmt: str = "xes",
    test_pct: float = 0.2,
    min_test_cases: int = 10,
    max_test_cases: int = 200,
    seed: int = 42,
    mapping: Optional[dict] = None,
    min_test_trace_length: int = 3,
) -> TrainTestSplit:
    """Load an event log and produce a stratified train/test split for evaluation.

    Test cases are selected via stratified sampling across trace-length deciles
    so the test set covers the full length distribution.  Traces shorter than
    *min_test_trace_length* are excluded from the test candidate pool (they
    cannot produce a meaningful prefix for evaluation) but are always kept in
    the training set.  The remaining traces are written to a temporary XES file.

    The number of test traces is computed as:
        n = clamp(round(total_traces * test_pct), min_test_cases, max_test_cases)

    Args:
        log_path: Path to the XES or CSV event log.
        log_fmt: Format of the log file — "xes" or "csv".
        test_pct: Fraction of total traces to use as test set (e.g. 0.2 = 20%).
        min_test_cases: Lower bound on the number of test traces.
        max_test_cases: Upper bound on the number of test traces.
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

    n_test_cases = max(min_test_cases, min(max_test_cases, round(len(traces) * test_pct)))

    # Separate eligible test candidates from traces that are always in train.
    eligible_indices = [i for i, t in enumerate(traces) if len(t) >= min_test_trace_length]

    rng = random.Random(seed)
    eligible_traces = [traces[i] for i in eligible_indices]
    sampled_local = _stratified_sample_indices(eligible_traces, n_test_cases, rng)
    # Map back to original indices.
    test_indices = [eligible_indices[j] for j in sampled_local]

    return _build_train_test_split(log, traces, test_indices)
