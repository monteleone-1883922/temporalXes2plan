"""Unit tests for evaluation.test_case_selector."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pm4py
import pytest

from evaluation.test_case_selector import TrainTestSplit, split, _stratified_sample_indices

# Path to a small real XES file available in the test fixtures.
FIXTURE_XES = Path(__file__).parent.parent / "fixtures" / "partial_trace_sample_clinic.xes"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(length: int, case_id: str) -> pm4py.objects.log.obj.Trace:
    trace = pm4py.objects.log.obj.Trace()
    trace.attributes["concept:name"] = case_id
    for i in range(length):
        event = pm4py.objects.log.obj.Event()
        event["concept:name"] = f"act_{i}"
        trace.append(event)
    return trace


def _make_log(lengths: list[int]) -> pm4py.objects.log.obj.EventLog:
    log = pm4py.objects.log.obj.EventLog()
    for i, l in enumerate(lengths):
        log.append(_make_trace(l, str(i)))
    return log


def _run_split(lengths, n_test, seed=0, min_len=1):
    """Helper: run split() with a synthetic in-memory log via mocks."""
    log = _make_log(lengths)
    with patch("evaluation.test_case_selector._load_log", return_value=log), \
         patch("evaluation.test_case_selector.pm4py.write_xes"):
        with tempfile.NamedTemporaryFile(suffix=".xes", delete=False) as f:
            tmp = Path(f.name)
        with patch("evaluation.test_case_selector.tempfile.NamedTemporaryFile") as mock_ntf:
            mock_ntf.return_value.name = str(tmp)
            s = split("fake.xes", n_test_cases=n_test, seed=seed,
                      min_test_trace_length=min_len)
            s.train_path = tmp
    tmp.touch()
    return s, log


# ---------------------------------------------------------------------------
# _stratified_sample_indices (unit)
# ---------------------------------------------------------------------------

class TestStratifiedSampleIndices:
    def test_returns_correct_count(self):
        import random
        traces = [_make_trace(i + 1, str(i)) for i in range(30)]
        result = _stratified_sample_indices(traces, 10, random.Random(0))
        assert len(result) == 10

    def test_never_exceeds_available(self):
        import random
        traces = [_make_trace(1, str(i)) for i in range(5)]
        result = _stratified_sample_indices(traces, 20, random.Random(0))
        assert len(result) <= 5

    def test_empty_log_returns_empty(self):
        import random
        result = _stratified_sample_indices([], 5, random.Random(0))
        assert result == []

    def test_result_sorted(self):
        import random
        traces = [_make_trace(i + 1, str(i)) for i in range(20)]
        result = _stratified_sample_indices(traces, 8, random.Random(0))
        assert result == sorted(result)

    def test_indices_within_bounds(self):
        import random
        traces = [_make_trace(i + 1, str(i)) for i in range(20)]
        result = _stratified_sample_indices(traces, 8, random.Random(0))
        assert all(0 <= idx < len(traces) for idx in result)

    def test_no_duplicates(self):
        import random
        traces = [_make_trace(i + 1, str(i)) for i in range(20)]
        result = _stratified_sample_indices(traces, 8, random.Random(0))
        assert len(result) == len(set(result))

    def test_proportional_allocation_reflects_bucket_sizes(self):
        """Larger buckets must contribute more samples than smaller ones."""
        import random
        # 90 long traces + 10 short traces → long bucket should dominate
        traces = [_make_trace(10, str(i)) for i in range(90)]
        traces += [_make_trace(1, str(i + 90)) for i in range(10)]
        result = _stratified_sample_indices(traces, 20, random.Random(0))
        long_selected = sum(1 for idx in result if idx < 90)
        short_selected = sum(1 for idx in result if idx >= 90)
        assert long_selected > short_selected


# ---------------------------------------------------------------------------
# split() — sizes and disjointness
# ---------------------------------------------------------------------------

class TestSplitSizes:
    def test_split_sizes_sum_to_total(self):
        lengths = list(range(3, 33))  # 30 traces, all >= min_len=3
        s, log = _run_split(lengths, n_test=10, min_len=3)
        assert s.n_train + s.n_test == 30

    def test_n_test_cases_respected(self):
        lengths = list(range(3, 33))
        s, _ = _run_split(lengths, n_test=10, min_len=3)
        assert s.n_test == 10

    def test_n_test_capped_when_log_too_small(self):
        lengths = [3, 4, 5]  # only 3 eligible traces
        s, _ = _run_split(lengths, n_test=20, min_len=3)
        assert s.n_test <= 3

    def test_split_disjoint(self):
        lengths = list(range(3, 33))
        s, log = _run_split(lengths, n_test=10, min_len=3)
        test_ids = {t.attributes.get("concept:name") for t in s.test_cases}
        all_ids = {t.attributes.get("concept:name") for t in log}
        train_ids = all_ids - test_ids
        assert test_ids.isdisjoint(train_ids)

    def test_seed_reproducibility(self):
        lengths = list(range(3, 33))
        s1, _ = _run_split(lengths, n_test=10, seed=99, min_len=3)
        s2, _ = _run_split(lengths, n_test=10, seed=99, min_len=3)
        ids1 = [t.attributes.get("concept:name") for t in s1.test_cases]
        ids2 = [t.attributes.get("concept:name") for t in s2.test_cases]
        assert ids1 == ids2

    def test_different_seeds_give_different_splits(self):
        lengths = list(range(3, 53))  # 50 traces
        s1, _ = _run_split(lengths, n_test=10, seed=1, min_len=3)
        s2, _ = _run_split(lengths, n_test=10, seed=2, min_len=3)
        ids1 = [t.attributes.get("concept:name") for t in s1.test_cases]
        ids2 = [t.attributes.get("concept:name") for t in s2.test_cases]
        assert ids1 != ids2


# ---------------------------------------------------------------------------
# split() — short trace exclusion from test set
# ---------------------------------------------------------------------------

class TestShortTraceExclusion:
    def test_short_traces_not_in_test_set(self):
        # traces 0-4 have length 1 (too short), 5-24 have length 5
        lengths = [1] * 5 + [5] * 20
        s, log = _run_split(lengths, n_test=10, min_len=3)
        test_ids = {t.attributes.get("concept:name") for t in s.test_cases}
        short_ids = {str(i) for i in range(5)}
        assert test_ids.isdisjoint(short_ids)

    def test_short_traces_counted_in_train(self):
        lengths = [1] * 5 + [5] * 20
        s, _ = _run_split(lengths, n_test=10, min_len=3)
        # All 5 short traces must end up in train
        assert s.n_train >= 5

    def test_all_ineligible_gives_empty_test(self):
        lengths = [1, 2, 1, 2]  # all below min_len=3
        s, _ = _run_split(lengths, n_test=10, min_len=3)
        assert s.n_test == 0
        assert s.n_train == 4


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_deletes_train_file(self, tmp_path):
        train_file = tmp_path / "train.xes"
        train_file.write_text("<log/>")
        s = TrainTestSplit(train_path=train_file, test_cases=[], n_train=0, n_test=0)
        with s:
            assert train_file.exists()
        assert not train_file.exists()

    def test_context_manager_returns_self(self, tmp_path):
        train_file = tmp_path / "train.xes"
        train_file.write_text("<log/>")
        s = TrainTestSplit(train_path=train_file, test_cases=[], n_train=0, n_test=0)
        with s as result:
            assert result is s

    def test_context_manager_tolerates_missing_file(self, tmp_path):
        train_file = tmp_path / "nonexistent.xes"
        s = TrainTestSplit(train_path=train_file, test_cases=[], n_train=0, n_test=0)
        with s:
            pass  # should not raise


# ---------------------------------------------------------------------------
# Integration: real XES file (no mock)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FIXTURE_XES.exists(), reason="fixture XES not found")
class TestSplitWithRealLog:
    def test_train_path_is_readable_xes(self):
        s = split(str(FIXTURE_XES), log_fmt="xes", n_test_cases=2, seed=0)
        try:
            loaded = pm4py.read_xes(str(s.train_path))
            assert loaded is not None
        finally:
            if s.train_path.exists():
                s.train_path.unlink()

    def test_context_manager_deletes_train_file(self):
        with split(str(FIXTURE_XES), log_fmt="xes", n_test_cases=2, seed=0) as s:
            path = s.train_path
            assert path.exists()
        assert not path.exists()

    def test_split_sizes_with_real_log(self):
        with split(str(FIXTURE_XES), log_fmt="xes", n_test_cases=2, seed=0) as s:
            assert s.n_train + s.n_test > 0
            assert s.n_test <= 2

    def test_test_traces_meet_min_length(self):
        min_len = 3
        with split(str(FIXTURE_XES), log_fmt="xes", n_test_cases=5,
                   seed=0, min_test_trace_length=min_len) as s:
            for trace in s.test_cases:
                assert len(trace) >= min_len
