"""Unit tests for web.jobs — JobTracker thread-safe state machine."""
import threading

import pytest

from web.jobs import JobTracker


class TestJobTrackerCreate:
    def test_create_returns_string_id(self):
        jt = JobTracker()
        job_id = jt.create()
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_create_returns_unique_ids(self):
        jt = JobTracker()
        ids = {jt.create() for _ in range(10)}
        assert len(ids) == 10

    def test_new_job_has_running_status(self):
        jt = JobTracker()
        job_id = jt.create()
        assert jt.get(job_id)["status"] == "running"

    def test_new_job_has_empty_log(self):
        jt = JobTracker()
        job_id = jt.create()
        assert jt.get(job_id)["log"] == []

    def test_new_job_result_is_none(self):
        jt = JobTracker()
        job_id = jt.create()
        assert jt.get(job_id)["result"] is None

    def test_new_job_error_is_none(self):
        jt = JobTracker()
        job_id = jt.create()
        assert jt.get(job_id)["error"] is None


class TestJobTrackerAppendLog:
    def test_append_log_adds_line(self):
        jt = JobTracker()
        job_id = jt.create()
        jt.append_log(job_id, "step 1")
        assert jt.get(job_id)["log"] == ["step 1"]

    def test_append_log_multiple_lines_ordered(self):
        jt = JobTracker()
        job_id = jt.create()
        jt.append_log(job_id, "a")
        jt.append_log(job_id, "b")
        jt.append_log(job_id, "c")
        assert jt.get(job_id)["log"] == ["a", "b", "c"]

    def test_append_log_unknown_id_is_noop(self):
        jt = JobTracker()
        jt.append_log("nonexistent", "this should not raise")


class TestJobTrackerComplete:
    def test_complete_sets_done_status(self):
        jt = JobTracker()
        job_id = jt.create()
        jt.complete(job_id, {"answer": 42})
        assert jt.get(job_id)["status"] == "done"

    def test_complete_stores_result(self):
        jt = JobTracker()
        job_id = jt.create()
        jt.complete(job_id, {"plan_actions": ["a", "b"]})
        assert jt.get(job_id)["result"] == {"plan_actions": ["a", "b"]}

    def test_complete_unknown_id_is_noop(self):
        jt = JobTracker()
        jt.complete("bad-id", {})


class TestJobTrackerFail:
    def test_fail_sets_error_status(self):
        jt = JobTracker()
        job_id = jt.create()
        jt.fail(job_id, "something went wrong")
        assert jt.get(job_id)["status"] == "error"

    def test_fail_stores_error_message(self):
        jt = JobTracker()
        job_id = jt.create()
        jt.fail(job_id, "traceback here")
        assert jt.get(job_id)["error"] == "traceback here"

    def test_fail_unknown_id_is_noop(self):
        jt = JobTracker()
        jt.fail("bad-id", "error")


class TestJobTrackerGet:
    def test_get_missing_id_returns_none(self):
        jt = JobTracker()
        assert jt.get("does-not-exist") is None

    def test_get_returns_copy(self):
        jt = JobTracker()
        job_id = jt.create()
        snapshot = jt.get(job_id)
        # dict() copy: modifying the returned dict does not affect the tracker
        snapshot["status"] = "tampered"
        assert jt.get(job_id)["status"] == "running"


class TestJobTrackerThreadSafety:
    def test_concurrent_creates_all_succeed(self):
        jt = JobTracker()
        ids = []
        lock = threading.Lock()

        def create_job():
            job_id = jt.create()
            with lock:
                ids.append(job_id)

        threads = [threading.Thread(target=create_job) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 50
        assert len(set(ids)) == 50

    def test_concurrent_appends_are_consistent(self):
        jt = JobTracker()
        job_id = jt.create()

        def append_many():
            for i in range(20):
                jt.append_log(job_id, f"line-{i}")

        threads = [threading.Thread(target=append_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(jt.get(job_id)["log"]) == 100
