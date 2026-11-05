"""Tests for paper_eval.report (the RQ1 table from plan_results.jsonl)."""
import json

import pytest

from paper_eval import report as rp
from paper_eval.plan import RESULTS_FILE


def _row(status="solved", goal="completion", k=2, bucket="2", grounding=1.0, planning=2.0, wall=3.5, n_cases=1):
    return {
        "status": status, "goal": goal, "k": k, "bucket": bucket, "n_cases": n_cases,
        "grounding_s": grounding if status == "solved" else None,
        "planning_s": planning if status == "solved" else None,
        "wall_s": wall,
    }


class TestDescribe:
    def test_mean_variance_median_min_max(self):
        d = rp.describe([1.0, 2.0, 3.0, 6.0])
        assert d["n"] == 4 and d["mean"] == pytest.approx(3.0)
        assert d["variance"] == pytest.approx(14 / 3)  # sample variance, n - 1
        assert (d["median"], d["min"], d["max"]) == (2.5, 1.0, 6.0)

    def test_variance_needs_two_values(self):
        assert rp.describe([4.0])["variance"] is None
        assert rp.describe([4.0])["mean"] == 4.0

    def test_missing_values_are_ignored_and_empty_input_is_all_none(self):
        assert rp.describe([None, 2.0])["n"] == 1
        empty = rp.describe([])
        assert empty["n"] == 0 and empty["mean"] is None and empty["median"] is None


class TestSummarizeGroup:
    def test_only_solved_instances_enter_the_averages(self):
        rows = [
            _row(grounding=1.0, planning=2.0), _row(grounding=3.0, planning=4.0),
            _row("timeout", wall=60.0), _row("out_of_memory", wall=5.0),
        ]
        g = rp.summarize_group(rows)
        assert g["solved"]["grounding_s"]["mean"] == pytest.approx(2.0)
        assert g["solved"]["planning_s"]["mean"] == pytest.approx(3.0)
        # the timeout's 60 s and the OOM's 5 s do not leak into the total time
        assert g["solved"]["wall_s"]["n"] == 2 and g["solved"]["wall_s"]["mean"] == pytest.approx(3.5)

    def test_counts_every_status_and_every_case(self):
        rows = [_row(n_cases=2), _row("timeout", n_cases=3), _row("skipped_expired"), _row("timeout")]
        g = rp.summarize_group(rows)
        assert g["by_status"] == {"skipped_expired": 1, "solved": 1, "timeout": 2}
        assert g["n_instances"] == 4 and g["n_cases"] == 2 + 3 + 1 + 1

    def test_group_without_solved_instances_has_empty_statistics(self):
        g = rp.summarize_group([_row("timeout")])
        assert g["solved"]["planning_s"]["n"] == 0 and g["solved"]["planning_s"]["mean"] is None


class TestSummarizeRq1:
    def test_goals_are_kept_apart(self):
        rows = [_row(goal="completion"), _row(goal="deadline"), _row(goal="deadline")]
        s = rp.summarize_rq1(rows)
        assert s["completion"]["overall"]["n_instances"] == 1
        assert s["deadline"]["overall"]["n_instances"] == 2

    def test_buckets_run_from_short_to_long_prefixes_with_none_last(self):
        rows = [_row(k=12, bucket="11-15"), _row(k=2, bucket="2"), _row(k=5, bucket="5"), _row(k=30, bucket=None)]
        s = rp.summarize_rq1(rows)
        assert list(s["completion"]["by_bucket"]) == ["2", "5", "11-15", "None"]

    def test_overall_covers_every_bucket(self):
        rows = [_row(k=2, bucket="2", planning=1.0), _row(k=5, bucket="5", planning=3.0)]
        s = rp.summarize_rq1(rows)["completion"]
        assert s["overall"]["solved"]["planning_s"]["mean"] == pytest.approx(2.0)
        assert s["by_bucket"]["5"]["solved"]["planning_s"]["mean"] == pytest.approx(3.0)


class TestMarkdown:
    def test_table_has_a_row_per_bucket_and_counts(self):
        rows = [_row(), _row("timeout"), _row("unsolvable_structural"), _row("skipped_replay_failed", k=3, bucket="3")]
        text = rp.to_markdown(rp.summarize_rq1(rows))
        assert "### Goal: completion" in text
        assert "| all | 4 | 1 | 1 | 0 | 1 | 0 | 1 |" in text  # unsolvable and skipped are grouped
        assert "| 3 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |" in text

    def test_missing_statistics_are_a_dash(self):
        text = rp.to_markdown(rp.summarize_rq1([_row("timeout")]))
        assert "–" in text


class TestRunReport:
    def test_writes_json_and_markdown(self, tmp_path):
        with open(tmp_path / RESULTS_FILE, "w") as fh:
            for row in (_row(), _row("timeout")):
                fh.write(json.dumps(row) + "\n")
        summary = rp.run_report(tmp_path)
        assert json.loads((tmp_path / rp.SUMMARY_JSON).read_text()) == summary
        assert "Goal: completion" in (tmp_path / rp.SUMMARY_MD).read_text()

    def test_no_results_says_to_run_plan_first(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="plan"):
            rp.run_report(tmp_path)
