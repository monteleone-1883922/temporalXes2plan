"""
Tests for parsing.decision_mining — pure-function tests only.

Only load_and_prepare_log and find_decision_points are tested here: both are
pure data-transformation functions that operate on a pm4py EventLog / DataFrame
and do not require running the full decision-tree mining pipeline.

Tests that require discover_all_decision_rules (which runs the full Parser
pipeline and needs a real XES file) are marked @pytest.mark.integration and
are skipped by default.  Run them explicitly with:
    conda run -n temporalXes2plan python -m pytest -m integration
"""
import pytest

from tests.helpers import make_event, make_trace, make_log
from parsing.decision_mining import (
    META_COLUMNS_IGNORED,
    load_and_prepare_log,
    find_decision_points,
)


# ---------------------------------------------------------------------------
# Helper — build a minimal pm4py log with known column types
# ---------------------------------------------------------------------------

def _make_typed_log():
    """
    Build a two-case EventLog that contains:
      - a numeric column  'crp'   (float)
      - a boolean column  'admitted' (bool, converted to 0/1 by load_and_prepare_log)
      - a categorical col 'diagnosis' (string)
    Each case has two events so that 'next_activity' can be computed.
    """
    return make_log(
        make_trace(
            make_event("triage",    offset=0,  crp=4.5, admitted=True,  diagnosis="A"),
            make_event("treatment", offset=60, crp=2.1, admitted=True,  diagnosis="A"),
            case_id="c1",
        ),
        make_trace(
            make_event("triage",    offset=0,  crp=8.0, admitted=False, diagnosis="B"),
            make_event("discharge", offset=60, crp=7.5, admitted=False, diagnosis="B"),
            case_id="c2",
        ),
    )


# ===========================================================================
# load_and_prepare_log
# ===========================================================================

class TestLoadAndPrepareLog:
    def test_returns_five_element_tuple(self):
        """
        load_and_prepare_log must return a tuple of exactly 5 elements:
        (DataFrame, all_feature_cols, numeric_cols, bool_cols, categorical_cols).
        """
        result = load_and_prepare_log(
            log_path="unused_because_log_is_provided",
            meta_cols=META_COLUMNS_IGNORED,
            log=_make_typed_log(),
        )
        assert len(result) == 5

    def test_next_activity_column_added(self):
        """
        load_and_prepare_log must add a 'next_activity' column to the DataFrame,
        containing the concept:name of the following event within each case.
        """
        df, *_ = load_and_prepare_log(
            log_path="unused",
            meta_cols=META_COLUMNS_IGNORED,
            log=_make_typed_log(),
        )
        assert "next_activity" in df.columns

    def test_meta_columns_excluded_from_features(self):
        """
        Standard meta columns (concept:name, time:timestamp, etc.) must not appear
        in the all_feature_cols list, since they carry no predictive information
        for decision mining.
        """
        _, all_feature_cols, *_ = load_and_prepare_log(
            log_path="unused",
            meta_cols=META_COLUMNS_IGNORED,
            log=_make_typed_log(),
        )
        excluded = {"concept:name", "time:timestamp", "lifecycle:transition",
                    "org:resource", "org:group"}
        for col in all_feature_cols:
            assert col not in excluded, f"Meta column '{col}' leaked into features"

    def test_numeric_column_classified_correctly(self):
        """
        The 'crp' column, which contains float values, must appear in numeric_cols.
        """
        _, _, numeric_cols, _, _ = load_and_prepare_log(
            log_path="unused",
            meta_cols=META_COLUMNS_IGNORED,
            log=_make_typed_log(),
        )
        assert "crp" in numeric_cols

    def test_boolean_column_converted_to_integer(self):
        """
        Boolean columns are converted to 0/1 integers by load_and_prepare_log.
        The 'admitted' column must appear in bool_cols and the DataFrame must
        contain only 0 and 1 values for it.
        """
        df, _, _, bool_cols, _ = load_and_prepare_log(
            log_path="unused",
            meta_cols=META_COLUMNS_IGNORED,
            log=_make_typed_log(),
        )
        assert "admitted" in bool_cols
        assert set(df["admitted"].dropna().unique()).issubset({0, 1})

    def test_categorical_column_classified_correctly(self):
        """
        The 'diagnosis' column, which contains string values ('A', 'B'),
        must appear in categorical_cols.
        """
        _, _, _, _, categorical_cols = load_and_prepare_log(
            log_path="unused",
            meta_cols=META_COLUMNS_IGNORED,
            log=_make_typed_log(),
        )
        assert "diagnosis" in categorical_cols

    def test_no_exception_when_log_has_no_feature_columns(self):
        """
        A log with only meta columns and no domain attributes must not raise.
        A WARNING is expected (no features detected) but the function completes.
        """
        log = make_log(
            make_trace(
                make_event("A", offset=0),
                make_event("B", offset=30),
            )
        )
        # Should complete without exception even if all_feature_cols is empty
        df, all_feature_cols, *_ = load_and_prepare_log(
            log_path="unused",
            meta_cols=META_COLUMNS_IGNORED,
            log=log,
        )
        assert df is not None


# ===========================================================================
# find_decision_points
# ===========================================================================

class TestFindDecisionPoints:
    def test_activity_with_multiple_successors_is_a_decision_point(self):
        """
        An activity that leads to more than one distinct next activity in the log
        qualifies as a decision point, provided it meets the min_instances threshold.
        'triage' leads to both 'treatment' and 'discharge' across cases.
        """
        df, *_ = load_and_prepare_log(
            log_path="unused",
            meta_cols=META_COLUMNS_IGNORED,
            log=_make_typed_log(),
        )
        # min_instances=1 so both activities qualify regardless of frequency
        decision_pts = find_decision_points(df, min_instances=1)
        assert "triage" in decision_pts

    def test_activity_with_single_successor_is_not_a_decision_point(self):
        """
        'treatment' only leads to the end of its trace — it has no branching.
        It must not appear in the decision points list.
        """
        df, *_ = load_and_prepare_log(
            log_path="unused",
            meta_cols=META_COLUMNS_IGNORED,
            log=_make_typed_log(),
        )
        decision_pts = find_decision_points(df, min_instances=1)
        assert "treatment" not in decision_pts

    def test_min_instances_threshold_filters_rare_activities(self):
        """
        An activity that appears fewer times than min_instances must not be
        classified as a decision point, even if it has multiple successors.
        'triage' appears exactly 2 times; requiring 3 must exclude it.
        """
        df, *_ = load_and_prepare_log(
            log_path="unused",
            meta_cols=META_COLUMNS_IGNORED,
            log=_make_typed_log(),
        )
        decision_pts = find_decision_points(df, min_instances=100)
        assert "triage" not in decision_pts

    def test_empty_dataframe_returns_empty_list(self):
        """find_decision_points on an empty DataFrame must return an empty list."""
        import pandas as pd
        empty_df = pd.DataFrame(columns=["concept:name", "next_activity"])
        assert find_decision_points(empty_df, min_instances=1) == []


# ===========================================================================
# Integration-only tests (skipped by default)
# ===========================================================================

@pytest.mark.integration
def test_discover_all_decision_rules_runs_on_real_log():
    """
    Full end-to-end test for discover_all_decision_rules.
    Requires a real XES log at the path specified — run with:
        pytest -m integration --log-path /path/to/log.xes
    """
    pytest.skip("Integration test: provide a real XES log path to run this.")
