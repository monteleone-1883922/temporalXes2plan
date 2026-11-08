"""Tests for paper_eval.model_tables (the rows build_model writes as JSONL)."""
import logging

import pytest

from encoding.prepared_input import PreparedEffectGroup, PreparedTransition
from models import ActionDurationStats, AttributeCatalogEntry
from paper_eval import model_tables as mt


def _entry(kind, values, bins=None):
    return AttributeCatalogEntry(attribute_type=kind, possible_values=set(values), bin_boundaries=bins)


def _rows(**overrides):
    """attribute_rows on a toy log with one attribute of every fate, as {attribute: row}."""
    args = dict(
        parse_catalog={
            "crp": _entry("categorical", ["high", "low"]),
            "lactic": _entry("numerical", ["gte_2", "lte_2", "none"], [2.0]),
            "flag": _entry("boolean", [False, "False", True]),
        },
        final_catalog={
            "crp": _entry("categorical", ["high", "low"]),
            "flag": _entry("boolean", [False, "False", True]),
        },
        unused={"lactic"},
        noise={"counter": "N3 counter-like"},
        noise_details={"counter": {"type": "numerical", "n_distinct": 99}},
        ignored_by_user=["resource"],
        kept_by_user=["crp"],
        analysed_types={"crp": "categorical", "lactic": "numerical", "flag": "boolean", "idle": "categorical"},
    )
    args.update(overrides)
    rows, anomalies = mt.attribute_rows(**args)
    return {r["attribute"]: r for r in rows}, anomalies


class TestAttributeRows:
    def test_one_row_per_attribute_with_its_fate(self):
        rows, anomalies = _rows()
        assert {a: r["status"] for a, r in rows.items()} == {
            "crp": "kept", "flag": "kept", "lactic": "removed_unused",
            "counter": "removed_noise", "resource": "ignored_by_user", "idle": "removed_cascade"}
        assert anomalies == []

    def test_an_attribute_removed_as_unused_keeps_type_values_and_boundaries(self):
        row = _rows()[0]["lactic"]
        assert (row["type"], row["possible_values"], row["bin_boundaries"]) == (
            "numerical", ["gte_2", "lte_2", "none"], [2.0])
        assert (row["n_values"], row["n_bin_boundaries"]) == (3, 1)

    def test_n_values_counts_distinct_labels_not_set_elements(self):
        # the set holds the boolean False and the string "False": written the same way
        row = _rows()[0]["flag"]
        assert row["possible_values"] == ["False", "True"]
        assert row["n_values"] == 2

    def test_boundaries_exist_only_for_numerical_attributes(self):
        rows = _rows()[0]
        assert rows["crp"]["bin_boundaries"] is None and rows["crp"]["n_bin_boundaries"] is None
        assert rows["flag"]["n_bin_boundaries"] is None

    def test_a_numerical_attribute_without_cut_points_has_zero_not_none(self):
        parse = {"x": _entry("numerical", ["none"], [])}
        row = _rows(parse_catalog=parse, final_catalog=parse, unused=set(), noise={}, noise_details={},
                    ignored_by_user=[], kept_by_user=[])[0]["x"]
        assert (row["bin_boundaries"], row["n_bin_boundaries"]) == ([], 0)

    def test_a_noise_attribute_keeps_its_type_and_distinct_values_but_has_no_catalog_data(self):
        row = _rows()[0]["counter"]
        assert (row["type"], row["reason"], row["n_distinct_observed"]) == ("numerical", "N3 counter-like", 99)
        assert (row["n_values"], row["possible_values"], row["bin_boundaries"]) == (None, None, None)

    def test_an_ignored_attribute_was_never_analysed(self):
        row = _rows()[0]["resource"]
        assert row["status"] == "ignored_by_user"
        assert all(row[c] is None for c in ("type", "n_values", "possible_values", "bin_boundaries"))

    def test_an_analysed_attribute_that_never_reached_the_catalog_is_accounted_for(self):
        row = _rows()[0]["idle"]
        assert (row["type"], row["status"]) == ("categorical", "removed_cascade")
        assert row["reason"]
        assert all(row[c] is None for c in ("n_values", "possible_values", "bin_boundaries"))

    def test_analysed_types_are_optional(self):
        rows, _ = _rows(analysed_types=None)
        assert "idle" not in rows

    def test_kept_by_user_is_marked(self):
        rows = _rows()[0]
        assert rows["crp"]["kept_by_user"] is True and rows["flag"]["kept_by_user"] is False

    def test_the_sizes_add_up(self):
        # the analysed attributes: kept + removed_unused = size of the catalog before filter 2
        rows = _rows()[0]
        analysed = [r for r in rows.values() if r["status"] in ("kept", "removed_unused")]
        assert len(analysed) == 3
        assert sum(r["status"] == "kept" for r in analysed) == 2


class TestAnomalies:
    def test_an_unused_attribute_in_the_final_catalog_is_reported(self, caplog):
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            rows, anomalies = _rows(unused={"lactic", "crp"})
        assert any("'crp'" in a and "unused" in a for a in anomalies)
        assert "'crp'" in caplog.text
        assert rows["crp"]["status"] == "kept"

    def test_a_noise_attribute_that_reached_the_catalog_is_reported_and_not_duplicated(self):
        rows, anomalies = _rows(noise={"crp": "N1 constant"}, noise_details={})
        assert any("'crp'" in a and "noise" in a for a in anomalies)
        assert rows["crp"]["status"] == "kept"

    def test_an_ignored_attribute_in_the_catalog_is_reported(self):
        _, anomalies = _rows(ignored_by_user=["lactic"])
        assert any("'lactic'" in a and "ignore" in a for a in anomalies)

    def test_an_attribute_both_noise_and_ignored_is_reported(self):
        rows, anomalies = _rows(ignored_by_user=["counter"])
        assert any("'counter'" in a for a in anomalies)
        assert rows["counter"]["status"] == "removed_noise"

    def test_an_attribute_lost_without_a_reason_is_reported(self):
        rows, anomalies = _rows(unused=set())
        assert any("'lactic'" in a for a in anomalies)
        assert rows["lactic"]["status"] == mt.REMOVED_UNKNOWN

    def test_a_final_attribute_missing_from_the_parse_catalog_is_reported(self):
        final = {"crp": _entry("categorical", ["high"]), "ghost": _entry("categorical", ["a"])}
        rows, anomalies = _rows(final_catalog=final)
        assert any("'ghost'" in a for a in anomalies)
        assert rows["ghost"]["status"] == "kept"

    def test_a_consistent_input_has_no_anomaly_and_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            _, anomalies = _rows()
        assert anomalies == [] and caplog.text == ""


class TestTransitionDurationRows:
    def test_one_row_per_transition_with_the_duration_statistics(self):
        duration = ActionDurationStats(
            effective_min=1.0, effective_max=9.0, source="inter_event", mean=4.0, std_dev=2.0,
            observed_min=0.5, observed_max=12.0, count=7)
        transitions = {
            "A": PreparedTransition("A", ["p1"], [], [PreparedEffectGroup([], [], 1.0)] * 3, cost=0.5,
                                    duration=duration),
            "tau_1": PreparedTransition("tau_1", ["p2"], [], [], duration=None),
        }
        rows = {r["transition"]: r for r in mt.transition_duration_rows(transitions)}
        assert rows["A"] == {
            "transition": "A", "activity_name": "A", "n_effect_groups": 3, "cost": 0.5, "has_duration": True,
            "effective_min": 1.0, "effective_max": 9.0, "mean": 4.0, "std_dev": 2.0,
            "observed_min": 0.5, "observed_max": 12.0, "count": 7, "source": "inter_event"}
        assert rows["tau_1"]["has_duration"] is False
        assert all(rows["tau_1"][c] is None for c in (
            "effective_min", "effective_max", "mean", "std_dev", "observed_min", "observed_max", "count", "source"))
