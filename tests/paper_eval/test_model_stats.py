"""Tests for paper_eval.model_stats (descriptive statistics of a model)."""
import pytest

from encoding.prepared_input import (
    PreparedDomainInput, PreparedEffectGroup, PreparedTransition, PreparedXorBranch,
)
from models import Guard
from paper_eval import model_stats as ms


class TestSpread:
    def test_min_median_mean_max(self):
        s = ms._spread([1.0, 2.0, 9.0])
        assert (s["n"], s["min"], s["median"], s["max"]) == (3, 1.0, 2.0, 9.0)
        assert s["mean"] == pytest.approx(4.0)

    def test_empty(self):
        assert ms._spread([]) == {"n": 0, "min": None, "median": None, "mean": None, "max": None}


class TestAttributesRead:
    @staticmethod
    def _prepared(branch_guards=None, group_guard=None, preconditions=None):
        transition = PreparedTransition(
            activity_name="A", input_places=["p"], preconditions=preconditions or [],
            effect_groups=[PreparedEffectGroup(assignments=[("w", "x")], guard=group_guard or [], probability=1.0)],
        )
        branches = {"p": [PreparedXorBranch("A", branch_guards, 1.0)]} if branch_guards is not None else {}
        return PreparedDomainInput(transitions={"A": transition}, xor_branches=branches, attribute_catalog={})

    def test_written_only_attributes_are_not_read(self):
        assert ms.attributes_read(self._prepared()) == set()  # "w" is assigned, never read

    def test_xor_branch_guard_counts(self):
        assert ms.attributes_read(self._prepared(branch_guards=[[Guard("a", "x")]])) == {"a"}

    def test_effect_group_guard_counts(self):
        assert ms.attributes_read(self._prepared(group_guard=[[Guard("b", "y"), Guard("c", None, True)]])) == {"b", "c"}

    def test_preconditions_count(self):
        assert ms.attributes_read(self._prepared(preconditions=[[Guard("d", "z")]])) == {"d"}

    def test_branch_without_guards_reads_nothing(self):
        assert ms.attributes_read(self._prepared(branch_guards=None)) == set()
