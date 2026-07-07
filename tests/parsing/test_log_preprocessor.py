"""
Tests for parsing.log_preprocessor.LogPreprocessor.

All tests use synthetic PetriNetLog objects — no real event logs or discovery
algorithms are needed.
"""
import pytest
from typing import Dict, List, Optional, Set

from pm4py import PetriNet, Marking

from tests.helpers import _transition, _place
from parsing.log_preprocessor import LogPreprocessor
from models import (
    AnalysisConfig,
    FiringStep,
    PetriNetLog,
    PreprocessedLog,
    TraceExecution,
    TransitionFiringData,
)


# ---------------------------------------------------------------------------
# PetriNetLog factory helpers (same pattern as test_decision_mining)
# ---------------------------------------------------------------------------

def _step(
    transition: PetriNet.Transition,
    from_places: Set[PetriNet.Place],
    is_tau: bool = False,
    attributes: Optional[Dict] = None,
    activity_name: Optional[str] = None,
) -> FiringStep:
    return FiringStep(
        transition=transition,
        activity_name=activity_name or (transition.label or "tau"),
        is_tau=is_tau,
        from_places=from_places,
        attributes=attributes or {},
    )


def _execution(trace_id: str, steps: List[FiringStep]) -> TraceExecution:
    return TraceExecution(trace_id=trace_id, steps=steps)


def _pn_log(executions: List[TraceExecution]) -> PetriNetLog:
    return PetriNetLog(
        executions=executions,
        net=PetriNet("test"),
        initial_marking=Marking(),
        final_marking=Marking(),
    )


def _preprocessor(
    config: Optional[AnalysisConfig] = None,
    discretizer=None,
) -> LogPreprocessor:
    return LogPreprocessor(
        config=config or AnalysisConfig(),
        discretizer=discretizer,
    )


# ---------------------------------------------------------------------------
# Shared net structure:
#   p_in → t_pre → p_mid → t_work → p_out
# ---------------------------------------------------------------------------

def _simple_net():
    t_pre = _transition("t_pre", "Pre")
    t_work = _transition("t_work", "Work")
    p_in = _place("p_in")
    p_mid = _place("p_mid")
    p_out = _place("p_out")
    return t_pre, t_work, p_in, p_mid, p_out


# XOR structure:
#   p_in → t_src → p_xor → t_B
#                        → t_C
def _xor_net():
    t_src = _transition("t_src", "Src")
    t_B = _transition("t_B", "B")
    t_C = _transition("t_C", "C")
    p_in = _place("p_in")
    p_xor = _place("p_xor")
    return t_src, t_B, t_C, p_in, p_xor


# ===========================================================================
# Basic output structure
# ===========================================================================

class TestPreprocessOutputStructure:
    def test_returns_preprocessed_log(self):
        result = _preprocessor().preprocess(_pn_log([]), {})
        assert isinstance(result, PreprocessedLog)

    def test_empty_log_produces_empty_indexes(self):
        result = _preprocessor().preprocess(_pn_log([]), {})
        assert result.transition_firings == {}
        assert result.xor_firings == {}

    def test_transition_firings_keyed_by_raw_activity_name(self):
        t = _transition("t1", "ER Registration")
        p = _place("p")
        execs = [_execution("c1", [_step(t, {p})])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        assert "ER Registration" in result.transition_firings

    def test_firing_data_has_correct_activity_name(self):
        t = _transition("t1", "Check CRP")
        p = _place("p")
        execs = [_execution("c1", [_step(t, {p})])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["Check CRP"][0]
        assert fd.activity_name == "Check CRP"

    def test_firing_data_from_places_is_frozenset(self):
        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution("c1", [_step(t, {p})])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["A"][0]
        assert isinstance(fd.from_places, frozenset)
        assert p in fd.from_places


# ===========================================================================
# Tau transitions excluded
# ===========================================================================

class TestTauExclusion:
    def test_tau_steps_produce_no_firing_data(self):
        t_tau = _transition("t_tau", None)
        p = _place("p")
        execs = [_execution("c1", [
            _step(t_tau, {p}, is_tau=True, attributes={"x": "1"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        assert result.transition_firings == {}

    def test_tau_steps_do_not_update_state(self):
        """Attributes set by a tau step must not appear in later pre_state."""
        t_tau = _transition("t_tau", None)
        t_work = _transition("t_work", "Work")
        p = _place("p")
        execs = [_execution("c1", [
            _step(t_tau, {p}, is_tau=True, attributes={"hidden": "yes"}),
            _step(t_work, {p}, attributes={"score": "1"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["Work"][0]
        assert "hidden" not in fd.pre_state


# ===========================================================================
# Causal ordering (pre_state)
# ===========================================================================

class TestCausalOrdering:
    def setup_method(self):
        self.t_pre, self.t_work, self.p_in, self.p_mid, self.p_out = _simple_net()

    def test_pre_state_contains_attributes_from_prior_steps(self):
        execs = [_execution("c1", [
            _step(self.t_pre, {self.p_in}, attributes={"risk": "high"}),
            _step(self.t_work, {self.p_mid}, attributes={"score": "1"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["Work"][0]
        assert fd.pre_state.get("risk") == "high"

    def test_pre_state_does_not_contain_own_step_new_attrs(self):
        """Attributes first introduced by the step itself must not be in pre_state."""
        execs = [_execution("c1", [
            _step(self.t_work, {self.p_mid}, attributes={"brand_new": "x"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["Work"][0]
        assert "brand_new" not in fd.pre_state

    def test_pre_state_contains_overwritten_value_from_earlier_step(self):
        """If a prior step set attr=A and the current step sets attr=B,
        pre_state must contain A (the prior value), not B."""
        execs = [_execution("c1", [
            _step(self.t_pre, {self.p_in}, attributes={"status": "old"}),
            _step(self.t_work, {self.p_mid}, attributes={"status": "new"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["Work"][0]
        assert fd.pre_state["status"] == "old"

    def test_state_accumulates_across_steps_in_execution(self):
        """Multiple prior steps contribute to pre_state of a later step."""
        t_a = _transition("t_a", "A")
        t_b = _transition("t_b", "B")
        t_c = _transition("t_c", "C")
        p = _place("p")
        execs = [_execution("c1", [
            _step(t_a, {p}, attributes={"x": "1"}),
            _step(t_b, {p}, attributes={"y": "2"}),
            _step(t_c, {p}, attributes={"z": "3"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["C"][0]
        assert fd.pre_state == {"x": "1", "y": "2"}

    def test_state_resets_between_executions(self):
        """Each execution starts with an empty state — no leakage."""
        t = _transition("t1", "A")
        p = _place("p")
        execs = [
            _execution("c1", [_step(t, {p}, attributes={"x": "1"})]),
            _execution("c2", [_step(t, {p}, attributes={"y": "2"})]),
        ]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        # Second execution's firing must not see x from first execution.
        fd1 = result.transition_firings["A"][0]
        fd2 = result.transition_firings["A"][1]
        assert fd1.pre_state == {}
        assert "x" not in fd2.pre_state

    def test_target_own_attrs_appear_in_state_of_later_firing(self):
        """If the same transition fires twice in one execution, the second
        firing's pre_state must include values written by the first."""
        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution("c1", [
            _step(t, {p}, attributes={"x": "first"}),
            _step(t, {p}, attributes={"x": "second"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        firings = result.transition_firings["A"]
        assert len(firings) == 2
        assert firings[0].pre_state == {}
        assert firings[1].pre_state == {"x": "first"}


# ===========================================================================
# changed_attrs computation
# ===========================================================================

class TestChangedAttrs:
    def test_first_appearance_counts_as_change(self):
        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution("c1", [_step(t, {p}, attributes={"x": "new"})])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["A"][0]
        assert fd.changed_attrs == {"x": "new"}

    def test_same_value_as_state_not_counted(self):
        """If the attribute already has the same value, it is NOT a change."""
        t_pre = _transition("t_pre", "Pre")
        t_work = _transition("t_work", "Work")
        p = _place("p")
        execs = [_execution("c1", [
            _step(t_pre, {p}, attributes={"x": "same"}),
            _step(t_work, {p}, attributes={"x": "same"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["Work"][0]
        assert fd.changed_attrs == {}

    def test_different_value_counted_as_change(self):
        t_pre = _transition("t_pre", "Pre")
        t_work = _transition("t_work", "Work")
        p = _place("p")
        execs = [_execution("c1", [
            _step(t_pre, {p}, attributes={"x": "old"}),
            _step(t_work, {p}, attributes={"x": "new"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["Work"][0]
        assert fd.changed_attrs == {"x": "new"}

    def test_none_value_excluded_from_changed(self):
        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution("c1", [_step(t, {p}, attributes={"x": None})])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["A"][0]
        assert "x" not in fd.changed_attrs

    def test_multiple_attrs_some_changed_some_not(self):
        t_pre = _transition("t_pre", "Pre")
        t_work = _transition("t_work", "Work")
        p = _place("p")
        execs = [_execution("c1", [
            _step(t_pre, {p}, attributes={"a": "1", "b": "2"}),
            _step(t_work, {p}, attributes={"a": "1", "b": "3", "c": "new"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        fd = result.transition_firings["Work"][0]
        # a unchanged, b changed, c first appearance
        assert "a" not in fd.changed_attrs
        assert fd.changed_attrs["b"] == "3"
        assert fd.changed_attrs["c"] == "new"


# ===========================================================================
# ignored_attributes filtering
# ===========================================================================

class TestIgnoredAttributes:
    def test_ignored_attributes_excluded_from_pre_state(self):
        cfg = AnalysisConfig(ignored_attributes={"concept:name", "secret"})
        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution("c1", [
            _step(t, {p}, attributes={"secret": "hide", "visible": "show"}),
            _step(t, {p}, attributes={}),
        ])]
        result = LogPreprocessor(config=cfg).preprocess(_pn_log(execs), {})
        fd = result.transition_firings["A"][1]
        assert "secret" not in fd.pre_state
        assert "visible" in fd.pre_state

    def test_ignored_attributes_excluded_from_changed_attrs(self):
        cfg = AnalysisConfig(ignored_attributes={"concept:name", "secret"})
        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution("c1", [
            _step(t, {p}, attributes={"secret": "hide", "visible": "show"}),
        ])]
        result = LogPreprocessor(config=cfg).preprocess(_pn_log(execs), {})
        fd = result.transition_firings["A"][0]
        assert "secret" not in fd.changed_attrs
        assert "visible" in fd.changed_attrs


# ===========================================================================
# Discretizer integration
# ===========================================================================

class TestDiscretizerIntegration:
    def test_numeric_value_discretized_in_changed_attrs(self):
        from parsing.discretizer import Discretizer
        disc = Discretizer()
        disc.boundaries = {"crp": [6.0]}
        pp = LogPreprocessor(discretizer=disc)

        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution("c1", [_step(t, {p}, attributes={"crp": 4.5})])]
        result = pp.preprocess(_pn_log(execs), {})
        fd = result.transition_firings["A"][0]
        assert fd.changed_attrs["crp"] == "lte_6_0"

    def test_numeric_value_discretized_in_pre_state(self):
        from parsing.discretizer import Discretizer
        disc = Discretizer()
        disc.boundaries = {"crp": [6.0]}
        pp = LogPreprocessor(discretizer=disc)

        t_pre = _transition("t_pre", "Pre")
        t_work = _transition("t_work", "Work")
        p = _place("p")
        execs = [_execution("c1", [
            _step(t_pre, {p}, attributes={"crp": 4.5}),
            _step(t_work, {p}, attributes={}),
        ])]
        result = pp.preprocess(_pn_log(execs), {})
        fd = result.transition_firings["Work"][0]
        assert fd.pre_state["crp"] == "lte_6_0"

    def test_attr_without_boundaries_kept_raw(self):
        from parsing.discretizer import Discretizer
        disc = Discretizer()
        disc.boundaries = {}
        pp = LogPreprocessor(discretizer=disc)

        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution("c1", [_step(t, {p}, attributes={"crp": 4.5})])]
        result = pp.preprocess(_pn_log(execs), {})
        fd = result.transition_firings["A"][0]
        assert fd.changed_attrs["crp"] == 4.5


# ===========================================================================
# XOR-split indexing
# ===========================================================================

class TestXorIndexing:
    def setup_method(self):
        self.t_src, self.t_B, self.t_C, self.p_in, self.p_xor = _xor_net()
        self.xor_splits = {self.p_xor: [self.t_B, self.t_C]}

    def test_xor_firings_keyed_by_place_name(self):
        execs = [_execution("c1", [
            _step(self.t_src, {self.p_in}),
            _step(self.t_B, {self.p_xor}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), self.xor_splits)
        assert "p_xor" in result.xor_firings

    def test_xor_firings_counts_match_traversals(self):
        execs = [
            _execution("c1", [
                _step(self.t_src, {self.p_in}),
                _step(self.t_B, {self.p_xor}),
            ]),
            _execution("c2", [
                _step(self.t_src, {self.p_in}),
                _step(self.t_C, {self.p_xor}),
            ]),
            _execution("c3", [
                _step(self.t_src, {self.p_in}),
                _step(self.t_B, {self.p_xor}),
            ]),
        ]
        result = _preprocessor().preprocess(_pn_log(execs), self.xor_splits)
        assert len(result.xor_firings["p_xor"]) == 3

    def test_xor_firings_activity_names_are_branch_names(self):
        execs = [
            _execution("c1", [_step(self.t_B, {self.p_xor})]),
            _execution("c2", [_step(self.t_C, {self.p_xor})]),
        ]
        result = _preprocessor().preprocess(_pn_log(execs), self.xor_splits)
        names = {fd.activity_name for fd in result.xor_firings["p_xor"]}
        assert names == {"B", "C"}

    def test_non_branch_transition_from_xor_place_not_indexed(self):
        """A transition that fires from the XOR place but is NOT one of its
        registered branches must NOT appear in xor_firings."""
        t_other = _transition("t_other", "Other")
        execs = [_execution("c1", [_step(t_other, {self.p_xor})])]
        result = _preprocessor().preprocess(_pn_log(execs), self.xor_splits)
        assert result.xor_firings.get("p_xor", []) == []

    def test_xor_and_transition_indexes_share_same_objects(self):
        """xor_firings and transition_firings must point to the same
        TransitionFiringData instances (no data duplication)."""
        execs = [_execution("c1", [
            _step(self.t_src, {self.p_in}),
            _step(self.t_B, {self.p_xor}, attributes={"x": "1"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), self.xor_splits)
        fd_trans = result.transition_firings["B"][0]
        fd_xor = result.xor_firings["p_xor"][0]
        assert fd_trans is fd_xor

    def test_xor_firings_have_correct_pre_state(self):
        """The pre_state of a XOR traversal must contain attributes set by
        prior steps (causal ordering preserved through XOR indexing)."""
        execs = [_execution("c1", [
            _step(self.t_src, {self.p_in}, attributes={"risk": "high"}),
            _step(self.t_B, {self.p_xor}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), self.xor_splits)
        fd = result.xor_firings["p_xor"][0]
        assert fd.pre_state.get("risk") == "high"

    def test_multiple_xor_places_indexed_independently(self):
        p2 = _place("p_xor2")
        t_D = _transition("t_D", "D")
        t_E = _transition("t_E", "E")
        xor_splits = {
            self.p_xor: [self.t_B, self.t_C],
            p2: [t_D, t_E],
        }
        execs = [_execution("c1", [
            _step(self.t_B, {self.p_xor}),
            _step(t_D, {p2}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), xor_splits)
        assert len(result.xor_firings["p_xor"]) == 1
        assert len(result.xor_firings["p_xor2"]) == 1


# ===========================================================================
# Firing counts (transition_firings population)
# ===========================================================================

class TestTransitionFiringCounts:
    def test_single_transition_single_firing(self):
        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution("c1", [_step(t, {p})])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        assert len(result.transition_firings["A"]) == 1

    def test_multiple_firings_across_executions(self):
        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution(f"c{i}", [_step(t, {p})]) for i in range(7)]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        assert len(result.transition_firings["A"]) == 7

    def test_multiple_transitions_counted_independently(self):
        t_a = _transition("t_a", "A")
        t_b = _transition("t_b", "B")
        p = _place("p")
        execs = [
            _execution("c1", [_step(t_a, {p}), _step(t_b, {p})]),
            _execution("c2", [_step(t_a, {p})]),
        ]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        assert len(result.transition_firings["A"]) == 2
        assert len(result.transition_firings["B"]) == 1

    def test_same_transition_fires_twice_in_one_execution(self):
        t = _transition("t1", "A")
        p = _place("p")
        execs = [_execution("c1", [
            _step(t, {p}, attributes={"x": "1"}),
            _step(t, {p}, attributes={"x": "2"}),
        ])]
        result = _preprocessor().preprocess(_pn_log(execs), {})
        assert len(result.transition_firings["A"]) == 2
