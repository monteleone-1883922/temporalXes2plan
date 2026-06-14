"""Unit tests for Parser._inject_xor_taus().

Tests invoke the method directly on synthetic TransitionInfo / predecessor
dicts, without running the full parsing pipeline.
"""
import math

import pytest

from models import EffectInfo, TransitionInfo, XorBranchInfo
from parsing.xes_parser import Parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _appearance2_effect(attr: str = "diagnosis") -> EffectInfo:
    return EffectInfo(
        attribute=attr,
        presence_probability=0.7,
        appearance_level=2,
        appearance_guards=None,
        value_probabilities={"flu": 0.6, "cold": 0.4},
        value_level=2,
        value_guards=None,
    )


def _deterministic_effect(attr: str = "diagnosis") -> EffectInfo:
    return EffectInfo(
        attribute=attr,
        presence_probability=1.0,
        appearance_level=1,
        appearance_guards=None,
        value_probabilities={"flu": 1.0},
        value_level=1,
        value_guards=None,
    )


def _xor2_branch(probability: float = 0.6) -> XorBranchInfo:
    return XorBranchInfo(
        probability=probability,
        total_samples=100,
        cascade_level=2,
        guards=None,
    )


def _inject(transitions, transition_predecessors):
    """Call _inject_xor_taus without constructing a full Parser."""
    # We bind the method directly since it doesn't access self beyond dict ops
    return Parser._inject_xor_taus(None, transitions, transition_predecessors)


# ---------------------------------------------------------------------------
# Injection triggered
# ---------------------------------------------------------------------------

class TestInjectionTriggered:

    def test_tau_added_to_transitions(self):
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(0.6),
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        _inject(transitions, preds)

        assert "xor_tau_left" in transitions

    def test_tau_has_no_effects(self):
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(0.6),
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        _inject(transitions, preds)

        assert transitions["xor_tau_left"].effects == {}

    def test_tau_inherits_xor_branch(self):
        branch = _xor2_branch(0.6)
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, branch,
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        _inject(transitions, preds)

        assert transitions["xor_tau_left"].xor_branch is branch

    def test_tau_inherits_input_places(self):
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(),
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        _inject(transitions, preds)

        assert transitions["xor_tau_left"].input_places == ["p_xor"]

    def test_original_xor_branch_cleared(self):
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(),
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        _inject(transitions, preds)

        assert transitions["left"].xor_branch is None

    def test_original_input_places_redirected(self):
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(),
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        _inject(transitions, preds)

        assert transitions["left"].input_places == ["xor_tau_left"]

    def test_tau_predecessor_entry_added(self):
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(),
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        _inject(transitions, preds)

        assert preds["xor_tau_left"] == ["p_xor"]

    def test_original_predecessor_redirected(self):
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(),
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        _inject(transitions, preds)

        assert preds["left"] == ["xor_tau_left"]

    def test_returned_set_contains_tau_name(self):
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(),
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        result = _inject(transitions, preds)

        assert result == {"xor_tau_left"}

    def test_multiple_transitions_both_injected(self):
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(0.6),
                                   {"diagnosis": _appearance2_effect()}),
            "right": TransitionInfo("right", ["p_xor"], 40, _xor2_branch(0.4),
                                    {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"], "right": ["p_xor"]}
        result = _inject(transitions, preds)

        assert result == {"xor_tau_left", "xor_tau_right"}
        assert "xor_tau_left" in transitions
        assert "xor_tau_right" in transitions


# ---------------------------------------------------------------------------
# Injection NOT triggered
# ---------------------------------------------------------------------------

class TestInjectionNotTriggered:

    def test_no_xor_branch_no_injection(self):
        transitions = {
            "left": TransitionInfo("left", ["p"], 60, None,
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p"]}
        result = _inject(transitions, preds)

        assert result == set()
        assert "xor_tau_left" not in transitions

    def test_cascade_level1_no_injection(self):
        transitions = {
            "left": TransitionInfo(
                "left", ["p_xor"], 60,
                XorBranchInfo(0.6, 100, cascade_level=1,
                              guards=[[]], ),
                {"diagnosis": _appearance2_effect()},
            ),
        }
        preds = {"left": ["p_xor"]}
        result = _inject(transitions, preds)

        assert result == set()

    def test_cascade_level3_no_injection(self):
        transitions = {
            "left": TransitionInfo(
                "left", ["p_xor"], 60,
                XorBranchInfo(0.6, 100, cascade_level=3, guards=None),
                {"diagnosis": _appearance2_effect()},
            ),
        }
        preds = {"left": ["p_xor"]}
        result = _inject(transitions, preds)

        assert result == set()

    def test_cascade_level2_no_effects_no_injection(self):
        """cascade_level=2 but no effects at all → no tau needed."""
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(), {}),
        }
        preds = {"left": ["p_xor"]}
        result = _inject(transitions, preds)

        assert result == set()

    def test_cascade_level2_only_deterministic_effects_no_injection(self):
        """cascade_level=2 but all effects are appearance_level=1 → no conflict."""
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(),
                                   {"diagnosis": _deterministic_effect()}),
        }
        preds = {"left": ["p_xor"]}
        result = _inject(transitions, preds)

        assert result == set()

    def test_mixed_transitions_only_candidate_injected(self):
        """Only the transition with cascade_level=2 + appearance_level=2 gets a tau."""
        transitions = {
            "a": TransitionInfo("a", ["p1"], 50, _xor2_branch(0.5),
                                {"diagnosis": _appearance2_effect()}),
            "b": TransitionInfo("b", ["p1"], 50, _xor2_branch(0.5),
                                {"diagnosis": _deterministic_effect()}),
        }
        preds = {"a": ["p1"], "b": ["p1"]}
        result = _inject(transitions, preds)

        assert result == {"xor_tau_a"}
        assert "xor_tau_b" not in transitions
        assert transitions["b"].xor_branch is not None


# ---------------------------------------------------------------------------
# Isolation: tau injection does not mutate predecessor list objects
# ---------------------------------------------------------------------------

class TestNoSideEffects:

    def test_tau_predecessors_are_copy(self):
        """Changing the tau predecessor list must not affect the original."""
        transitions = {
            "left": TransitionInfo("left", ["p_xor"], 60, _xor2_branch(),
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        _inject(transitions, preds)

        # Mutate the tau entry — original (now "left") must be unaffected
        preds["xor_tau_left"].append("extra")
        assert preds["left"] == ["xor_tau_left"]

    def test_tau_input_places_are_copy(self):
        """tau.input_places must be independent from the original list."""
        original_places = ["p_xor"]
        transitions = {
            "left": TransitionInfo("left", original_places, 60, _xor2_branch(),
                                   {"diagnosis": _appearance2_effect()}),
        }
        preds = {"left": ["p_xor"]}
        _inject(transitions, preds)

        tau = transitions["xor_tau_left"]
        tau.input_places.append("extra")
        assert original_places == ["p_xor"]
