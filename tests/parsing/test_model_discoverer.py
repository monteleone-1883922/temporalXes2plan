"""
Tests for parsing.model_discoverer.ModelDiscoverer.

Unit tests for _extract_activities_and_silent work with synthetic Transition
objects and need no event log.  Integration tests for discover() use a minimal
synthetic log built with the make_log/make_trace/make_event helpers.
"""
import pytest
from pm4py import PetriNet

from tests.helpers import _transition, make_event, make_trace, make_log
from parsing.model_discoverer import ModelDiscoverer
from models import PetriNetModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_log():
    """Minimal 3-trace log: A -> B -> C, sufficient to drive inductive miner."""
    return make_log(
        make_trace(make_event("A"), make_event("B"), make_event("C"), case_id="c1"),
        make_trace(make_event("A"), make_event("B"), make_event("C"), case_id="c2"),
        make_trace(make_event("A"), make_event("B"), make_event("C"), case_id="c3"),
    )


# ===========================================================================
# __init__
# ===========================================================================

class TestModelDiscovererInit:
    def test_default_algorithm_is_inductive(self):
        assert ModelDiscoverer().discovery_algorithm == "inductive"

    def test_invalid_algorithm_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported discovery algorithm"):
            ModelDiscoverer("nonexistent")

    def test_all_valid_algorithms_are_accepted(self):
        for algo in ["alpha", "inductive", "heuristics", "ilp"]:
            assert ModelDiscoverer(algo).discovery_algorithm == algo


# ===========================================================================
# _extract_activities_and_silent
# ===========================================================================

class TestExtractActivitiesAndSilent:
    def _call(self, transitions):
        return ModelDiscoverer()._extract_activities_and_silent(transitions)

    def test_labeled_transitions_appear_sanitized_in_activities(self):
        t1 = _transition("t1", "ER Registration")
        t2 = _transition("t2", "CRP Measurement")
        acts, silent = self._call([t1, t2])
        assert "er_registration" in acts
        assert "crp_measurement" in acts
        assert silent == {}

    def test_silent_transition_gets_tau_name_and_appears_in_activities(self):
        t = _transition("t1", None)
        acts, silent = self._call([t])
        assert t in silent
        tau_name = silent[t]
        assert tau_name.startswith("tau_")
        assert tau_name in acts

    def test_multiple_silent_transitions_get_unique_tau_names(self):
        transitions = [_transition(f"t{i}", None) for i in range(3)]
        acts, silent = self._call(transitions)
        tau_names = list(silent.values())
        assert len(set(tau_names)) == 3
        assert set(tau_names) <= acts

    def test_mixed_transitions_are_each_handled_correctly(self):
        t_labeled = _transition("tL", "Triage")
        t_silent = _transition("tS", None)
        acts, silent = self._call([t_labeled, t_silent])
        assert "triage" in acts
        assert t_silent in silent
        assert silent[t_silent] in acts
        assert t_labeled not in silent

    def test_empty_input_returns_empty_sets(self):
        acts, silent = self._call([])
        assert acts == set()
        assert silent == {}


# ===========================================================================
# discover  (integration — runs the real inductive miner)
# ===========================================================================

class TestDiscover:
    def test_returns_petrinet_model_not_tuple(self, simple_log):
        result = ModelDiscoverer().discover(simple_log)
        assert isinstance(result, PetriNetModel)

    def test_petrinet_field_is_petri_net_instance(self, simple_log):
        result = ModelDiscoverer().discover(simple_log)
        assert isinstance(result.petrinet, PetriNet)

    def test_initial_and_final_markings_are_non_empty(self, simple_log):
        result = ModelDiscoverer().discover(simple_log)
        assert len(result.initial_marking) > 0
        assert len(result.final_marking) > 0

    def test_activities_contains_sanitized_log_activity_names(self, simple_log):
        result = ModelDiscoverer().discover(simple_log)
        assert "a" in result.activities
        assert "b" in result.activities
        assert "c" in result.activities

    def test_silent_transitions_field_is_a_dict(self, simple_log):
        result = ModelDiscoverer().discover(simple_log)
        assert isinstance(result.silent_transitions, dict)
