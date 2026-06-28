"""
Tests for parsing.model_discoverer.ModelDiscoverer.

Unit tests for _extract_activities_and_silent work with synthetic Transition
objects and need no event log.  Integration tests for discover() use a minimal
synthetic log built with the make_log/make_trace/make_event helpers.
"""
import pytest
from pm4py import PetriNet
from pm4py.objects.petri_net.obj import Marking

from tests.helpers import _transition, _place, _arc, make_event, make_trace, make_log
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
        for algo in ["alpha", "inductive", "heuristics", "ilp", "powl"]:
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

    def test_trans_inputs_and_trans_outputs_are_present(self, simple_log):
        """discover() must populate trans_inputs and trans_outputs via build_arc_maps()."""
        result = ModelDiscoverer().discover(simple_log)
        assert result.trans_inputs is not None
        assert result.trans_outputs is not None

    def test_trans_inputs_keys_are_transitions(self, simple_log):
        """trans_inputs keys must be PetriNet.Transition objects from the discovered net."""
        result = ModelDiscoverer().discover(simple_log)
        net_transitions = set(result.petrinet.transitions)
        for transition in result.trans_inputs:
            assert transition in net_transitions

    def test_trans_outputs_keys_are_transitions(self, simple_log):
        """trans_outputs keys must be PetriNet.Transition objects from the discovered net."""
        result = ModelDiscoverer().discover(simple_log)
        net_transitions = set(result.petrinet.transitions)
        for transition in result.trans_outputs:
            assert transition in net_transitions

    def test_trans_inputs_values_are_sets_of_places(self, simple_log):
        """Every value in trans_inputs must be a set containing Place objects."""
        result = ModelDiscoverer().discover(simple_log)
        net_places = set(result.petrinet.places)
        for places in result.trans_inputs.values():
            assert isinstance(places, set)
            assert places <= net_places

    def test_trans_maps_cover_all_transitions_with_arcs(self, simple_log):
        """
        Every transition that has at least one arc must appear in trans_inputs
        or trans_outputs (or both). A transition with no arcs would be isolated
        and is not expected in a well-formed discovered net.
        """
        result = ModelDiscoverer().discover(simple_log)
        all_keyed = set(result.trans_inputs) | set(result.trans_outputs)
        for t in result.petrinet.transitions:
            assert t in all_keyed

    # --- place_inputs ---

    def test_place_inputs_is_present(self, simple_log):
        result = ModelDiscoverer().discover(simple_log)
        assert result.place_inputs is not None

    def test_place_inputs_keys_are_places_from_the_net(self, simple_log):
        result = ModelDiscoverer().discover(simple_log)
        net_places = set(result.petrinet.places)
        for place in result.place_inputs:
            assert place in net_places

    def test_place_inputs_values_are_lists_of_transitions(self, simple_log):
        result = ModelDiscoverer().discover(simple_log)
        net_transitions = set(result.petrinet.transitions)
        for transitions in result.place_inputs.values():
            assert isinstance(transitions, list)
            for t in transitions:
                assert t in net_transitions

    def test_place_inputs_is_inverse_of_trans_outputs(self, simple_log):
        """For every Trans→Place arc, the place must list that transition in place_inputs."""
        result = ModelDiscoverer().discover(simple_log)
        for transition, output_places in result.trans_outputs.items():
            for place in output_places:
                assert transition in result.place_inputs.get(place, [])

    def test_and_joins_field_does_not_exist(self, simple_log):
        """and_joins was removed from PetriNetModel; accessing it must raise AttributeError."""
        result = ModelDiscoverer().discover(simple_log)
        assert not hasattr(result, "and_joins")


# ===========================================================================
# prune_transitions
# ===========================================================================

def _xor_model() -> PetriNetModel:
    """Build a PetriNetModel for a XOR split with two branches.

        p_in -> t_src -> p_xor -> t_b -> p_out_b
                              -> t_c -> p_out_c

    Returns a model whose indexes are consistent with the net structure.
    """
    net = PetriNet("xor")
    p_in, p_xor, p_out_b, p_out_c = (
        _place("p_in"), _place("p_xor"), _place("p_out_b"), _place("p_out_c")
    )
    t_src, t_b, t_c = (
        _transition("t_src", "src"),
        _transition("t_b", "b"),
        _transition("t_c", "c"),
    )
    for p in (p_in, p_xor, p_out_b, p_out_c):
        net.places.add(p)
    for t in (t_src, t_b, t_c):
        net.transitions.add(t)
    for a in (
        _arc(p_in, t_src), _arc(t_src, p_xor),
        _arc(p_xor, t_b), _arc(t_b, p_out_b),
        _arc(p_xor, t_c), _arc(t_c, p_out_c),
    ):
        net.arcs.add(a)

    return PetriNetModel(
        petrinet=net,
        initial_marking=Marking({p_in: 1}),
        final_marking=Marking({p_out_b: 1}),
        activities={"src", "b", "c"},
        silent_transitions={},
        trans_inputs={t_src: {p_in}, t_b: {p_xor}, t_c: {p_xor}},
        trans_outputs={t_src: {p_xor}, t_b: {p_out_b}, t_c: {p_out_c}},
        xor_splits={p_xor: [t_b, t_c]},
        place_inputs={p_xor: [t_src], p_out_b: [t_b], p_out_c: [t_c]},
    )


def _t_named(model: PetriNetModel, label: str) -> PetriNet.Transition:
    """Return the transition with the given label from the model's net."""
    return next(t for t in model.petrinet.transitions if t.label == label)


class TestPruneTransitions:
    def test_empty_to_remove_returns_self(self):
        model = _xor_model()
        assert model.prune_transitions(set()) is model

    def test_pruned_transition_removed_from_net(self):
        model = _xor_model()
        t_c = _t_named(model, "c")
        pruned = model.prune_transitions({t_c})
        assert t_c not in pruned.petrinet.transitions

    def test_arcs_of_pruned_transition_removed(self):
        model = _xor_model()
        t_c = _t_named(model, "c")
        pruned = model.prune_transitions({t_c})
        for a in pruned.petrinet.arcs:
            assert a.source is not t_c and a.target is not t_c

    def test_orphaned_place_removed(self):
        model = _xor_model()
        t_c = _t_named(model, "c")
        pruned = model.prune_transitions({t_c})
        names = {p.name for p in pruned.petrinet.places}
        assert "p_out_c" not in names

    def test_marking_place_not_pruned_even_if_orphaned(self):
        model = _xor_model()
        # Removing t_b orphans p_out_b, but it is in the final marking.
        t_b = _t_named(model, "b")
        pruned = model.prune_transitions({t_b})
        names = {p.name for p in pruned.petrinet.places}
        assert "p_out_b" in names

    def test_xor_split_collapses_to_single_branch(self):
        model = _xor_model()
        t_c = _t_named(model, "c")
        pruned = model.prune_transitions({t_c})
        # Only one branch remains, so p_xor is no longer a XOR split.
        assert not pruned.xor_splits

    def test_activities_updated(self):
        model = _xor_model()
        t_c = _t_named(model, "c")
        pruned = model.prune_transitions({t_c})
        assert pruned.activities == {"src", "b"}

    def test_trans_inputs_and_outputs_drop_pruned(self):
        model = _xor_model()
        t_c = _t_named(model, "c")
        pruned = model.prune_transitions({t_c})
        assert t_c not in pruned.trans_inputs
        assert t_c not in pruned.trans_outputs

    def test_place_inputs_drop_pruned_transition(self):
        model = _xor_model()
        t_c = _t_named(model, "c")
        pruned = model.prune_transitions({t_c})
        # p_out_c is orphaned and gone; remaining place_inputs lists exclude t_c.
        for transitions in pruned.place_inputs.values():
            assert t_c not in transitions
