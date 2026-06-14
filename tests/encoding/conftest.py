"""Shared fixtures for encoding tests."""
import pytest
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.obj import Marking

from models import (
    AttributeCatalogEntry, EffectInfo, ParseResult, PetriNetModel,
    TransitionInfo, XorBranchInfo,
)


def _arc(src, tgt):
    return PetriNet.Arc(src, tgt)


@pytest.fixture
def sequence_net():
    """Linear net: p_start -> t_a -> p_mid -> t_b -> p_end.

    No XOR splits, no tau transitions.
    """
    net = PetriNet("seq")

    p_start = PetriNet.Place("p_start")
    p_mid = PetriNet.Place("p_mid")
    p_end = PetriNet.Place("p_end")
    net.places.update([p_start, p_mid, p_end])

    t_a = PetriNet.Transition("t_a", label="Activity A")
    t_b = PetriNet.Transition("t_b", label="Activity B")
    net.transitions.update([t_a, t_b])

    for src, tgt in [(p_start, t_a), (t_a, p_mid), (p_mid, t_b), (t_b, p_end)]:
        net.arcs.add(_arc(src, tgt))

    im = Marking({p_start: 1})
    fm = Marking({p_end: 1})

    model = PetriNetModel(
        petrinet=net,
        initial_marking=im,
        final_marking=fm,
        activities={"activity_a", "activity_b"},
        silent_transitions={},
        trans_inputs={t_a: {p_start}, t_b: {p_mid}},
        trans_outputs={t_a: {p_mid}, t_b: {p_end}},
        xor_splits={},
        place_inputs={p_mid: [t_a], p_end: [t_b]},
    )
    return model


@pytest.fixture
def xor_net():
    """XOR split net: p_start -> t_src -> p_xor ->{t_left, t_right} -> p_end_*.

    p_xor is an XOR split place with two outgoing transitions.
    """
    net = PetriNet("xor")

    p_start = PetriNet.Place("p_start")
    p_xor = PetriNet.Place("p_xor")
    p_end_left = PetriNet.Place("p_end_left")
    p_end_right = PetriNet.Place("p_end_right")
    net.places.update([p_start, p_xor, p_end_left, p_end_right])

    t_src = PetriNet.Transition("t_src", label="Source")
    t_left = PetriNet.Transition("t_left", label="Left Branch")
    t_right = PetriNet.Transition("t_right", label="Right Branch")
    net.transitions.update([t_src, t_left, t_right])

    for src, tgt in [
        (p_start, t_src), (t_src, p_xor),
        (p_xor, t_left), (t_left, p_end_left),
        (p_xor, t_right), (t_right, p_end_right),
    ]:
        net.arcs.add(_arc(src, tgt))

    im = Marking({p_start: 1})
    fm = Marking()

    model = PetriNetModel(
        petrinet=net,
        initial_marking=im,
        final_marking=fm,
        activities={"source", "left_branch", "right_branch"},
        silent_transitions={},
        trans_inputs={t_src: {p_start}, t_left: {p_xor}, t_right: {p_xor}},
        trans_outputs={t_src: {p_xor}, t_left: {p_end_left}, t_right: {p_end_right}},
        xor_splits={p_xor: [t_left, t_right]},
        place_inputs={
            p_xor: [t_src],
            p_end_left: [t_left],
            p_end_right: [t_right],
        },
    )
    return model


@pytest.fixture
def tau_net():
    """Net with a tau transition: p_start -> tau_0 -> p_mid -> t_a -> p_end."""
    net = PetriNet("tau")

    p_start = PetriNet.Place("p_start")
    p_mid = PetriNet.Place("p_mid")
    p_end = PetriNet.Place("p_end")
    net.places.update([p_start, p_mid, p_end])

    t_tau = PetriNet.Transition("tau_0", label=None)
    t_a = PetriNet.Transition("t_a", label="Activity A")
    net.transitions.update([t_tau, t_a])

    for src, tgt in [(p_start, t_tau), (t_tau, p_mid), (p_mid, t_a), (t_a, p_end)]:
        net.arcs.add(_arc(src, tgt))

    im = Marking({p_start: 1})
    fm = Marking({p_end: 1})

    model = PetriNetModel(
        petrinet=net,
        initial_marking=im,
        final_marking=fm,
        activities={"activity_a"},
        silent_transitions={t_tau: "tau_0"},
        trans_inputs={t_tau: {p_start}, t_a: {p_mid}},
        trans_outputs={t_tau: {p_mid}, t_a: {p_end}},
        xor_splits={},
        place_inputs={p_mid: [t_tau], p_end: [t_a]},
    )
    return model


@pytest.fixture
def deterministic_effect():
    """A deterministic effect: always present, single value, no guards."""
    return EffectInfo(
        attribute="diagnosis",
        presence_probability=1.0,
        appearance_level=1,
        appearance_guards=None,
        value_probabilities={"flu": 1.0},
        value_level=1,
        value_guards=None,
    )


@pytest.fixture
def non_deterministic_effect():
    """A non-deterministic effect: multiple values, level 2."""
    return EffectInfo(
        attribute="outcome",
        presence_probability=0.8,
        appearance_level=1,
        appearance_guards=None,
        value_probabilities={"recovered": 0.7, "deceased": 0.3},
        value_level=2,
        value_guards=None,
    )


@pytest.fixture
def boolean_effect():
    """A deterministic boolean effect."""
    return EffectInfo(
        attribute="urgent",
        presence_probability=1.0,
        appearance_level=1,
        appearance_guards=None,
        value_probabilities={True: 1.0},
        value_level=1,
        value_guards=None,
    )


@pytest.fixture
def catalog_categorical():
    """Attribute catalog with one categorical attribute."""
    return {
        "diagnosis": AttributeCatalogEntry(
            attribute_type="categorical",
            possible_values={"flu", "cold", "covid"},
        ),
    }


@pytest.fixture
def catalog_boolean():
    """Attribute catalog with one boolean attribute."""
    return {
        "urgent": AttributeCatalogEntry(
            attribute_type="boolean",
            possible_values=set(),
        ),
    }


@pytest.fixture
def catalog_mixed():
    """Attribute catalog with categorical, numerical, and boolean attributes."""
    return {
        "diagnosis": AttributeCatalogEntry(
            attribute_type="categorical",
            possible_values={"flu", "cold"},
        ),
        "crp": AttributeCatalogEntry(
            attribute_type="numerical",
            possible_values={"lte_10_0", "gte_10_0"},
        ),
        "urgent": AttributeCatalogEntry(
            attribute_type="boolean",
            possible_values=set(),
        ),
    }


@pytest.fixture
def simple_parse_result(sequence_net, deterministic_effect, catalog_categorical):
    """Minimal ParseResult: 2 transitions, 1 deterministic effect on activity_a."""
    return ParseResult(
        petri_net_model=sequence_net,
        place_predecessors={
            "p_mid": ["activity_a"],
            "p_end": ["activity_b"],
        },
        transition_predecessors={
            "activity_a": ["p_start"],
            "activity_b": ["p_mid"],
        },
        transitions={
            "activity_a": TransitionInfo(
                activity_name="activity_a",
                input_places=["p_start"],
                total_firings=100,
                xor_branch=None,
                effects={"diagnosis": deterministic_effect},
            ),
            "activity_b": TransitionInfo(
                activity_name="activity_b",
                input_places=["p_mid"],
                total_firings=80,
                xor_branch=None,
                effects={},
            ),
        },
        start_place="p_start",
        end_place="p_end",
        attribute_catalog=catalog_categorical,
    )


@pytest.fixture
def tau_parse_result(tau_net):
    """ParseResult with a tau transition and no effects."""
    return ParseResult(
        petri_net_model=tau_net,
        place_predecessors={
            "p_mid": ["tau_0"],
            "p_end": ["activity_a"],
        },
        transition_predecessors={
            "activity_a": ["p_mid"],
        },
        transitions={
            "activity_a": TransitionInfo(
                activity_name="activity_a",
                input_places=["p_mid"],
                total_firings=50,
                xor_branch=None,
                effects={},
            ),
        },
        start_place="p_start",
        end_place="p_end",
        attribute_catalog={},
    )
