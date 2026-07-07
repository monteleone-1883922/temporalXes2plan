"""Shared fixtures for encoding tests.

Only structural PetriNetModel builders live here — every test exercising the
new PreparedDomainInput-based engine hand-builds its own
PreparedDomainInput/PreparedTransition fixtures inline (see
test_build_domain_from_prepared_info.py / test_build_prepared_domain_input.py),
so no ParseResult/TransitionInfo fixtures are needed at this shared level.
"""
import pytest
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.obj import Marking

from models import PetriNetModel


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
