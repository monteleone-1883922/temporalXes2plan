"""Shared fixtures for tests.replay.

Unlike tests/encoding/conftest.py's nets (built only for PreparedDomainInput
consumption, never fed to real pm4py replay), the nets here are wired with
petri_utils.add_arc_from_to so that in_arcs/out_arcs are populated too — the
replay module's Phase 1 (replay._log_alignment) calls real pm4py conformance
checking functions, which traverse the net structure directly and silently
see an empty net otherwise.
"""
import pytest
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

from models import PetriNetModel


def _wire(net: PetriNet, *edges):
    for src, tgt in edges:
        add_arc_from_to(src, tgt, net)


@pytest.fixture
def tau_seq_net():
    """p_in -> t_a("A") -> p_mid1 -> t_tau(None) -> p_mid2 -> t_b("B") -> p_out."""
    t_a = PetriNet.Transition("t_a", "A")
    t_tau = PetriNet.Transition("t_tau", None)
    t_b = PetriNet.Transition("t_b", "B")
    p_in = PetriNet.Place("p_in")
    p_mid1 = PetriNet.Place("p_mid1")
    p_mid2 = PetriNet.Place("p_mid2")
    p_out = PetriNet.Place("p_out")

    net = PetriNet("tau_seq")
    net.places.update([p_in, p_mid1, p_mid2, p_out])
    net.transitions.update([t_a, t_tau, t_b])
    _wire(
        net,
        (p_in, t_a), (t_a, p_mid1),
        (p_mid1, t_tau), (t_tau, p_mid2),
        (p_mid2, t_b), (t_b, p_out),
    )

    trans_inputs = {t_a: {p_in}, t_tau: {p_mid1}, t_b: {p_mid2}}
    trans_outputs = {t_a: {p_mid1}, t_tau: {p_mid2}, t_b: {p_out}}

    model = PetriNetModel(
        petrinet=net,
        initial_marking=Marking({p_in: 1}),
        final_marking=Marking({p_out: 1}),
        activities={"A", "B"},
        silent_transitions={t_tau: "tau_1"},
        trans_inputs=trans_inputs,
        trans_outputs=trans_outputs,
        xor_splits={},
        place_inputs={p_mid1: [t_a], p_mid2: [t_tau], p_out: [t_b]},
    )
    return model
