"""ModelDiscoverer._canonicalize_net gives names that depend only on the net's structure."""
import random
import uuid

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from parsing.model_discoverer import ModelDiscoverer


def _build(order_seed: int):
    """source -> A -> p -> (tau_a | tau_b | B) -> sink, built in a shuffled order
    with random uuid names, like pm4py's inductive miner output."""
    rng = random.Random(order_seed)
    net = PetriNet("n")
    places = {n: PetriNet.Place(str(uuid.uuid4())) for n in ("source", "p", "sink")}
    transitions = {
        "A": PetriNet.Transition(str(uuid.uuid4()), "A"),
        "B": PetriNet.Transition(str(uuid.uuid4()), "B"),
        "tau_a": PetriNet.Transition(str(uuid.uuid4()), None),
        "tau_b": PetriNet.Transition(str(uuid.uuid4()), None),
    }
    for obj in rng.sample(list(places.values()), 3):
        net.places.add(obj)
    for obj in rng.sample(list(transitions.values()), 4):
        net.transitions.add(obj)
    arcs = [("source", "A"), ("A", "p"), ("p", "B"), ("p", "tau_a"), ("p", "tau_b"),
            ("B", "sink"), ("tau_a", "sink"), ("tau_b", "sink")]
    rng.shuffle(arcs)
    for src, dst in arcs:
        s = places.get(src) or transitions[src]
        d = places.get(dst) or transitions[dst]
        petri_utils.add_arc_from_to(s, d, net)
    return net, Marking({places["source"]: 1}), Marking({places["sink"]: 1})


def _signature(net):
    return (
        sorted((t.name, t.label, sorted(a.source.name for a in t.in_arcs),
                sorted(a.target.name for a in t.out_arcs)) for t in net.transitions),
        sorted(p.name for p in net.places),
    )


def test_names_do_not_depend_on_construction_order_or_uuids():
    signatures = []
    for seed in range(5):
        net, im, fm = _build(seed)
        ModelDiscoverer._canonicalize_net(net, im, fm)
        signatures.append(_signature(net))
    assert all(s == signatures[0] for s in signatures[1:])


def test_names_follow_the_expected_scheme():
    net, im, fm = _build(0)
    ordered = ModelDiscoverer._canonicalize_net(net, im, fm)
    assert {p.name for p in net.places} == {"p_1", "p_2", "p_3"}
    assert sorted(t.name for t in ordered if t.label is None) == ["tau_1", "tau_2"]
    assert sorted(t.name for t in ordered if t.label) == ["t_1", "t_2"]


def test_initial_and_final_places_get_the_same_name_in_every_build():
    names = set()
    for seed in range(5):
        net, im, fm = _build(seed)
        ModelDiscoverer._canonicalize_net(net, im, fm)
        names.add(([m.name for m in im][0], [m.name for m in fm][0]))
    assert len(names) == 1
