"""Tests for encoding/prepared_input.py: dataclasses, (de)serialization, and
the PreparedDomainInput.build_graph_from_petri_net classmethod."""
import json

from encoding.prepared_input import (
    GraphEdge,
    GraphNode,
    PreparedDomainInput,
    PreparedEffectGroup,
    PreparedTransition,
    PreparedXorBranch,
)
from models import AttributeCatalogEntry, Guard


class TestBuildGraphFromPetriNet:
    def test_sequence_net_has_no_xor_or_and_nodes(self, sequence_net):
        nodes, edges = PreparedDomainInput.build_graph_from_petri_net(sequence_net)

        node_types = {n.type for n in nodes}
        assert node_types == {"place", "transition"}
        assert len(nodes) == 5  # 3 places + 2 transitions
        assert len(edges) == 4

    def test_xor_net_marks_split_place(self, xor_net):
        nodes, edges = PreparedDomainInput.build_graph_from_petri_net(xor_net)

        by_id = {n.id: n for n in nodes}
        assert by_id["p_xor"].type == "xor_split"
        assert by_id["p_start"].type == "place"
        assert by_id["t_left"].type == "transition"
        assert by_id["t_left"].label == "Left Branch"  # raw label, not sanitized

        edge_pairs = {(e.source, e.target) for e in edges}
        assert ("p_xor", "t_left") in edge_pairs
        assert ("p_xor", "t_right") in edge_pairs

    def test_tau_net_marks_silent_node_with_its_synthetic_label(self, tau_net):
        nodes, edges = PreparedDomainInput.build_graph_from_petri_net(tau_net)

        by_id = {n.id: n for n in nodes}
        assert by_id["tau_0"].type == "silent"
        assert by_id["tau_0"].label == "tau_0"

    def test_and_split_transition_is_flagged(self):
        from pm4py.objects.petri_net.obj import Marking, PetriNet

        net = PetriNet("and")
        p_in = PetriNet.Place("p_in")
        p_out_a = PetriNet.Place("p_out_a")
        p_out_b = PetriNet.Place("p_out_b")
        net.places.update([p_in, p_out_a, p_out_b])

        t_and = PetriNet.Transition("t_and", label="Fork")
        net.transitions.add(t_and)
        net.arcs.add(PetriNet.Arc(p_in, t_and))
        net.arcs.add(PetriNet.Arc(t_and, p_out_a))
        net.arcs.add(PetriNet.Arc(t_and, p_out_b))

        from models import PetriNetModel

        model = PetriNetModel(
            petrinet=net,
            initial_marking=Marking({p_in: 1}),
            final_marking=Marking(),
            activities={"fork"},
            silent_transitions={},
            trans_inputs={t_and: {p_in}},
            trans_outputs={t_and: {p_out_a, p_out_b}},
            xor_splits={},
            place_inputs={p_out_a: [t_and], p_out_b: [t_and]},
        )

        nodes, _ = PreparedDomainInput.build_graph_from_petri_net(model)
        by_id = {n.id: n for n in nodes}
        assert by_id["t_and"].type == "and_split"


class TestSerializationRoundTrip:
    def test_guard_predicate_and_negation_round_trip(self):
        guard = Guard(attribute="risk", value="high", negated=True)
        group = PreparedEffectGroup(assignments=[("risk", "high")], guard=[[guard]], probability=0.5)

        data = group.to_dict()
        restored = PreparedEffectGroup.from_dict(data)

        assert restored == group

    def test_boolean_guard_value_becomes_none_after_round_trip(self):
        guard = Guard(attribute="admitted", value=None, negated=False)
        data = guard and PreparedXorBranch(
            activity_name="a", conditions=[[guard]], probability=1.0
        ).to_dict()
        restored = PreparedXorBranch.from_dict(data)

        assert restored.conditions == [[guard]]

    def test_prepared_domain_input_round_trip_preserves_build_relevant_fields(self):
        nodes = [
            GraphNode(id="p_start", type="place", label="p_start"),
            GraphNode(id="t_a", type="transition", label="Activity A"),
        ]
        edges = [GraphEdge(source="p_start", target="t_a")]
        transitions = {
            "activity_a": PreparedTransition(
                activity_name="activity_a",
                input_places=["p_start"],
                preconditions=[[Guard(attribute="risk", value="high", negated=False)]],
                effect_groups=[
                    PreparedEffectGroup(
                        assignments=[("diagnosis", "flu")], guard=[], probability=0.6
                    )
                ],
                cost=0.0,
                duration=None,
            )
        }
        xor_branches = {
            "p_xor": [
                PreparedXorBranch(activity_name="t_left", conditions=[], probability=0.5)
            ]
        }
        attribute_catalog = {
            "risk": AttributeCatalogEntry(attribute_type="categorical", possible_values={"high", "low"})
        }

        original = PreparedDomainInput(
            nodes=nodes,
            edges=edges,
            transitions=transitions,
            xor_branches=xor_branches,
            attribute_catalog=attribute_catalog,
        )

        data = original.to_dict()
        json.dumps(data)  # must be JSON-serializable
        restored = PreparedDomainInput.from_dict(data)

        assert restored.nodes == original.nodes
        assert restored.edges == original.edges
        assert restored.transitions == original.transitions
        assert restored.xor_branches == original.xor_branches
        assert restored.attribute_catalog.keys() == original.attribute_catalog.keys()
        assert (
            restored.attribute_catalog["risk"].attribute_type
            == original.attribute_catalog["risk"].attribute_type
        )
        assert (
            set(restored.attribute_catalog["risk"].possible_values)
            == original.attribute_catalog["risk"].possible_values
        )
