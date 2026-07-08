"""PreparedDomainInput: the meeting-point object between the automatic pipeline
and the frontend GUI.

Both data sources (the XES pipeline's ParseResult and the web GUI's
current.json) converge into this same set of dataclasses before reaching the
shared build engine in encoding/domain_builder.py. Serialization and
deserialization live next to the class they (de)serialize, as two methods of
PreparedDomainInput, since they are pure inverses of each other operating on
the same object.

No sanitization happens anywhere in this module: names and values are carried
exactly as they arrive. The only place that sanitizes is
encoding/pddl_model.py, in its rendering methods.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ActionDurationStats,
    AttributeCatalogEntry,
    Guard,
    PetriNetModel,
)


def _guard_to_dict(guard: Guard) -> Dict[str, Any]:
    predicate = "<>" if guard.negated else "="
    value = guard.value if guard.value is not None else "true"
    return {"attribute": guard.attribute, "predicate": predicate, "value": value}


def _guard_from_dict(data: Dict[str, Any]) -> Guard:
    negated = data.get("predicate") == "<>"
    value = data.get("value")
    if value == "true":
        value = None
    return Guard(attribute=data["attribute"], value=value, negated=negated)


def _sop_to_dict(sop: List[List[Guard]]) -> List[List[Dict[str, Any]]]:
    return [[_guard_to_dict(g) for g in clause] for clause in sop]


def _sop_from_dict(data: List[List[Dict[str, Any]]]) -> List[List[Guard]]:
    return [[_guard_from_dict(g) for g in clause] for clause in data]


@dataclass(frozen=True)
class PreparedEffectGroup:
    assignments: List[Tuple[str, str]]
    guard: List[List[Guard]]
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": [
                {"attribute": attr, "value": value} for attr, value in self.assignments
            ],
            "guard": _sop_to_dict(self.guard),
            "probability": self.probability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreparedEffectGroup":
        return cls(
            assignments=[
                (entry["attribute"], entry["value"]) for entry in data.get("assignments", [])
            ],
            guard=_sop_from_dict(data.get("guard", [])),
            probability=data.get("probability", 1.0),
        )


@dataclass(frozen=True)
class VariantInfo:
    """Metadata for one PDDL action variant, keyed by its final variant name.

    Produced as a byproduct of the axis A x B x C Cartesian product in
    transition_action_builder._build_prepared_transition_actions, so that a
    plan replayer can look up "which effect group did variant execute_x_v2
    apply" directly by name instead of regenerating the Cartesian product at
    runtime (see docs/trace_replayer_analysis.md §6.3).
    """
    activity_name: str
    preconditions: List[List[Guard]]
    effect_group: Optional[PreparedEffectGroup]


@dataclass(frozen=True)
class PreparedXorBranch:
    activity_name: str
    conditions: Optional[List[List[Guard]]]
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity_name": self.activity_name,
            "conditions": _sop_to_dict(self.conditions),
            "probability": self.probability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreparedXorBranch":
        return cls(
            activity_name=data["activity_name"],
            conditions=_sop_from_dict(data.get("conditions", [])),
            probability=data.get("probability", 1.0),
        )


@dataclass(frozen=True)
class PreparedTransition:
    activity_name: str
    input_places: List[str]
    preconditions: List[List[Guard]] # confirmed preconditions directly for the transition (not due to effects or xors)
    effect_groups: List[PreparedEffectGroup]
    cost: float = 0.0
    duration: Optional[ActionDurationStats] = None

    def to_dict(self) -> Dict[str, Any]:
        duration = None
        if self.duration is not None:
            duration = {
                "effective_min": self.duration.effective_min,
                "effective_max": self.duration.effective_max,
                "source": self.duration.source,
                "mean": self.duration.mean,
                "std_dev": self.duration.std_dev,
                "observed_min": self.duration.observed_min,
                "observed_max": self.duration.observed_max,
                "count": self.duration.count,
            }
        return {
            "activity_name": self.activity_name,
            "input_places": list(self.input_places),
            "preconditions": _sop_to_dict(self.preconditions),
            "effect_groups": [group.to_dict() for group in self.effect_groups],
            "cost": self.cost,
            "duration": duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreparedTransition":
        duration = None
        raw_duration = data.get("duration")
        if raw_duration:
            duration = ActionDurationStats(
                effective_min=raw_duration["effective_min"],
                effective_max=raw_duration["effective_max"],
                source=raw_duration.get("source", "external"),
                mean=raw_duration.get("mean"),
                std_dev=raw_duration.get("std_dev"),
                observed_min=raw_duration.get("observed_min"),
                observed_max=raw_duration.get("observed_max"),
                count=raw_duration.get("count"),
            )
        return cls(
            activity_name=data["activity_name"],
            input_places=list(data.get("input_places", [])),
            preconditions=_sop_from_dict(data.get("preconditions", [])),
            effect_groups=[
                PreparedEffectGroup.from_dict(group) for group in data.get("effect_groups", [])
            ],
            cost=data.get("cost", 0.0),
            duration=duration,
        )


# GraphNode.type values that represent a Petri net place vs a transition.
# Shared by any code that needs to classify nodes without re-deriving these
# sets (e.g. graph indexing, PDDL types/constants construction).
PLACE_NODE_TYPES = frozenset({"place", "xor_split"})
TRANS_NODE_TYPES = frozenset({"transition", "and_split", "silent"})


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str  # "place" | "xor_split" | "transition" | "and_split" | "silent"
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type, "label": self.label}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        return cls(id=data["id"], type=data["type"], label=data.get("label", ""))


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        return cls(source=data["source"], target=data["target"])


@dataclass
class PreparedDomainInput:
    transitions: Dict[str, PreparedTransition]
    xor_branches: Dict[str, List[PreparedXorBranch]]  # place_name -> branches
    attribute_catalog: Dict[str, AttributeCatalogEntry]
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)

    @classmethod
    def build_graph_from_petri_net(
        cls, pnm: PetriNetModel
    ) -> Tuple[List["GraphNode"], List["GraphEdge"]]:
        """Build the GraphNode/GraphEdge lists directly from a discovered Petri net.

        Mirrors the node classification already done by
        web/serializer.py::_build_graph today (place vs xor_split, transition
        vs and_split vs silent), but never sanitizes: labels are carried
        exactly as pm4py produced them, since sanitization belongs only to
        encoding/pddl_model.py.

        Args:
            pnm: Structural Petri net model, as produced by ModelDiscoverer
                and carried on ParseResult.petri_net_model.

        Returns:
            (nodes, edges) ready to assign to PreparedDomainInput.nodes/.edges.
        """
        xor_place_names = {place.name for place in pnm.xor_splits}

        and_transition_labels = set()
        for t in pnm.petrinet.transitions:
            if not t.label:
                continue
            inputs = pnm.trans_inputs.get(t, set())
            outputs = pnm.trans_outputs.get(t, set())
            if len(inputs) > 1 or len(outputs) > 1:
                and_transition_labels.add(t.label)

        nodes: List[GraphNode] = []

        for place in pnm.petrinet.places:
            node_type = "xor_split" if place.name in xor_place_names else "place"
            nodes.append(GraphNode(id=place.name, type=node_type, label=place.name))

        for t in pnm.petrinet.transitions:
            t_id = t.name if t.name else str(id(t))
            is_silent = t.label is None

            if is_silent:
                node_type = "silent"
                label = pnm.silent_transitions.get(t, "")
            elif t.label in and_transition_labels:
                node_type = "and_split"
                label = t.label
            else:
                node_type = "transition"
                label = t.label

            nodes.append(GraphNode(id=t_id, type=node_type, label=label or ""))

        edges: List[GraphEdge] = []
        for arc in pnm.petrinet.arcs:
            source, target = arc.source, arc.target
            source_id = source.name if hasattr(source, "name") else str(id(source))
            target_id = target.name if hasattr(target, "name") else str(id(target))
            edges.append(GraphEdge(source=source_id, target=target_id))
        return nodes, edges

    def to_dict(self) -> Dict[str, Any]:
        """Serialize into a dict with the same shape as today's current.json."""
        return {
            "graph": {
                "nodes": [node.to_dict() for node in self.nodes],
                "edges": [edge.to_dict() for edge in self.edges],
            },
            "transitions": {
                activity_name: transition.to_dict()
                for activity_name, transition in self.transitions.items()
            },
            "xor_splits": {
                place_name: {
                    "branches": {
                        branch.activity_name: branch.to_dict() for branch in branches
                    }
                }
                for place_name, branches in self.xor_branches.items()
            },
            "attribute_catalog": {
                attribute: {
                    "type": entry.attribute_type,
                    "possible_values": sorted(str(v) for v in entry.possible_values),
                    "bin_boundaries": entry.bin_boundaries or [],
                }
                for attribute, entry in self.attribute_catalog.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreparedDomainInput":
        """Deserialize from current.json (or any dict with the same shape)."""
        graph = data.get("graph", {})
        nodes = [GraphNode.from_dict(node) for node in graph.get("nodes", [])]
        edges = [GraphEdge.from_dict(edge) for edge in graph.get("edges", [])]

        transitions = {
            activity_name: PreparedTransition.from_dict(transition)
            for activity_name, transition in data.get("transitions", {}).items()
        }

        xor_branches: Dict[str, List[PreparedXorBranch]] = {}
        for place_name, split in data.get("xor_splits", {}).items():
            branches = split.get("branches", {})
            xor_branches[place_name] = [
                PreparedXorBranch.from_dict(branch) for branch in branches.values()
            ]

        attribute_catalog = {
            attribute: AttributeCatalogEntry(
                attribute_type=entry["type"],
                possible_values=set(entry.get("possible_values", [])),
                bin_boundaries=entry.get("bin_boundaries") or None,
            )
            for attribute, entry in data.get("attribute_catalog", {}).items()
        }

        return cls(
            nodes=nodes,
            edges=edges,
            transitions=transitions,
            xor_branches=xor_branches,
            attribute_catalog=attribute_catalog,
        )
