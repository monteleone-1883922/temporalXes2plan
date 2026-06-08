from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set

from pm4py import PetriNet, Marking
from pm4py.objects.powl.obj import Transition


@dataclass
class FiringStep:
    """A single transition firing in a Petri net execution, annotated with event attributes.

    Produced by PetriNetLogBuilder for each activated transition during replay.
    from_places contains only input places that actually held a token when the
    transition fired (runtime state). Output places are static structure and can
    be looked up via PetriNetModel.trans_outputs.
    """
    transition: PetriNet.Transition
    activity_name: str
    is_tau: bool
    from_places: Set[PetriNet.Place]
    attributes: Dict[str, Any]
    duration_seconds: Optional[float] = None


@dataclass
class TraceExecution:
    """A complete Petri net execution for one log trace.

    steps is an ordered list of FiringSteps, one per activated transition
    (including tau transitions). Navigation helpers allow querying by place
    or by transition without iterating steps manually.
    """
    trace_id: str
    steps: List[FiringStep]

    def steps_through_place(self, place: PetriNet.Place) -> List[FiringStep]:
        """All steps that consumed a token from the given place."""
        return [s for s in self.steps if place in s.from_places]

    def steps_for_transition(self, transition: PetriNet.Transition) -> List[FiringStep]:
        """All steps where the given transition fired."""
        return [s for s in self.steps if s.transition is transition]


@dataclass
class PetriNetLog:
    """Event log expressed as Petri net executions with annotated attributes.

    Replaces the raw pm4py EventLog for all downstream analysis. Each execution
    corresponds to one log trace that met the replay fitness threshold.
    """
    executions: List[TraceExecution]
    net: PetriNet
    initial_marking: Marking
    final_marking: Marking


@dataclass
class XorSplitStats:
    """Branch probabilities and sample size for a single XOR-split decision point.

    Produced by ProbabilityEstimator.compute_decision_points_probabilities() for
    each XOR-split place in the Petri net.
    """
    probabilities: Dict[str, float]
    total_executions: int


@dataclass
class Guard:
    """An atomic condition extracted from a decision tree split.

    Produced by DecisionMiner._format_condition() and collected into SOP form
    by _build_sop_guards(). Intentionally format-agnostic: conversion to PDDL
    or any other target language is the responsibility of downstream encoders.

    Attributes:
        attribute: Sanitized attribute name, e.g. "risk", "admitted".
        value: Category value for one-hot conditions, e.g. "high".
            None for boolean conditions where only presence/absence matters.
        negated: True when the condition is asserted as false,
            e.g. admitted=False or risk != high.
    """
    attribute: str
    value: Optional[str]
    negated: bool = False


@dataclass
class XorSplitGuards:
    """Decision tree guards for one XOR-split in SOP (sum-of-products) form.

    Produced by DecisionMiner.mine_xor_splits() for each split that passes
    dt_min_samples and dt_min_accuracy thresholds.

    guards maps each branch activity to a list of AND-clause lists:
        outer list = OR  (multiple DT paths reaching this branch)
        inner list = AND (Guard conditions along one DT path)
    Example:
        {"approve": [[Guard("amount", "lte_500"), Guard("priority", "gte_3")],
                     [Guard("risk", "low")]]}
    """
    guards: Dict[str, List[List["Guard"]]]
    total_samples: int
    dt_accuracy: float


@dataclass
class AttributeEffect:
    """Attribute change probabilities for a single non-tau transition.

    Produced by ProbabilityEstimator.compute_attribute_effect_probabilities()
    for each labeled transition in the Petri net.
    """
    presence_probabilities: Dict[str, float]
    value_probabilities: Dict[str, Dict[Any, float]]
    total_firings: int


@dataclass
class PetriNetModel:
    """Complete structural description of a discovered Petri net.

    Produced by ModelDiscoverer.discover() and used as the authoritative
    structural representation throughout the parsing pipeline.

    trans_inputs / trans_outputs are the arc maps ready to be passed directly
    to PetriNetLogBuilder without a separate build_arc_maps() call.

    place_inputs is the inverse of trans_outputs: for each place, the transitions
    that produce tokens into it (Trans→Place arcs).
    xor_splits is the inverse of trans_inputs restricted to places with >1 outgoing
    transition.
    """
    petrinet: PetriNet
    initial_marking: Marking
    final_marking: Marking
    activities: Set[str]
    silent_transitions: Dict[Transition, str]
    trans_inputs: Dict[Transition, Set[PetriNet.Place]]
    trans_outputs: Dict[Transition, Set[PetriNet.Place]]
    xor_splits: Dict[PetriNet.Place, List[Transition]]
    place_inputs: Dict[PetriNet.Place, List[Transition]]


@dataclass
class AnalysisConfig:
    """Global configuration for the XES log analysis pipeline.

    Controls pre-discretization (k-means++), decision tree cascade thresholds,
    and statistical fallback modes for XOR splits and conditional effects.
    """

    # --- Pre-discretization (k-means++) ---
    kmeans_max_k: int = 5
    kmeans_min_cluster_fraction: float = 0.05
    kmeans_silhouette_threshold: float = 0.10
    kmeans_n_init: int = 10

    # --- Decision tree cascade (shared for XOR splits and conditional effects) ---
    dt_min_samples: int = 30
    dt_min_accuracy: float = 0.75
    dt_max_depth: int = 3

    # --- XOR split statistical fallback (Level 2) ---
    # majority_only | weighted | pruned_weighted
    xor_statistical_mode: str = "pruned_weighted"
    xor_prune_threshold: float = 0.10

    # --- Conditional effect statistical fallback (Level 2) ---
    # appearance (2A): threshold | always | duplicate
    effect_appearance_mode: str = "duplicate"
    effect_appearance_threshold: float = 0.20
    # value (2B): majority_only | duplicate
    effect_value_mode: str = "duplicate"

    # --- Token replay (XOR split probability estimation) ---
    # Minimum trace_fitness for a replayed trace to be included in branch counting.
    # Traces below this threshold are skipped and logged.
    replay_min_fitness: float = 0.8

    # --- Attribute filtering ---
    # Raw attribute names (as they appear in the XES log) to exclude from all
    # mining steps (CorrelationMiner, ProbabilityEstimator effect analysis).
    ignored_attributes: Set[str] = field(default_factory=lambda: {
        'case:concept:name', 'concept:name', 'time:timestamp',
        'lifecycle:transition', 'org:resource', 'org:group',
        'variant-index', 'Resource', 'org:role',
    })

    # --- Logging ---
    log_removed_effects: bool = True
