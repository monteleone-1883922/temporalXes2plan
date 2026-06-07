from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from pm4py import PetriNet, Marking
from pm4py.objects.powl.obj import Transition


@dataclass
class FiringStep:
    """A single transition firing in a Petri net execution, annotated with event attributes.

    Produced by PetriNetLogBuilder for each activated transition during replay.
    from_places contains only input places that actually held a token when the
    transition fired (runtime state). Output places are static structure and can
    be looked up via StructureAnalyzer.build_arc_maps() trans_outputs.
    """
    transition: PetriNet.Transition
    activity_name: str
    is_tau: bool
    from_places: Set[PetriNet.Place]
    attributes: Dict[str, Any]


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
    """
    petrinet: PetriNet
    initial_marking: Marking
    final_marking: Marking
    activities: Set[str]
    silent_transitions: Dict[Transition, str]


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

    # --- Logging ---
    log_removed_effects: bool = True
