from dataclasses import dataclass, field, fields as dataclass_fields
from collections import defaultdict
from typing import Any, DefaultDict, Dict, FrozenSet, List, Optional, Set, Tuple

from pm4py import PetriNet, Marking
from pm4py.objects.powl.obj import Transition

import core_utils as utils


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
class TransitionFiringData:
    """Preprocessed data for one firing of a labeled transition.

    Produced by LogPreprocessor.preprocess() during a single pass over the
    PetriNetLog.  Materialises the attribute state that was accumulated before
    the transition fired (pre_state) and the attributes that actually changed
    in this firing (changed_attrs).

    Downstream consumers (ProbabilityEstimator, DecisionMiner) use these
    pre-built snapshots instead of scanning the raw log themselves.

    Attributes:
        activity_name: Sanitized label of the transition that fired.
        pre_state: Accumulated attribute state immediately before this firing.
            Represents the "decision context" — what was known when the
            transition was about to execute.  Keys are sanitized attribute
            names; values are discretized when a Discretizer was provided.
        changed_attrs: Attributes whose value in this step differs from
            pre_state (or that appear for the first time in this execution).
            Keys are sanitized attribute names; values are the new
            (possibly discretized) values.
        from_places: Input places that held a token when the transition fired.
            Used to index XOR-split traversals.
    """
    activity_name: str
    pre_state: Dict[str, Any]
    changed_attrs: Dict[str, Any]
    from_places: FrozenSet[PetriNet.Place]


@dataclass
class PreprocessedLog:
    """Result of a single-pass preprocessing of a PetriNetLog.

    Produced by LogPreprocessor.preprocess().  Both indexes below point to the
    same TransitionFiringData objects — no data is duplicated.

    Attributes:
        transition_firings: Per-transition index.  Maps each sanitized activity
            name to the list of TransitionFiringData for every firing of that
            transition, ordered by appearance in the log.
        xor_firings: Per-XOR-split-place index.  Maps each place name to the
            list of TransitionFiringData for every traversal through that place
            (i.e. every step whose from_places included the XOR-split place and
            whose transition was one of the place's outgoing branches).
    """
    transition_firings: Dict[str, List[TransitionFiringData]]
    xor_firings: Dict[str, List[TransitionFiringData]]


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
class EffectGuards:
    """Decision tree guards for one effect analysis in SOP (sum-of-products) form.

    Produced by DecisionMiner for each (transition, attribute) pair that passes
    the sample and accuracy thresholds.

    subtype is 'appearance' for 2A analysis (does the attribute change?) or
    'value' for 2B analysis (what value does it take?).

    guards maps each outcome to a list of AND-clause lists, analogous to
    XorSplitGuards:
        2A example: {"appears":     [[Guard("risk", "high", False)], ...],
                     "not_appears": [[Guard("risk", "low",  False)], ...]}
        2B example: {"lte_6_0": [[Guard("admitted", None, False)], ...],
                     "gte_6_0": [[Guard("admitted", None, True)],  ...]}
    """
    subtype: str
    guards: Dict[str, List[List["Guard"]]]
    total_samples: int
    dt_accuracy: float


@dataclass
class ConditionalEffect:
    """Conditional effect analysis for one (transition, attribute) pair.

    Produced by DecisionMiner.mine_effects() for each attribute that is a
    plausible effect target of a labeled transition.

    appearance holds the 2A DT result (when does the attribute appear?).
    value holds the 2B DT result (what value does it take?).
    Both may be None; the corresponding status field explains why.

    Status vocabulary (same for appearance_status and value_status):
        "dt"            DT trained, accuracy ok, guards populated.
        "deterministic" Screened out: always present (2A) or single value (2B).
        "never"         Screened out: attribute never changes for this transition.
        "insufficient"  Pre-check failed: too few samples to train.
        "fallback"      DT trained but discarded (low accuracy or pruning).
    """
    attribute: str
    presence_probability: float
    possible_values: List[Any]

    appearance: Optional["EffectGuards"]
    appearance_status: str

    value: Optional["EffectGuards"]
    value_status: str


@dataclass
class TransitionEffects:
    """All conditional effect analyses for one labeled transition.

    Produced by DecisionMiner.mine_effects().

    effects maps each analysed attribute name to its ConditionalEffect.
    Attributes screened out as "never" are absent from the dict.
    """
    transition_name: str
    total_firings: int
    effects: Dict[str, ConditionalEffect]


@dataclass
class AttributeEffect:
    """Attribute change probabilities for a single non-tau transition.

    Produced by ProbabilityEstimator.compute_attribute_effect_probabilities()
    for each labeled transition in the Petri net.
    """
    presence_probabilities: Dict[str, float]
    value_probabilities: Dict[str, Dict[Any, float]]
    total_firings: int


# ---------------------------------------------------------------------------
# Screening dataclasses — decisions made BEFORE DT training
# ---------------------------------------------------------------------------

class ScreeningBase:
    """Mixin for screening dataclasses that need JSON serialization.

    Provides to_dict() which recursively converts the dataclass (and any
    nested ScreeningBase instances or plain dicts) to a JSON-serializable
    structure.  Non-string dict keys are coerced to strings because JSON
    requires string keys (EffectAttrScreening.values uses Dict[Any, ...]).
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert this screening object to a JSON-serializable dict."""
        return _screening_to_dict(self)


def _screening_to_dict(obj: Any) -> Any:
    """Recursively convert a ScreeningBase dataclass to a serializable form."""
    if isinstance(obj, ScreeningBase):
        return {
            f.name: _screening_to_dict(getattr(obj, f.name))
            for f in dataclass_fields(obj)
        }
    if isinstance(obj, dict):
        return {str(k): _screening_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_screening_to_dict(v) for v in obj]
    return obj


@dataclass
class XorBranchScreening(ScreeningBase):
    """Screening result for one branch of an XOR-split place.

    Attributes:
        activity_name: Sanitized name of the branch transition.
        probability: Observed firing probability from XorSplitStats.
        status: "active" (kept for DT training), "pruned" (probability below
            xor_prune_threshold), or "certain" (sole remaining active branch).
    """
    activity_name: str
    probability: float
    status: str


@dataclass
class XorSplitScreening(ScreeningBase):
    """Screening result for one XOR-split place.

    Attributes:
        total_samples: Number of traversals through this XOR place.
        action: "dt" (train decision tree), "deterministic" (single certain
            branch), or "fallback" (insufficient samples or no active branches).
        branches: Per-branch screening results keyed by activity name.
    """
    total_samples: int
    action: str
    branches: Dict[str, XorBranchScreening]


@dataclass
class EffectValueScreening(ScreeningBase):
    """Screening result for one possible value of an attribute effect.

    Attributes:
        probability: Observed value probability from AttributeEffect.
        status: "active" (kept for DT training), "pruned" (probability below
            effect_value_prune_threshold), or "certain" (probability above
            effect_value_certain_threshold).
    """
    probability: float
    status: str


@dataclass
class EffectAttrScreening(ScreeningBase):
    """Screening result for one attribute of a transition effect.

    Attributes:
        presence_probability: How often this attribute changes when the
            transition fires.
        appearance_action: What to do for 2A analysis: "dt", "deterministic",
            "insufficient".
        appearance_samples: Available samples for 2A DT training.
        value_action: What to do for 2B analysis: "dt", "deterministic",
            "insufficient".
        value_samples: Available samples for 2B DT training.
        values: Per-value screening results.
    """
    presence_probability: float
    appearance_action: str
    appearance_samples: int
    value_action: str
    value_samples: int
    values: Dict[Any, "EffectValueScreening"]


@dataclass
class TransitionScreening(ScreeningBase):
    """Screening result for all effect attributes of one labeled transition.

    Attributes:
        transition_name: Sanitized activity name.
        total_firings: Total number of firings in the preprocessed log.
        attributes: Per-attribute screening results.
    """
    transition_name: str
    total_firings: int
    attributes: Dict[str, EffectAttrScreening]


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

    def prune_transitions(
        self, to_remove: Set[PetriNet.Transition]
    ) -> "PetriNetModel":
        """Remove transitions from the Petri net and rebuild all indexes.

        Mutates the underlying PetriNet (transitions, arcs, places) and returns
        a new PetriNetModel with consistent indexes.  The caller should discard
        the old model after calling this method.

        Args:
            to_remove: Transitions to remove.

        Returns:
            New PetriNetModel with updated structure and indexes.
        """
        if not to_remove:
            return self

        for t in to_remove:
            self.petrinet.transitions.discard(t)

        arcs_to_remove = {
            a for a in self.petrinet.arcs
            if a.source in to_remove or a.target in to_remove
        }
        for a in arcs_to_remove:
            self.petrinet.arcs.discard(a)

        connected_places: Set[PetriNet.Place] = set()
        for a in self.petrinet.arcs:
            if isinstance(a.source, PetriNet.Place):
                connected_places.add(a.source)
            if isinstance(a.target, PetriNet.Place):
                connected_places.add(a.target)
        marking_places = set(self.initial_marking) | set(self.final_marking)
        orphaned = self.petrinet.places - connected_places - marking_places
        for p in orphaned:
            self.petrinet.places.discard(p)

        new_trans_inputs = {
            t: places for t, places in self.trans_inputs.items()
            if t not in to_remove
        }
        new_trans_outputs = {
            t: places for t, places in self.trans_outputs.items()
            if t not in to_remove
        }

        new_xor_splits: Dict[PetriNet.Place, List[Transition]] = {}
        for place, transitions in self.xor_splits.items():
            if place in orphaned:
                continue
            remaining = [t for t in transitions if t not in to_remove]
            if len(remaining) > 1:
                new_xor_splits[place] = remaining

        new_place_inputs: Dict[PetriNet.Place, List[Transition]] = {}
        for place, transitions in self.place_inputs.items():
            if place in orphaned:
                continue
            remaining = [t for t in transitions if t not in to_remove]
            if remaining:
                new_place_inputs[place] = remaining

        new_activities = {
            utils.sanitize_name(t.label)
            for t in self.petrinet.transitions if t.label
        }
        new_silent = {
            t: name for t, name in self.silent_transitions.items()
            if t not in to_remove
        }

        return PetriNetModel(
            petrinet=self.petrinet,
            initial_marking=self.initial_marking,
            final_marking=self.final_marking,
            activities=new_activities,
            silent_transitions=new_silent,
            trans_inputs=new_trans_inputs,
            trans_outputs=new_trans_outputs,
            xor_splits=new_xor_splits,
            place_inputs=new_place_inputs,
        )


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

    # --- Statistical fallback level threshold (shared for XOR splits and effects) ---
    # Minimum number of samples required to trust observed probabilities (level 2).
    # Below this threshold the system falls back to level 3: equal weights for XOR
    # branches, or effect removal for conditional effects.
    probability_min_samples: int = 10

    # --- DT leaf pruning ---
    # Leaves with fewer samples or lower purity than these thresholds are removed
    # from the SOP guards after training.
    # dt_prune_orphan_mode controls what happens when pruning would remove ALL leaves
    # for an activity:
    #   "keep_best" — keep the single highest-purity leaf (tie-break: most samples)
    #   "drop"      — remove the activity from guards entirely (WARNING logged)
    dt_prune_min_leaf_samples: int = 5
    dt_prune_min_purity: float = 0.70
    dt_prune_orphan_mode: str = "keep_best"

    # --- XOR split statistical fallback (Level 2) ---
    # majority_only | weighted | pruned_weighted
    xor_statistical_mode: str = "pruned_weighted"
    xor_prune_threshold: float = 0.10

    # --- Conditional effect screening ---
    # Attributes with presence_probability below never_threshold are not effects.
    # Attributes above always_threshold are always present: skip 2A, only run 2B.
    effect_never_threshold: float = 0.05
    effect_always_threshold: float = 0.95

    # --- Effect value screening ---
    # Values with probability below prune_threshold are excluded from DT training.
    # Values above certain_threshold are treated as always-this-value.
    effect_value_prune_threshold: float = 0.10
    effect_value_certain_threshold: float = 0.95

    # --- Effect co-occurrence classification ---
    # Pairs with joint probability >= related_effect_prob are classified as
    # "related" (always appear together in the same firing).
    # Pairs with joint probability <= incompatible_effect_prob are classified as
    # "incompatible" (never appear together in the same firing).
    # Pairs in between carry no structural constraint.
    related_effect_prob: float = 0.9
    incompatible_effect_prob: float = 0.05

    # --- Conditional effect statistical fallback (Level 2) ---
    # appearance (2A): threshold | always | duplicate | duplicate_no_cost
    # "duplicate_no_cost" creates variant actions without probability cost tracking.
    effect_appearance_mode: str = "duplicate"
    effect_appearance_threshold: float = 0.20
    # value (2B): majority_only | duplicate | duplicate_no_cost
    # "duplicate_no_cost" creates value variants without probability cost tracking.
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

    # --- Attribute precondition mining ---
    # Minimum frequency for a (attribute, value) pair in pre_state to be
    # recognised as a structural precondition of a transition.
    # Only firings where the attribute is already present in pre_state are counted.
    attr_precondition_min_frequency: float = 0.95
    # Minimum number of firings where the attribute is present before attempting
    # the analysis for that (transition, attribute) pair.
    attr_precondition_min_firings: int = 10

    # --- Logging ---
    log_removed_effects: bool = True

    # --- Snapshot output ---
    # Directory where pre-pruning snapshots (PNML + JSON) are written.
    snapshot_dir: str = "output/snapshots"


# ---------------------------------------------------------------------------
# Encoder-ready output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ActionDurationStats:
    """Duration statistics for a single action, suitable for PDDL duration constraints.

    For log-derived sources ('lifecycle', 'inter_event') effective_min and
    effective_max are:
        effective_min = max(observed_min, mean - std_dev)
        effective_max = min(observed_max, mean + std_dev)

    For externally supplied data ('external') effective_min/max are set
    directly from user-provided bounds; all statistical fields remain None.

    Attributes:
        effective_min: Lower bound for the PDDL duration inequality (seconds).
        effective_max: Upper bound for the PDDL duration inequality (seconds).
        source: Origin — 'lifecycle', 'inter_event', or 'external'.
        mean: Mean duration in seconds (log-derived only).
        std_dev: Standard deviation in seconds (log-derived only).
        observed_min: Smallest raw duration observed in the log (log-derived only).
        observed_max: Largest raw duration observed in the log (log-derived only).
        count: Number of observations used to compute the statistics (log-derived only).
    """
    effective_min: float
    effective_max: float
    source: str
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    observed_min: Optional[float] = None
    observed_max: Optional[float] = None
    count: Optional[int] = None


@dataclass
class AttributeCatalogEntry:
    """Type and active value domain for one attribute in the final ParseResult.

    Only attributes referenced by at least one EffectInfo or XOR guard survive
    into this catalog — attributes filtered out during the cascade are absent.

    For discretized numerical attributes possible_values contains interval
    labels (e.g. 'lte_10_0', 'gte_10_0_lte_97_5') because that is the
    representation the encoder operates on.  Boolean attributes carry an
    empty set: their guards use the negated flag rather than explicit values.

    Attributes:
        attribute_type: One of 'boolean', 'numerical', 'categorical'.
        possible_values: Set of values that appear in effects or guard conditions.
    """
    attribute_type: str
    possible_values: Set[Any]


@dataclass
class XorBranchInfo:
    """XOR routing information for a single branch transition.

    Attached to TransitionInfo when the transition is one of the branches of
    an XOR split.  All fields concern this specific branch only; the cascade
    level and sample count refer to the split point as a whole.

    Attributes:
        probability: Observed firing probability of this branch in the log.
        total_samples: Total traversal count at the XOR split place.
        cascade_level: 1 = DT guards available, 2 = statistical fallback,
            3 = no data (all branches kept with equal cost).
        guards: DT conditions for THIS branch in SOP form (List[List[Guard]]).
            None when cascade_level > 1.
    """
    probability: float
    total_samples: int
    cascade_level: int
    guards: Optional[List[List[Guard]]]


@dataclass
class EffectInfo:
    """Encoder-ready conditional effect info for one (transition, attribute) pair.

    Produced by Parser after applying the full cascade (DT → statistical → default).
    Both appearance and value sides carry a cascade level and optional DT guards.

    Attributes:
        attribute: Sanitized attribute name.
        presence_probability: How often this attribute changes when the transition fires.
        appearance_level: 1=DT, 2=statistical fallback, 3=effect removed.
        appearance_guards: DT guards for appearance (2A); None if level > 1.
        value_probabilities: Observed probability of each active value (pruned values excluded).
        value_level: 1=DT, 2=statistical fallback, 3=effect removed.
        value_guards: DT guards for value (2B); None if level > 1.
    """
    attribute: str
    presence_probability: float
    appearance_level: int
    appearance_guards: Optional[EffectGuards]
    value_probabilities: Dict[Any, float]
    value_level: int
    value_guards: Optional[EffectGuards]


@dataclass
class TransitionInfo:
    """Encoder-ready summary of one labeled transition.

    Aggregates structural (input places), statistical (firings), XOR routing,
    and conditional effect information into a single encoder-facing object.

    Attributes:
        activity_name: Sanitized activity name.
        input_places: Names of the Petri net places that must hold a token to
            enable this transition (direct structural predecessors).
        total_firings: Number of times this transition fired in the log.
        xor_branch: XOR routing info if this transition is a branch of an XOR
            split; None otherwise.
        effects: Encoder-ready conditional effects keyed by attribute name.
            Attributes screened as "never" and effects at cascade level 3 are
            excluded.
    """
    activity_name: str
    input_places: List[str]
    total_firings: int
    xor_branch: Optional[XorBranchInfo]
    effects: Dict[str, EffectInfo]
    duration: Optional[ActionDurationStats] = None
    related_effects: Set[FrozenSet[str]] = field(default_factory=set)
    incompatible_effects: Set[FrozenSet[str]] = field(default_factory=set)
    attribute_preconditions: List[Guard] = field(default_factory=list)


@dataclass
class ParseResult:
    """Complete encoder-ready output of the XES parsing pipeline.

    Produced by Parser after Petri net discovery, log preprocessing, decision
    mining, and post-pruning cleanup.  All structures reflect the pruned net —
    removed branches and effects are absent.

    Attributes:
        petri_net_model: Petri net model after pruning (consistent indexes).
        place_predecessors: For each place, the transitions whose output arcs
            lead into it (Trans→Place arcs).  Derived from place_inputs.
        transition_predecessors: For each transition, the places that must hold
            a token to enable it (Place→Trans arcs).  Derived from trans_inputs.
        transitions: Encoder-ready info for every labeled transition.
        start_place: Name of the initial marking place.
        end_place: Name of the final marking place.
        attribute_catalog: Attributes that survive into the final output, with
            their type and the set of values referenced by effects or guards.
            Attributes filtered out during the cascade are absent.
        negated_attributes: Attributes that appear with negated=True in at least
            one guard (XOR split or effect).  The encoder generates negative
            predicates (_is_not / _false) only for these attributes.
    """
    petri_net_model: PetriNetModel
    place_predecessors: Dict[str, List[str]]
    transition_predecessors: Dict[str, List[str]]
    transitions: Dict[str, TransitionInfo]
    start_place: str
    end_place: str
    attribute_catalog: Dict[str, AttributeCatalogEntry]
    negated_attributes: Set[str] = field(default_factory=set)
    xor_virtual_taus: Set[str] = field(default_factory=set)
    artificial_xor_places: Set[str] = field(default_factory=set)
    variant_to_art_place: Dict[str, str] = field(default_factory=dict)
    artificial_xor_data: Dict[str, Any] = field(default_factory=dict)
