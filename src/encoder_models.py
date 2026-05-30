from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Union

from correlation_miner import RelationshipMetrics
from xes_parser import Parser


@dataclass
class ActionVariant:
    """
    Represents a variant of a base action, created to handle complex preconditions
    (like OR conditions) that PDDL cannot natively express in standard forms.

    Attributes:
        name (str): The unique name for this variant (e.g., "activity_v0").
        base_action (str): The name of the original activity this variant derives from.
        conditions (List[str]): The specific subset of preconditions this variant requires.
        index (int): A numerical identifier for this variant.

    Example:
        ActionVariant(
            name="check_blood_v0",
            base_action="check_blood",
            conditions=["(completed prep_blood)"],
            index=0
        )
    """
    name: str           # unique name of this variant, e.g. "check_blood_v0"
    base_action: str    # name of the original activity this variant derives from, e.g. "check_blood"
    conditions: List[str]  # specific subset of preconditions assigned to this variant, e.g. ["(completed prep_blood)"]
    index: int          # sequential numeric index distinguishing variants of the same base action


@dataclass
class MeasurementVariant:
    """
    Represents a specific outcome of a measurement activity, used to generate
    distinct PDDL actions for each possible observed value.

    Attributes:
        name (str): The specific name for this measurement outcome (e.g., "measure_crp-high").
        base_action (str): The original measurement activity (e.g., "measure_crp").
        attribute (str): The attribute being measured (e.g., "crp").
        outcome_value (str): The observed value for this variant (e.g., "high").

    Example:
        MeasurementVariant(
            name="measure_crp-high",
            base_action="measure_crp",
            attribute="crp",
            outcome_value="high"
        )
    """
    name: str           # unique name of this variant, built as base_action + "-" + outcome_value, e.g. "measure_crp-high"
    base_action: str    # name of the original measurement activity, e.g. "measure_crp"
    attribute: str      # name of the attribute this activity measures, e.g. "crp"
    outcome_value: str  # specific attribute value for which this variant was created, e.g. "high"


@dataclass
class ConditionalEffect:
    """
    Represents an effect that is applied conditionally.

    Attributes:
        condition (str): The PDDL condition that triggers the effect.
        alternatives (List[str]): The effects that occur when the condition is met.

    Example:
        ConditionalEffect(
            condition="(crp high)",
            alternatives=["(enabled treat_high_crp)"]
        )
    """
    condition: str          # PDDL guard condition that triggers the effect, e.g. "(crp high)"
    alternatives: List[str] # list of possible effects applied when the condition holds, e.g. ["(enabled treat_high_crp)"]


@dataclass
class PDDLEncodingContext:
    """
    Holds the shared state and discovered parameters for PDDL generation.
    This replaces the large number of instance attributes in the old Encoder class,
    making it easier to pass context down to specialized builder classes.
    """
    # --- Configuration parameters ---
    parser: Parser              # XES parser instance holding all log data (activities, attributes, relationships, Petri net)
    domain_name: str            # name assigned to the generated PDDL domain, e.g. "process_domain"
    min_confidence: float       # minimum probability threshold for including an attribute-activity relationship as a precondition
    minimal_preconditions: bool # if True, skips predecessor completion requirements (useful for suffix planning where the prefix is already given)
    init: Optional[List[str]] = None  # additional PDDL predicates to insert into the initial state (:init), provided by the user
    goal: Optional[List[str]] = None  # additional PDDL predicates to insert into the goal (:goal), provided by the user

    # --- Properties computed from the parser ---
    activities: Set[str] = field(default_factory=set)
    # set of all activity names discovered in the log (already sanitized for PDDL), e.g. {"check_blood", "administer_drug"}

    attribute_categories: Dict[str, str] = field(default_factory=dict)
    # maps attribute name -> type category: "boolean", "numerical", or "categorical"
    # e.g. {"fever": "boolean", "crp": "numerical", "diagnosis": "categorical"}

    value_mappings: Dict[str, Any] = field(default_factory=dict)
    # maps attribute -> {raw_value -> pddl_value}: translates raw log values into PDDL-compatible names
    # e.g. {"diagnosis": {"Community-acquired Pneumonia": "community_acquired_pneumonia"}}

    # --- Discovered relationships between attributes and activities ---
    attr_activity_relationships: Dict[str, Dict[str, Dict[str, RelationshipMetrics]]] = field(default_factory=dict)
    # structure: {attribute -> {value -> {activity -> {"probability": float, "count": int}}}}
    # captures how likely a given attribute value is observed before a given activity is executed
    # e.g. {"crp": {"high": {"administer_antibiotic": {"probability": 0.85, "count": 34}}}}

    activity_attr_effects: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Union[int, float]]]]]] = field(default_factory=dict)
    # structure: {activity -> {attribute -> {value_before -> {value_after -> {"probability": float, "count": int}}}}}
    # describes how each activity changes attribute values in the log
    # e.g. {"administer_antibiotic": {"crp": {"high": {"low": {"probability": 0.9, "count": 27}}}}}

    tau_activities: Set[str] = field(default_factory=set)
    # subset of `activities` containing silent transitions (names starting with "tau_")
    # these are invisible Petri net transitions with no corresponding real-world event in the log

    # --- Decision point data ---
    decision_points: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # raw data from the parser: maps a key (activity name or complex string representing a set of sources)
    # to the branching probabilities of successor activities: {key -> {successor_activity -> probability}}
    # keys may be complex strings like "{'exec_triage', 'exec_register'}", normalized by _process_decision_points

    decision_preconditions: Dict[str, List[Set[str]]] = field(default_factory=dict)
    # for each activity, a list of precondition sets observed in the parser's decision_samples
    # each set represents a distinct path through which the activity was reached in the log
    # e.g. {"administer_drug": [{"(completed triage)", "(diagnosis sepsis)"}, {"(completed fast_track)"}]}

    decision_attributes: Set[str] = field(default_factory=set)
    # attribute names that appear as conditions inside decision_preconditions
    # used to identify which attributes influence process routing
    # e.g. {"diagnosis", "age_group"}

    processed_decision_points: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # normalized version of `decision_points` with sanitized activity names as keys
    # built by _process_decision_points from the raw parser data
    # e.g. {"triage": {"administer_drug": 0.6, "discharge": 0.4}}

    decision_points_conditions: Dict[Any, Any] = field(default_factory=dict)
    # structure: {decision_point_key -> {attribute -> {value -> {successor_activity -> ...}}}}
    # links each decision point to the attribute-value conditions that determine which branch is enabled
    # used by _process_decision_point_conditions to generate PDDL conditional effects (when ...)

    # --- Process structure ---
    parallel_activities: Dict[str, List[str]] = field(default_factory=dict)
    # raw data from the parser: maps a source activity (typically after an AND-split)
    # to the list of activities that can be executed in parallel after it
    # e.g. {"register": ["blood_test", "x_ray"]}

    parallel_activity_map: Dict[str, Set[str]] = field(default_factory=dict)
    # bidirectional map computed by _build_parallel_activity_map:
    # for each activity, the complete set of activities that can run in parallel with it
    # e.g. {"blood_test": {"x_ray", "register"}, "x_ray": {"blood_test", "register"}}

    # --- Action generation state ---
    split_actions: Dict[str, List[ActionVariant]] = field(default_factory=dict)
    # maps base_action_name -> list of ActionVariants generated to handle OR preconditions
    # populated during domain generation when an action has multiple alternative entry paths
    # e.g. {"check_blood": [ActionVariant("check_blood_v0", ...), ActionVariant("check_blood_v1", ...)]}

    effect_alternatives: Dict[str, List[Union[List[str], ConditionalEffect]]] = field(default_factory=dict)
    # for each action, a list of "alternative groups" representing its non-deterministic or conditional effects
    # each group is either: a list of strings (alternative values for an attribute)
    #   or a list containing a ConditionalEffect (attribute-based conditional effect)
    # e.g. {"measure_crp": [["(crp low)", "(crp high)", "(crp normal)"]],
    #       "triage": [[ConditionalEffect(condition="(diagnosis sepsis)", alternatives=["(enabled icu)"])]]}
