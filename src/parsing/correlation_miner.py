import logging
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union, Any
import core_utils as utils

logger = utils.get_logger(__name__)

class RelationshipMetrics(dict):
    """
    Represents statistical metrics (count and probability) for a mined relationship or effect.
    Allows both dictionary-like access (metrics['probability']) and object-like access (metrics.probability) 
    for maximum clarity and 100% backward-compatibility.
    
    Attributes:
        count (int): The absolute frequency of the observation in the log.
        probability (float): The relative probability/confidence of the effect (between 0.0 and 1.0).
    """
    def __init__(self, count: int, probability: float) -> None:
        super().__init__(count=count, probability=probability)

    @property
    def count(self) -> int:
        """The absolute frequency of the observation."""
        return self['count']

    @property
    def probability(self) -> float:
        """The relative probability/confidence of the effect (between 0.0 and 1.0)."""
        return self['probability']

    def __repr__(self) -> str:
        return f"RelationshipMetrics(count={self.count}, probability={self.probability:.3f})"


class CorrelationMiner:
    """
    Mines correlations: attribute-activity relationships and activity-attribute effects.
    """
    # Attribute names that should be excluded from mining (e.g. process control variables)
    IGNORED_ATTRIBUTES = {'case:concept:name', 'concept:name', 'time:timestamp', 'lifecycle:transition', 'org:resource', 'org:group', 'variant-index', 'Resource', 'org:role'}
    MIN_PROBABILITY_THRESHOLD = 0.1
    STRONG_PROBABILITY_THRESHOLD = 0.2
    MIN_SUPPORT_COUNT = 1
    MIN_SUPPORT_FOR_RELATIONSHIPS = 1

    def __init__(
        self, 
        full_log: Any, 
        predecessors: Dict[str, List[str]], 
        intervals: Dict[str, List[float]]
    ) -> None:
        """
        Initialize CorrelationMiner.

        Args:
            full_log: The full event log.
            predecessors: Predecessor mappings.
            intervals: Attribute discretization intervals.
        """
        self.full_log = full_log
        self.predecessors = predecessors
        self.intervals = intervals

    def discover_attribute_activity_relationships(self) -> Dict[str, Dict[str, Dict[str, RelationshipMetrics]]]:
        """
        Discover how specific attribute values influence subsequent activities.
        Analyzes the log to see if having a certain attribute value in an activity 
        leads to a specific subsequent target activity with high confidence.

        Returns:
            Nested dictionary of significant relationships mapping:
            Attribute -> AttributeValue -> TargetActivity -> RelationshipMetrics

            ESEMPIO DI OUTPUT RITORNATO (Nested Dictionary with Objects):
            {
                "case:Age": {
                    "young": {
                        "ER_Triage": RelationshipMetrics(count=24, probability=0.85)
                    },
                    "old": {
                        "Admission": RelationshipMetrics(count=15, probability=0.72)
                    }
                },
                "diagnosticLacticAcid": {
                    "0.0-1.5": {
                        "Release_A": RelationshipMetrics(count=45, probability=0.55)
                    }
                }
            }
        """
        # Step 1: Accumulate raw transition counts grouped by attribute value
        # Structure: relationships[attribute][value][target_activity] = count
        relationships = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for trace in tqdm(self.full_log, desc="Discovering attribute-activity relationships"):
            for i in range(len(trace) - 1):
                current_event = trace[i]
                next_event = trace[i+1]
                next_activity = next_event['concept:name']
                source_activity = current_event['concept:name']
                sanitized_source = utils.sanitize_name(source_activity)
                sanitized_target = utils.sanitize_name(next_activity)
                
                # Check if this transition is structural (exists in the discovered Petri net)
                if sanitized_target in self.predecessors and sanitized_source in self.predecessors[sanitized_target]:
                    for attr, val in current_event.items():
                        if attr in self.IGNORED_ATTRIBUTES:
                            continue
                        
                        sanitized_attr = utils.sanitize_name(attr)
                        relationships[sanitized_attr][str(val)][sanitized_target] += 1
                        
        significant_relationships = {}

        # Step 2: Filter out non-discriminating relationships and retain high-confidence ones
        for attr, values in relationships.items():
            if len(values) <= 1:
                # If an attribute has only one value throughout, it has no discriminative power
                logger.debug(f"Skipping attribute {attr} in relationship discovery (only {len(values)} unique value(s) found)")
                continue
                
            # Find activities that occur regardless of the attribute value
            common_activities = set.intersection(*[set(activities.keys()) for activities in values.values()])
            non_discriminative = {activity for activity in common_activities 
                                 if all(activities[activity] / sum(activities.values()) >= self.STRONG_PROBABILITY_THRESHOLD 
                                       for activities in values.values())}
            significant_attr_relationships = {}
            
            for val, activities in values.items():
                total = sum(activities.values())
                if total >= self.MIN_SUPPORT_FOR_RELATIONSHIPS:
                    # Map the raw dict values to structured RelationshipMetrics objects
                    significant_activities = {
                        activity: RelationshipMetrics(count=count, probability=count/total)
                        for activity, count in activities.items()
                        if activity not in non_discriminative and count/total >= 0.2
                    }
                    if significant_activities:
                        significant_attr_relationships[val] = significant_activities
            
            if significant_attr_relationships:
                significant_relationships[attr] = significant_attr_relationships
        
        return significant_relationships

    def discover_activity_attribute_effects(self) -> Dict[str, Dict[str, Dict[str, Dict[str, RelationshipMetrics]]]]:
        """
        Discover how activities affect attribute values.
        Identifies two kinds of effects:
        1. Transitions: When an activity changes an existing attribute from value A to value B.
        2. Introductions ('NEW'): When an activity initializes/introduces a new attribute.

        Returns:
            Nested dictionary of significant attribute effects mapping:
            SourceActivity -> Attribute -> PreValue/'NEW' -> PostValue -> RelationshipMetrics

            ESEMPIO DI OUTPUT RITORNATO (Nested Dictionary with Objects):
            {
                "ER_Triage": {
                    "case:Age": {
                        "NEW": {
                            "young": RelationshipMetrics(count=18, probability=0.90),
                            "old": RelationshipMetrics(count=2, probability=0.10)
                        }
                    }
                },
                "LacticAcid_Measurement": {
                    "lacticacid": {
                        "0.0-1.5": {
                            "1.5-3.0": RelationshipMetrics(count=5, probability=0.25)
                        }
                    }
                }
            }
        """
        # Step 1: Collect transition changes and attribute introductions
        # effects structure: effects[activity][attribute][prev_value][new_value] = count
        effects = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        # new_value_introductions structure: introductions[activity][attribute][introduced_value] = count
        new_value_introductions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        valid_transitions = self._get_valid_transitions()
        self._collect_attribute_effects(effects, new_value_introductions, valid_transitions)
        
        # Step 2: Build the consolidated list of significant effects
        return self._build_significant_effects(effects, new_value_introductions)

    def _get_valid_transitions(self) -> Set[Tuple[str, str]]:
        """
        Helper to get structural transitions (source -> target) from predecessor list.

        Returns:
            Set of tuples representing structural transition pairs.
            E.g. {("ER_Registration", "ER_Triage"), ("ER_Triage", "LacticAcid_Measurement")}
        """
        return {
            (source_activity, target_activity)
            for target_activity, source_activities in self.predecessors.items() for source_activity in source_activities
        }

    def _collect_attribute_effects(
        self, 
        effects: Dict[str, Dict[str, Dict[str, Dict[str, int]]]], 
        new_value_introductions: Dict[str, Dict[str, Dict[str, int]]], 
        valid_transitions: Set[Tuple[str, str]]
    ) -> None:
        """
        Helper to iterate the log and collect raw counts for effects and introductions.

        Args:
            effects: Raw dictionary to accumulate value transitions.
                Structure: { activity: { attribute: { prev_val: { curr_val: count } } } }
                E.g. {"ER_Triage": {"lacticacid": {"0.0-1.5": {"1.5-3.0": 5}}}}
            new_value_introductions: Raw dictionary to accumulate first-time attribute values.
                Structure: { activity: { attribute: { value: count } } }
                E.g. {"ER_Registration": {"case:Age": {"young": 180}}}
            valid_transitions: Set of structural transitions.
        """
        for trace in tqdm(self.full_log, desc="Collecting attribute effects"):
            attributes_seen_in_trace = set()
            prev_event = None
            
            for event in trace:
                activity = event['concept:name']
                sanitized_activity = utils.sanitize_name(activity)
                
                # Check for attribute introductions (first occurrence in trace)
                self._track_new_attribute_introductions(
                    event, sanitized_activity, attributes_seen_in_trace, new_value_introductions
                )
                
                # Check for value transitions between consecutive activities
                if prev_event is not None:
                    self._track_attribute_changes(
                        prev_event, event, sanitized_activity, valid_transitions, effects
                    )
                
                prev_event = event

    def _track_new_attribute_introductions(
        self, 
        event: Dict[str, Any], 
        sanitized_activity: str, 
        attributes_seen_in_trace: Set[str], 
        new_value_introductions: Dict[str, Dict[str, Dict[str, int]]]
    ) -> None:
        """
        Tracks the introduction of new attributes (not seen previously in the same trace).

        Args:
            event: The current event dictionary.
            sanitized_activity: Sanitized name of the current activity.
            attributes_seen_in_trace: Set tracking which attributes have already been encountered in this trace.
            new_value_introductions: Output dictionary to record raw frequency counts.
                Structure: { activity: { attribute: { value: count } } }
        """
        for attr, val in event.items():
            if attr in self.IGNORED_ATTRIBUTES:
                continue
            sanitized_attr = utils.sanitize_name(attr)
            if sanitized_attr not in attributes_seen_in_trace:
                discretized_val = utils.discretize_value(sanitized_attr, val, self.intervals)
                new_value_introductions[sanitized_activity][sanitized_attr][discretized_val] += 1
                attributes_seen_in_trace.add(sanitized_attr)

    def _track_attribute_changes(
        self, 
        prev_event: Dict[str, Any], 
        event: Dict[str, Any], 
        sanitized_activity: str, 
        valid_transitions: Set[Tuple[str, str]], 
        effects: Dict[str, Dict[str, Dict[str, Dict[str, int]]]]
    ) -> None:
        """
        Tracks changes in attribute values between consecutive events along structural transitions.

        Args:
            prev_event: The previous event dictionary.
            event: The current event dictionary.
            sanitized_activity: Sanitized name of the current activity.
            valid_transitions: Set of valid structural transition pairs.
            effects: Output dictionary to record transition frequencies.
                Structure: { prev_activity: { attribute: { prev_val: { curr_val: count } } } }
        """
        prev_activity = prev_event['concept:name']
        sanitized_prev_activity = utils.sanitize_name(prev_activity)
        if (sanitized_prev_activity, sanitized_activity) not in valid_transitions:
            return
        
        for attr in set(prev_event.keys()).intersection(event.keys()):
            if attr in self.IGNORED_ATTRIBUTES:
                continue

            prev_val = prev_event.get(attr)
            curr_val = event.get(attr)
            if prev_val == curr_val:
                continue
                
            sanitized_attr = utils.sanitize_name(attr)
            prev_val_str = str(prev_val).lower() if isinstance(prev_val, bool) else str(prev_val)
            curr_val_str = str(curr_val).lower() if isinstance(curr_val, bool) else str(curr_val)
            effects[sanitized_prev_activity][sanitized_attr][prev_val_str][curr_val_str] += 1

    def _build_significant_effects(
        self, 
        effects: Dict[str, Dict[str, Dict[str, Dict[str, int]]]], 
        new_value_introductions: Dict[str, Dict[str, Dict[str, int]]]
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, RelationshipMetrics]]]]:
        """
        Filters and shapes the collected raw counts into structured significant effects.

        Args:
            effects: Raw value transition counts.
            new_value_introductions: Raw attribute introduction counts.

        Returns:
            Nested dictionary of significant effects.
            Structure: { activity: { attribute: { prev_val/'NEW': { curr_val: RelationshipMetrics } } } }
        """
        significant_effects = {}
        
        # 1. Process value-to-value transitions
        for activity, attributes in effects.items():
            significant_effects[activity] = {}
            for attr, from_values in attributes.items():
                significant_from_values = self._filter_significant_transitions(from_values)
                if significant_from_values:
                    significant_effects[activity][attr] = significant_from_values
                    
        # 2. Process attribute introductions ('NEW')
        for activity, attributes in new_value_introductions.items():
            if activity not in significant_effects:
                significant_effects[activity] = {}
            
            for attr, values in attributes.items():
                total_introductions = sum(values.values())
                if total_introductions >= 1:
                    if attr not in significant_effects[activity]:
                        significant_effects[activity][attr] = {}
                    
                    significant_effects[activity][attr]['NEW'] = {}
                    for val, count in values.items():
                        probability = count / total_introductions
                        if probability >= self.MIN_PROBABILITY_THRESHOLD:
                            # Wrap metrics in RelationshipMetrics object
                            significant_effects[activity][attr]['NEW'][val] = RelationshipMetrics(
                                count=count,
                                probability=probability
                            )
        
        return significant_effects

    def _filter_significant_transitions(self, from_values: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, RelationshipMetrics]]:
        """
        Filters out low-probability value transitions and wraps the results in RelationshipMetrics objects.

        Args:
            from_values: Dictionary of target value counts from a specific starting value.
                Structure: { prev_val: { curr_val: count } }
                E.g. {"0.0-1.5": {"1.5-3.0": 10, "3.0-5.0": 1}}

        Returns:
            Filtered dictionary with mapped metrics objects:
            Structure: { prev_val: { curr_val: RelationshipMetrics } }
            E.g. {"0.0-1.5": {"1.5-3.0": RelationshipMetrics(count=10, probability=0.91)}}
        """
        significant_from_values = {}
        
        for from_val, to_values in from_values.items():
            total_transitions = sum(to_values.values())
            if total_transitions >= self.MIN_SUPPORT_COUNT:
                significant_to_values = {}
                for to_val, count in to_values.items():
                    probability = count / total_transitions
                    if probability >= self.MIN_PROBABILITY_THRESHOLD:
                        # Wrap metrics in RelationshipMetrics object
                        significant_to_values[to_val] = RelationshipMetrics(
                            count=count,
                            probability=probability
                        )
                
                if significant_to_values:
                    significant_from_values[from_val] = significant_to_values
        
        return significant_from_values
