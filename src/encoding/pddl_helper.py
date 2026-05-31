import re
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple, Set, Union
import pandas as pd
import pm4py

# Import core utilities directly to avoid circular dependency
from core_utils import get_logger, sanitize_name, discretize_value

logger = get_logger(__name__)


def extract_base_activity_name(action_name: str) -> str:
    """
    Extract the base activity name by removing discretized measurement suffixes or fallback markers.

    Args:
        action_name: The full action name string.

    Returns:
        The base activity name.

    ESEMPI:
        "crp-gte_12_15" -> "crp"
        "crp_very_low" -> "crp"
        "ER_Triage_fallback" -> "ER_Triage"
    """
    if not action_name:
        return action_name
    
    # Handle measurement variants using hyphen or underscore separators
    # Patterns: activity-gte_..., activity_gte_..., or activity-lte_.../activity_lte_...
    m = re.match(r'^(.+?)[-_](gte_|lte_|gt_|lt_|eq_).+$', action_name)
    if m:
        return m.group(1)
    
    # Handle discretized suffixes with underscores
    measurement_suffixes = [
        '_very_low', '_low', '_medium', '_high', '_very_high'
    ]
    for suffix in measurement_suffixes:
        if action_name.endswith(suffix):
            return action_name[:-len(suffix)]
    
    # Handle single character suffixes (legacy pattern)
    if len(action_name) > 2 and action_name[-2] == '_' and action_name[-1].isalpha():
        return action_name[:-2]

    # Handle fallback variants (e.g., action_fallback, action_fallback_v1)
    fallback_match = re.search(r'(.+?)_fallback(?:_v\d+)?$', action_name)
    if fallback_match:
        return fallback_match.group(1)

    return action_name


def find_attribute_key_in_event(event: Dict[str, Any], attr_name: str) -> Optional[str]:
    """
    Find the actual matching key used in an event dictionary for a given attribute name,
    considering case sensitivity and common formatting differences.

    Args:
        event: The event dictionary to inspect.
        attr_name: The attribute name to look for.

    Returns:
        The matching key from the event dictionary, or None if not found.

    ESEMPIO:
        event={"case:Age": 45}, attr_name="age" -> "case:Age"
    """
    variations = [
        attr_name,
        attr_name.capitalize(),
        attr_name.upper(),
        attr_name.lower(),
        attr_name.title(),
        attr_name.replace('_', ':'),
        attr_name.replace(':', '_'),
        attr_name.replace(' ', '_'),
        attr_name.replace('_', ' ')
    ]
    for variation in variations:
        if variation in event:
            return variation
    attr_normalized = attr_name.lower().replace(':', '_').replace(' ', '_')
    for key in event.keys():
        key_normalized = key.lower().replace(':', '_').replace(' ', '_')
        if key_normalized == attr_normalized:
            return key
    
    return None


def extract_attribute_predicates(parser: Any, event: Dict[str, Any], include_target: Optional[str] = None) -> List[str]:
    """
    Extract PDDL attribute predicates from a single event dictionary based on discretization thresholds.

    Args:
        parser: The XES parser instance containing intervals.
        event: The event dictionary containing attribute values.
        include_target: Optional attribute name to specifically include.

    Returns:
        A list of PDDL predicate strings.

    ESEMPIO DI OUTPUT:
        ["(crp lte_10_0)", "(diagnosticlacticacid lte_1_5)"]
    """
    predicates = []
    if hasattr(parser, 'decision_thresholds') and parser.decision_thresholds:
        for attr_name in parser.decision_thresholds.keys():
            if include_target and attr_name != include_target:
                continue
                
            trace_attr_key = find_attribute_key_in_event(event, attr_name)
            if trace_attr_key and trace_attr_key in event:
                raw_value = event[trace_attr_key]
                
                if raw_value is not None and not pd.isna(raw_value):
                    try:
                        numeric_value = float(raw_value)
                        discretized_value = discretize_value(attr_name, numeric_value, parser.intervals)
                        
                        if discretized_value:
                            sanitized_attr = sanitize_name(attr_name)
                            predicate = f"({sanitized_attr} {discretized_value})"
                            predicates.append(predicate)
                    except (ValueError, TypeError):
                        continue
    
    return predicates


def get_first_attribute_value_in_trace(parser: Any, trace_events: List[Dict[str, Any]], target_attr: str) -> Optional[float]:
    """
    Finds the first available numeric value for an attribute in a trace.

    Args:
        parser: The XES parser instance.
        trace_events: List of event dictionaries representing the trace.
        target_attr: The target attribute name.

    Returns:
        The first numeric float value found, or None.
    """
    if not trace_events:
        return None
    
    for event in trace_events:
        trace_attr_key = find_attribute_key_in_event(event, target_attr)
        if trace_attr_key and trace_attr_key in event:
            raw_value = event[trace_attr_key]
            if raw_value is not None and not pd.isna(raw_value):
                try:
                    return float(raw_value)
                except (ValueError, TypeError):
                    continue
    return None


def extract_initial_attribute_predicates(
    parser: Any, 
    trace_events: List[Dict[str, Any]], 
    include_target: Optional[str] = None
) -> List[str]:
    """
    Extract initial attribute predicates using the first available value of each attribute in the trace.
    Handles both numerical (discretized) and categorical attributes.

    Args:
        parser: XES parser instance.
        trace_events: List of event dictionaries representing the trace.
        include_target: Optional attribute name to specifically filter by.

    Returns:
        List of PDDL predicate strings.
    """
    predicates = []
    added_attrs = set()
    
    # 1. Handle numerical attributes with thresholds (discretized)
    if hasattr(parser, 'decision_thresholds') and parser.decision_thresholds:
        for attr_name in parser.decision_thresholds.keys():
            if include_target and attr_name != include_target:
                continue
                
            first_value = get_first_attribute_value_in_trace(parser, trace_events, attr_name)
            
            if first_value is not None:
                discretized_value = discretize_value(attr_name, first_value, parser.intervals)
                if discretized_value:
                    sanitized_attr = sanitize_name(attr_name)
                    predicate = f"({sanitized_attr} {discretized_value})"
                    predicates.append(predicate)
                    added_attrs.add(sanitized_attr)
    
    # 2. Handle categorical attributes from decision samples
    if hasattr(parser, 'decision_samples') and parser.decision_samples:
        categorical_attrs = set()
        for sample in parser.decision_samples:
            for precond in sample.get('preconditions', []):
                if isinstance(precond, str) and precond.startswith('(') and precond.endswith(')'):
                    inner = precond[1:-1]
                    parts = inner.split()
                    if len(parts) == 2:
                        attr_name, _ = parts
                        if attr_name not in added_attrs:
                            category = parser.attribute_categories.get(attr_name, '')
                            if category == 'categorical':
                                categorical_attrs.add(attr_name)
        
        for sanitized_attr in categorical_attrs:
            attr_value = get_categorical_attribute_value_from_trace(parser, trace_events, sanitized_attr)
            if attr_value:
                predicate = f"({sanitized_attr} {attr_value})"
                predicates.append(predicate)
                added_attrs.add(sanitized_attr)
    
    return predicates


def get_categorical_attribute_value_from_trace(
    parser: Any, 
    trace_events: List[Dict[str, Any]], 
    sanitized_attr: str
) -> Optional[str]:
    """
    Finds and sanitizes the value of a categorical attribute in a trace.

    Args:
        parser: The XES parser instance.
        trace_events: List of event dictionaries representing the trace.
        sanitized_attr: Sanitized attribute name to look for.

    Returns:
        The sanitized string value, or None.
    """
    if not trace_events:
        return None
    
    attribute_domains = getattr(parser, 'attribute_domains', {})
    valid_values = attribute_domains.get(sanitized_attr, set())
    first_event = trace_events[0]
    
    # Check case/trace level attributes
    for key, value in first_event.items():
        key_sanitized = sanitize_name(key)
        if key_sanitized == sanitized_attr or key_sanitized == f"case_{sanitized_attr}":
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                sanitized_value = sanitize_name(str(value))
                if sanitized_value and not sanitized_value[0].isalpha():
                    sanitized_value = f"e{sanitized_value}"
                if not valid_values or sanitized_value in valid_values:
                    return sanitized_value
    
    # Check event level attributes
    for event in trace_events:
        for key, value in event.items():
            key_sanitized = sanitize_name(key)
            if key_sanitized == sanitized_attr:
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    sanitized_value = sanitize_name(str(value))
                    if sanitized_value and not sanitized_value[0].isalpha():
                        sanitized_value = f"e{sanitized_value}"
                    if not valid_values or sanitized_value in valid_values:
                        return sanitized_value
    
    return None


def compute_enabled_activities_after_prefix(parser: Any, prefix: List[str]) -> Set[str]:
    """
    Compute which activities should be enabled after executing a prefix by replaying the prefix
    on the discovered Petri net.

    Args:
        parser: The XES parser containing the Petri net, initial marking, and transitions.
        prefix: List of executed activity name strings.

    Returns:
        Set of enabled activity names.
    """
    from pm4py.objects.petri_net import semantics
    
    if not prefix:
        if hasattr(parser, 'start_activities'):
            return set(parser.start_activities.keys()) if isinstance(parser.start_activities, dict) else set(parser.start_activities)
        return set()
    
    if not hasattr(parser, 'petrinet') or not hasattr(parser, 'initial_marking'):
        return _compute_enabled_from_graph(parser, prefix)
    
    # Map sanitized activity name to transition objects
    activity_to_transitions = {}
    for t in parser.petrinet.transitions:
        if t.label is not None:
            sanitized = sanitize_name(t.label)
            if sanitized not in activity_to_transitions:
                activity_to_transitions[sanitized] = []
            activity_to_transitions[sanitized].append(t)
        else:
            for tau_name, tau_t in getattr(parser, 'silent_transitions', []):
                if tau_t == t:
                    if tau_name not in activity_to_transitions:
                        activity_to_transitions[tau_name] = []
                    activity_to_transitions[tau_name].append(t)
                    break
    
    current_marking = parser.initial_marking.copy() if hasattr(parser.initial_marking, 'copy') else dict(parser.initial_marking)
    
    from pm4py.objects.petri_net.obj import Marking
    if not isinstance(current_marking, Marking):
        marking_obj = Marking()
        for place, tokens in current_marking.items():
            marking_obj[place] = tokens
        current_marking = marking_obj
    
    # Replay prefix events
    for activity in prefix:
        transitions = activity_to_transitions.get(activity, [])
        fired = False
        for t in transitions:
            if semantics.is_enabled(t, parser.petrinet, current_marking):
                current_marking = semantics.execute(t, parser.petrinet, current_marking)
                fired = True
                break
        
        if not fired:
            # Try to fire tau transitions to enable the activity
            max_tau_attempts = 50
            for _ in range(max_tau_attempts):
                tau_fired = False
                for t in parser.petrinet.transitions:
                    if t.label is None and semantics.is_enabled(t, parser.petrinet, current_marking):
                        current_marking = semantics.execute(t, parser.petrinet, current_marking)
                        tau_fired = True
                        break
                
                if not tau_fired:
                    break
                    
                for t in transitions:
                    if semantics.is_enabled(t, parser.petrinet, current_marking):
                        current_marking = semantics.execute(t, parser.petrinet, current_marking)
                        fired = True
                        break
                if fired:
                    break
    
    # Collect enabled transitions from the resulting marking
    enabled = set()
    for t in parser.petrinet.transitions:
        if semantics.is_enabled(t, parser.petrinet, current_marking):
            if t.label is not None:
                enabled.add(sanitize_name(t.label))
            else:
                for tau_name, tau_t in getattr(parser, 'silent_transitions', []):
                    if tau_t == t:
                        enabled.add(tau_name)
                        break
    
    return enabled


def compute_enabled_activities_after_last_activity(parser: Any, last_activity: str) -> Set[str]:
    """
    Compute enabled activities successor set immediately after executing a single activity,
    expanding variants/measurements as necessary.

    Args:
        parser: The XES parser containing process model data.
        last_activity: The executed activity name string.

    Returns:
        Set of enabled activity names.
    """
    enabled = set()
    if not last_activity:
        return enabled

    if hasattr(parser, 'petrinet') and hasattr(parser, 'initial_marking'):
        try:
            enabled_from_prefix = compute_enabled_activities_after_prefix(parser, [last_activity])
            if enabled_from_prefix:
                enabled |= enabled_from_prefix
        except Exception:
            pass

    if hasattr(parser, 'direct_transition_graph') and parser.direct_transition_graph:
        succs = parser.direct_transition_graph.get(last_activity, [])
        for s in succs:
            enabled.add(s)

    for activity in getattr(parser, 'activities', []):
        if activity is None:
            continue
        if activity.startswith(last_activity + '_') or activity == last_activity:
            if hasattr(parser, 'direct_transition_graph') and parser.direct_transition_graph:
                succs = parser.direct_transition_graph.get(activity, [])
                for s in succs:
                    enabled.add(s)

    if not enabled and hasattr(parser, 'petrinet') and hasattr(parser, 'initial_marking'):
        try:
            from pm4py.objects.petri_net import semantics
            activity_to_transitions = {}
            for t in parser.petrinet.transitions:
                if t.label is not None:
                    sanitized = sanitize_name(t.label)
                    if sanitized not in activity_to_transitions:
                        activity_to_transitions[sanitized] = []
                    activity_to_transitions[sanitized].append(t)
                else:
                    for tau_name, tau_t in getattr(parser, 'silent_transitions', []):
                        if tau_t == t:
                            if tau_name not in activity_to_transitions:
                                activity_to_transitions[tau_name] = []
                            activity_to_transitions[tau_name].append(t)
                            break

            from pm4py.objects.petri_net.obj import Marking
            current_marking = parser.initial_marking.copy() if hasattr(parser.initial_marking, 'copy') else dict(parser.initial_marking)
            if not isinstance(current_marking, Marking):
                marking_obj = Marking()
                for place, tokens in current_marking.items():
                    marking_obj[place] = tokens
                current_marking = marking_obj

            transitions = activity_to_transitions.get(last_activity, [])
            fired = False
            for t in transitions:
                if semantics.is_enabled(t, parser.petrinet, current_marking):
                    current_marking = semantics.execute(t, parser.petrinet, current_marking)
                    fired = True
                    break

            if fired:
                for t in parser.petrinet.transitions:
                    if semantics.is_enabled(t, parser.petrinet, current_marking):
                        if t.label is not None:
                            enabled.add(sanitize_name(t.label))
                        else:
                            for tau_name, tau_t in getattr(parser, 'silent_transitions', []):
                                if tau_t == t:
                                    enabled.add(tau_name)
                                    break
        except Exception:
            pass

    return enabled


def compute_reachable_activities_from_last_activity(parser: Any, last_activity: str, max_depth: int = 10) -> Set[str]:
    """
    Graph-based reachability from a given activity up to max_depth hops.

    Args:
        parser: The XES parser.
        last_activity: The starting activity name.
        max_depth: Maximum hops to search.

    Returns:
        Set of reachable activity names.
    """
    reachable = set()
    if not last_activity:
        return reachable

    if hasattr(parser, 'direct_transition_graph') and parser.direct_transition_graph:
        frontier = [last_activity]
        depth = 0
        while frontier and depth < max_depth:
            next_frontier = []
            for a in frontier:
                succs = parser.direct_transition_graph.get(a, [])
                for s in succs:
                    if s not in reachable:
                        reachable.add(s)
                        next_frontier.append(s)
            frontier = next_frontier
            depth += 1

    return reachable


def _compute_enabled_from_graph(parser: Any, prefix: List[str]) -> Set[str]:
    """
    Fallback method using direct transition graph when Petri net is not available.
    """
    enabled = set()
    completed = set(prefix)
    
    if hasattr(parser, 'direct_transition_graph'):
        activity_predecessors = defaultdict(set)
        for pred, successors in parser.direct_transition_graph.items():
            for succ in successors:
                activity_predecessors[succ].add(pred)
        
        for activity in parser.activities:
            if activity in completed:
                continue
            predecessors = activity_predecessors.get(activity, set())
            if predecessors and any(pred in completed for pred in predecessors):
                enabled.add(activity)
    
    return enabled


def compute_initial_state(
    parser: Any, 
    prefix: List[str], 
    trace_events: Optional[List[Dict[str, Any]]] = None, 
    prefix_length: int = 0, 
    include_target_attr: Optional[str] = None
) -> List[str]:
    """
    Generates initial PDDL predicates after executing the prefix.

    Args:
        parser: The XES parser.
        prefix: List of executed activities.
        trace_events: Full event trace.
        prefix_length: Unused.
        include_target_attr: Target attribute filter.

    Returns:
        List of PDDL predicate strings.
    """
    if not prefix:
        init_predicates = []
        if hasattr(parser, 'start_activities') and parser.start_activities:
            for start_activity in parser.start_activities:
                init_predicates.append(f"(enabled {start_activity})")
        
        if trace_events and len(trace_events) > 0:
            init_predicates.extend(extract_initial_attribute_predicates(parser, trace_events, include_target_attr))
        
        return init_predicates
    
    init_predicates = []
    for activity in prefix:
        init_predicates.append(f"(completed {activity})")
    
    enabled_activities = compute_enabled_activities_after_prefix(parser, prefix)
    for activity in enabled_activities:
        init_predicates.append(f"(enabled {activity})")
    
    if trace_events and len(trace_events) > 0:
        init_predicates.extend(extract_initial_attribute_predicates(parser, trace_events, include_target_attr))
    
    return init_predicates


def compute_initial_state_from_last_activity(
    parser: Any, 
    prefix: List[str], 
    trace_events: Optional[List[Dict[str, Any]]] = None, 
    prefix_length: int = 0, 
    include_target_attr: Optional[str] = None
) -> List[str]:
    """
    Generates initial PDDL predicates enabling only successors of the last activity in prefix.
    """
    if not prefix:
        init_predicates = []
        if hasattr(parser, 'start_activities') and parser.start_activities:
            for start_activity in parser.start_activities:
                init_predicates.append(f"(enabled {start_activity})")
        if trace_events and len(trace_events) > 0:
            init_predicates.extend(extract_initial_attribute_predicates(parser, trace_events, include_target_attr))
        return init_predicates

    init_predicates = []
    for activity in prefix:
        init_predicates.append(f"(completed {activity})")

    last_activity = prefix[-1]
    enabled_activities = compute_enabled_activities_after_last_activity(parser, last_activity)
    for activity in enabled_activities:
        init_predicates.append(f"(enabled {activity})")

    if trace_events and len(trace_events) > 0:
        init_predicates.extend(extract_initial_attribute_predicates(parser, trace_events, include_target_attr))

    return init_predicates


def compute_goal_condition(
    parser: Any, 
    target_activity: str, 
    trace_events: Optional[List[Dict[str, Any]]] = None, 
    trace_length: int = 0, 
    target_attr: Optional[str] = None
) -> List[str]:
    """
    Generates target goal PDDL predicates.
    """
    goal_predicates = [f"(completed {target_activity})"]
    is_measurement_activity = False
    matching_attribute = target_attr
    
    if not matching_attribute and hasattr(parser, 'decision_thresholds') and parser.decision_thresholds:
        for attr_name in parser.decision_thresholds.keys():
            activity_clean = target_activity.lower().replace('_', '').replace(':', '').replace(' ', '')
            attr_clean = attr_name.lower().replace('_', '').replace(':', '').replace(' ', '')
            if activity_clean == attr_clean:
                is_measurement_activity = True
                matching_attribute = attr_name
                break
    
    if is_measurement_activity and matching_attribute:
        if trace_events and trace_length > 0:
            target_event = None
            for i in range(trace_length - 1, -1, -1):
                event = trace_events[i]
                event_activity = event.get('concept:name', '').lower().replace(' ', '').replace(':', '').replace('_', '')
                if event_activity == target_activity.lower().replace(' ', '').replace(':', '').replace('_', ''):
                    target_event = event
                    break
            
            if target_event:
                trace_attr_key = find_attribute_key_in_event(target_event, matching_attribute)
                if trace_attr_key and trace_attr_key in target_event:
                    raw_value = target_event[trace_attr_key]
                    if raw_value is not None and not pd.isna(raw_value):
                        try:
                            numeric_value = float(raw_value)
                            discretized_value = discretize_value(matching_attribute, numeric_value, parser.intervals)
                            if discretized_value:
                                sanitized_attr = sanitize_name(matching_attribute)
                                goal_predicates.append(f"({sanitized_attr} {discretized_value})")
                        except (ValueError, TypeError):
                            pass
    
    return goal_predicates


def read_state_file(file_path: str) -> List[str]:
    """
    Reads a state file containing PDDL predicates.
    """
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.warning(f"State file not found at {file_path}. Using empty state.")
        return []
