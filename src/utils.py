import os
import re
import sys
import subprocess

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Set, Union
import pandas as pd
import pm4py
import re


def create_base_argument_parser(description: str = "Run evaluation framework") -> ArgumentParser:
    """
    Create a base ArgumentParser with common arguments for the evaluation framework.

    Args:
        description: A brief description of the parser's purpose.

    Returns:
        The initialized ArgumentParser instance.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument('--log_coverage', type=float, default=0.001, 
                        help='Minimum cumulative coverage percentage for variant filtering (pm4py).')
    parser.add_argument('--xes_name', type=str, default='sepsis.xes', 
                        help='Name of the XES event log to use.')
    parser.add_argument('--pddl_name', type=str, default='xes_log', 
                        help='Name of the PDDL domain to generate.')
    parser.add_argument('--init', type=str, default='../pddl/input/init.pddl', 
                        help='Path to the file containing initial state predicates.')
    parser.add_argument('--goal', type=str, default='../pddl/input/goal_test.pddl', 
                        help='Path to the file containing goal state predicates.')
    parser.add_argument('--search', default="astar_lmcut", type=str,
                        choices=["astar_blind", "astar_hadd", "astar_hff", "astar_lmcut",
                                "eager_greedy_blind", "eager_greedy_hadd", "eager_greedy_hff",
                                "eager_greedy_lmcut", "seq_sat_lama_2011", "seq_opt_bjolp", "lama_first"],
                        help="Search algorithm to use for planning")
    parser.add_argument('--discovery_algorithm', type=str, default='alpha',
                        choices=['alpha', 'inductive', 'heuristics', 'ilp'],
                        help='Discovery algorithm to use for Petri net discovery. '
                             'Options: alpha (Alpha Miner), inductive (Inductive Miner), '
                             'heuristics (Heuristics Miner), ilp (ILP Miner)')
    parser.add_argument('--use_activity_classifier', action='store_true',
                        help='Use Activity classifier (combine concept:name + lifecycle:transition) '
                             'to identify activities. Use this for logs where events are not '
                             'uniquely identified by concept:name alone (e.g., BPIC 2013).')
    parser.add_argument('--max-samples', type=int, default=700,
                        help='Maximum number of generated samples (prefixes, cases) to evaluate overall per evaluation. Defaults to 700.')
    parser.add_argument('--max-traces', type=int, default=None,
                        help='Optional: maximum number of traces to read from the log. If not specified, all traces are used.')
    return parser


def extract_base_activity_name(action_name: str) -> str:
    """
    Extract the base activity name from measurement variants.
    
    Handles various patterns:
    - activity-gte_12_15 -> activity
    - activity-lte_3_4 -> activity
    - activity_very_low -> activity
    
    Args:
        action_name: String representing an action name
        
    Returns:
        Base activity name without measurement suffix
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


def filter_tau_activities_from_plan(plan_content: str) -> Tuple[str, List[str]]:
    """
    Filter tau (silent transition) activities from a plan content string.
    
    Args:
        plan_content: String content of a plan file
        
    Returns:
        Tuple of (filtered_plan_content, filtered_actions_list)
    """
    if not plan_content:
        return "", []
    
    plan_lines = plan_content.strip().split('\n')
    filtered_plan_lines = []
    filtered_actions = []
    
    for line in plan_lines:
        if line.strip() and not line.strip().startswith(';'):
            action_part = line.strip().strip('()')
            if action_part:
                action_name = action_part.split()[0]
                if not action_name.startswith('tau_'):
                    filtered_plan_lines.append(line)
                    filtered_actions.append(action_name)
            else:
                filtered_plan_lines.append(line)
        else:
            filtered_plan_lines.append(line)
    
    return '\n'.join(filtered_plan_lines), filtered_actions


def parse_plan_actions(plan_content: str) -> List[str]:
    """
    Parse action names from a plan content string, filtering out tau activities.
    
    Args:
        plan_content: String content of a plan file
        
    Returns:
        List of action names (without tau activities)
    """
    _, filtered_actions = filter_tau_activities_from_plan(plan_content)
    return filtered_actions


def find_attribute_key_in_event(event: Dict[str, Any], attr_name: str) -> Optional[str]:
    """
    Find the actual key used in an event dictionary for a given attribute name,
    considering various case and formatting variations.

    Args:
        event: The event dictionary to search in.
        attr_name: The attribute name to look for.

    Returns:
        The matching key from the event dictionary, or None if not found.
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
    Extract PDDL attribute predicates from an event based on decision thresholds.

    Args:
        parser: The XES parser instance containing decision thresholds and intervals.
        event: The event dictionary containing attribute values.
        include_target: Optional attribute name to specifically include.

    Returns:
        A list of PDDL predicate strings (e.g., "(attr value)").
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
    Get the first available value for a specific attribute in the trace events.
    Searches through all events in the trace to find the first non-null value.
    
    Args:
        parser: XES parser instance
        trace_events: List of event dictionaries
        target_attr: Target attribute name
        
    Returns:
        float or None: First numeric value found, or None
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


def extract_initial_attribute_predicates(parser: Any, trace_events: List[Dict[str, Any]], include_target: Optional[str] = None) -> List[str]:
    """
    Extract initial attribute predicates using the first available value of each attribute in the trace.
    This includes both numerical attributes (discretized via thresholds) and categorical attributes
    used in decision preconditions.
    
    Args:
        parser: XES parser instance
        trace_events: List of event dictionaries
        include_target: If specified, only include this attribute
        
    Returns:
        list: PDDL predicates for initial attribute values
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
    # These are attributes used in action preconditions that must be initialized
    if hasattr(parser, 'decision_samples') and parser.decision_samples:
        # Collect all categorical attributes used in preconditions
        categorical_attrs = set()
        for sample in parser.decision_samples:
            for precond in sample.get('preconditions', []):
                if isinstance(precond, str) and precond.startswith('(') and precond.endswith(')'):
                    inner = precond[1:-1]
                    parts = inner.split()
                    if len(parts) == 2:
                        attr_name, _ = parts
                        # Check if it's a categorical attribute (not already handled as numerical)
                        if attr_name not in added_attrs:
                            category = parser.attribute_categories.get(attr_name, '')
                            if category == 'categorical':
                                categorical_attrs.add(attr_name)
        
        # Extract values for categorical attributes from trace
        for sanitized_attr in categorical_attrs:
            # Find the original attribute name in the trace
            attr_value = get_categorical_attribute_value_from_trace(parser, trace_events, sanitized_attr)
            if attr_value:
                predicate = f"({sanitized_attr} {attr_value})"
                predicates.append(predicate)
                added_attrs.add(sanitized_attr)
    
    return predicates


def get_categorical_attribute_value_from_trace(parser: Any, trace_events: List[Dict[str, Any]], sanitized_attr: str) -> Optional[str]:
    """
    Get the value of a categorical attribute from the trace.
    Searches through trace attributes (case-level) and event attributes.
    
    Args:
        parser: XES parser instance
        trace_events: List of event dictionaries
        sanitized_attr: Sanitized attribute name to look for
        
    Returns:
        str: Sanitized value for the attribute, or None if not found
    """
    if not trace_events:
        return None
    
    # Get attribute domains for valid values
    attribute_domains = getattr(parser, 'attribute_domains', {})
    valid_values = attribute_domains.get(sanitized_attr, set())
    
    # Try to find the attribute in trace/case attributes first
    first_event = trace_events[0]
    
    # Check for case-level attributes (prefixed with 'case:')
    for key, value in first_event.items():
        key_sanitized = sanitize_name(key)
        # Handle 'case:attribute' -> 'case_attribute'
        if key_sanitized == sanitized_attr or key_sanitized == f"case_{sanitized_attr}":
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                sanitized_value = sanitize_name(str(value))
                # Prefix with 'e' if needed for PDDL compatibility
                if sanitized_value and not sanitized_value[0].isalpha():
                    sanitized_value = f"e{sanitized_value}"
                # Verify it's a valid value from the domain
                if not valid_values or sanitized_value in valid_values:
                    return sanitized_value
    
    # Check event-level attributes
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
    Compute which activities should be enabled after executing a prefix.
    
    Uses pm4py to replay the prefix on the Petri net and determine which
    transitions are enabled based on the resulting token marking.
    
    Args:
        parser: XES parser instance with petrinet, initial_marking, and activities
        prefix: List of completed activities (sanitized names)
        
    Returns:
        set: Activities that should be enabled (sanitized names)
    """
    import pm4py
    from pm4py.objects.petri_net import semantics
    
    # Start activities are enabled only if prefix is empty
    if not prefix:
        if hasattr(parser, 'start_activities'):
            return set(parser.start_activities.keys()) if isinstance(parser.start_activities, dict) else set(parser.start_activities)
        return set()
    
    # Check if we have the Petri net
    if not hasattr(parser, 'petrinet') or not hasattr(parser, 'initial_marking'):
        # Fallback to old method if Petri net not available
        return _compute_enabled_from_graph(parser, prefix)
    
    # Build mapping from sanitized activity name to transition
    activity_to_transitions = {}
    for t in parser.petrinet.transitions:
        if t.label is not None:
            sanitized = sanitize_name(t.label)
            if sanitized not in activity_to_transitions:
                activity_to_transitions[sanitized] = []
            activity_to_transitions[sanitized].append(t)
        else:
            # Tau transition - use the tau name from parser
            for tau_name, tau_t in getattr(parser, 'silent_transitions', []):
                if tau_t == t:
                    if tau_name not in activity_to_transitions:
                        activity_to_transitions[tau_name] = []
                    activity_to_transitions[tau_name].append(t)
                    break
    
    # Start with initial marking
    current_marking = parser.initial_marking.copy() if hasattr(parser.initial_marking, 'copy') else dict(parser.initial_marking)
    
    # Convert to proper Marking object if needed
    from pm4py.objects.petri_net.obj import Marking
    if not isinstance(current_marking, Marking):
        marking_obj = Marking()
        for place, tokens in current_marking.items():
            marking_obj[place] = tokens
        current_marking = marking_obj
    
    # Replay the prefix to get the current marking
    for activity in prefix:
        transitions = activity_to_transitions.get(activity, [])
        fired = False
        for t in transitions:
            if semantics.is_enabled(t, parser.petrinet, current_marking):
                current_marking = semantics.execute(t, parser.petrinet, current_marking)
                fired = True
                break
        
        if not fired:
            # Try to fire tau transitions to reach a state where activity is enabled
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
                    
                # Check if the activity is now enabled
                for t in transitions:
                    if semantics.is_enabled(t, parser.petrinet, current_marking):
                        current_marking = semantics.execute(t, parser.petrinet, current_marking)
                        fired = True
                        break
                if fired:
                    break
    
    # Now find all enabled transitions from current marking
    enabled = set()
    for t in parser.petrinet.transitions:
        if semantics.is_enabled(t, parser.petrinet, current_marking):
            if t.label is not None:
                enabled.add(sanitize_name(t.label))
            else:
                # Find tau name
                for tau_name, tau_t in getattr(parser, 'silent_transitions', []):
                    if tau_t == t:
                        enabled.add(tau_name)
                        break
    
    return enabled


def compute_enabled_activities_after_last_activity(parser: Any, last_activity: str) -> Set[str]:
    """
    Compute the set of activities enabled immediately after the execution of a
    single activity (last_activity) — i.e., the successors of the last activity.

    This function returns only the direct successors for the given activity,
    expanding variants/measurements as needed to include all possible successors
    that could be enabled by any possible variant of the last activity.
    """
    enabled = set()
    if not last_activity:
        return enabled

    # Prefer Petri net simulation to compute enabled activities: this yields
    # a more accurate closure (including tau transitions). Fall back to
    # direct_transition_graph if petrinet is not present.
    if hasattr(parser, 'petrinet') and hasattr(parser, 'initial_marking'):
        # Use compute_enabled_activities_after_prefix directly
        try:
            enabled_from_prefix = compute_enabled_activities_after_prefix(parser, [last_activity])
            if enabled_from_prefix:
                enabled |= enabled_from_prefix
        except Exception:
            pass

    # Use direct transition graph as a fallback or to extend the set
    if hasattr(parser, 'direct_transition_graph') and parser.direct_transition_graph:
        succs = parser.direct_transition_graph.get(last_activity, [])
        for s in succs:
            enabled.add(s)

    # Include successors of measurement or variant activities that share a base name
    # Example: if last_activity is `crp` and we have `crp_high`, `crp_low`, include
    # the successors of these specific variants as well.
    for activity in getattr(parser, 'activities', []):
        if activity is None:
            continue
        if activity.startswith(last_activity + '_') or activity == last_activity:
            if hasattr(parser, 'direct_transition_graph') and parser.direct_transition_graph:
                succs = parser.direct_transition_graph.get(activity, [])
                for s in succs:
                    enabled.add(s)

    # Fall back to petri net replay if available (and if nothing found)
    if not enabled and hasattr(parser, 'petrinet') and hasattr(parser, 'initial_marking'):
        # Replay a fake marking with last_activity completed
        # This tries to find successors reachable immediately after last_activity
        try:
            from pm4py.objects.petri_net import semantics
            # Build mapping from sanitized activity name to transition
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

            # Build a marking starting from the initial marking
            from pm4py.objects.petri_net.obj import Marking
            current_marking = parser.initial_marking.copy() if hasattr(parser.initial_marking, 'copy') else dict(parser.initial_marking)
            if not isinstance(current_marking, Marking):
                marking_obj = Marking()
                for place, tokens in current_marking.items():
                    marking_obj[place] = tokens
                current_marking = marking_obj

            # Fire transitions for last_activity if enabled
            transitions = activity_to_transitions.get(last_activity, [])
            fired = False
            for t in transitions:
                if semantics.is_enabled(t, parser.petrinet, current_marking):
                    current_marking = semantics.execute(t, parser.petrinet, current_marking)
                    fired = True
                    break

            # If fired, collect enabled successors from current_marking
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
    Compute activities reachable from the last activity following direct transitions
    (graph-based) up to max_depth hops. If a Petri net is available, fallback to
    computing reachability via direct_transition_graph to avoid expensive replay.
    Returns a set of activity names.
    """
    reachable = set()
    if not last_activity:
        return reachable

    # Use direct transition graph expansion
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
    Fallback method using direct transition graph when Petri net not available.

    Args:
        parser: The XES parser instance.
        prefix: List of completed activities.

    Returns:
        Set of enabled activities.
    """
    """Fallback method using direct transition graph when Petri net not available."""
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
    Compute the initial state predicates for planning after a given prefix.

    Args:
        parser: The XES parser instance.
        prefix: List of activities already completed.
        trace_events: Optional full list of event dictionaries for the trace.
        prefix_length: Length of the prefix (unused).
        include_target_attr: Optional target attribute to focus on.

    Returns:
        List of PDDL initial state predicates.
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
    Create an initial state that contains completed predicates for the given prefix
    and enables only those activities that are immediate successors of the last
    activity in the prefix (and its variants).
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

    # Only enabled activities that are successors of the last activity
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
    Compute the goal condition predicates for planning.

    Args:
        parser: The XES parser instance.
        target_activity: The name of the target activity to reach.
        trace_events: Optional full list of event dictionaries for the trace.
        trace_length: The point in the trace representing the goal.
        target_attr: Optional target attribute for measurement-based goals.

    Returns:
        List of PDDL goal condition predicates.
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


def parse_fast_downward_output(stdout: str, stderr: str) -> Dict[str, Optional[float]]:
    """
    Parse Fast Downward stdout and stderr to extract planning metrics.

    Args:
        stdout: Standard output from the planner.
        stderr: Standard error from the planner.

    Returns:
        A dictionary containing extracted metrics like 'expanded_nodes', 'search_time', etc.
    """
    metrics = {'expanded_nodes': None,
               'space_used': None,
               'solution_length': None,
               'search_time': None,
               'grounding_time': None,
               'total_time': None}
    full_output = stdout + "\n" + stderr
    fd_stdout = ""
    fd_stderr = ""
    stdout_match = re.search(r'STDOUT:\s*\n(.*?)(?=STDERR:|---------------------------|\Z)', full_output, re.DOTALL)
    if stdout_match:
        fd_stdout = stdout_match.group(1)
    else:
        # If no STDOUT marker found, use the full stdout directly
        fd_stdout = stdout
        
    stderr_match = re.search(r'STDERR:\s*\n(.*?)(?=---------------------------|\Z)', full_output, re.DOTALL)
    if stderr_match:
        fd_stderr = stderr_match.group(1)
    else:
        # If no STDERR marker found, use the full stderr directly  
        fd_stderr = stderr
        
    fd_output = fd_stdout + "\n" + fd_stderr

    expanded_patterns = [
        r'Expanded (\d+) state\(s\)',
        r'Expanded nodes: (\d+)',
        r'(\d+) nodes expanded',
        r'Expansions: (\d+)'
    ]
    for pattern in expanded_patterns:
        match = re.search(pattern, fd_output)
        if match:
            metrics['expanded_nodes'] = int(match.group(1))
            break
    
    space_patterns = [
        r'Peak memory: ([\d.]+)\s*KB',
        r'Peak memory: ([\d.]+)\s*MB',
        r'Memory usage: ([\d.]+)\s*MB',
        r'Memory: ([\d.]+)\s*MB',
        r'(\d+)\s*KB peak memory'
    ]
    
    for pattern in space_patterns:
        match = re.search(pattern, fd_output)
        if match:
            value = float(match.group(1))
            if 'KB' in pattern:
                value = value / 1024.0  # Convert KB to MB
            metrics['space_used'] = value
            break
    
    length_patterns = [
        r'Plan length: (\d+) step\(s\)',
        r'Plan length: (\d+)',
        r'Solution length: (\d+)',
        r'(\d+) steps',
        r'Plan cost: (\d+)'
    ]
    
    for pattern in length_patterns:
        match = re.search(pattern, fd_output)
        if match:
            metrics['solution_length'] = int(match.group(1))
            break
    
    # Parse timing information
    # Search time patterns (using actual Fast Downward output format)
    search_time_patterns = [
        r'Search time: ([\d.]+)s',
        r'Actual search time: ([\d.]+)s',
        r'Total search time: ([\d.]+)s'
    ]
    for pattern in search_time_patterns:
        match = re.search(pattern, fd_output)
        if match:
            metrics['search_time'] = float(match.group(1))
            break
    
    # Grounding/translation/preprocessing time patterns
    # Look for the translator "Done!" line with wall-clock time
    grounding_time_patterns = [
        r'Done! \[([\d.]+)s CPU, ([\d.]+)s wall-clock\]',  # Translator output format
        r'Translator time: ([\d.]+)s',
        r'Translation time: ([\d.]+)s',
        r'Grounding time: ([\d.]+)s',
        r'Preprocessing time: ([\d.]+)s'
    ]
    for pattern in grounding_time_patterns:
        match = re.search(pattern, fd_output)
        if match:
            # For the first pattern with two groups, use wall-clock time (second group)
            if 'wall-clock' in pattern:
                metrics['grounding_time'] = float(match.group(2))
            else:
                metrics['grounding_time'] = float(match.group(1))
            break
    
    # Compute total time as sum of grounding + search
    # Only compute if both values are available
    if metrics['grounding_time'] is not None and metrics['search_time'] is not None:
        metrics['total_time'] = metrics['grounding_time'] + metrics['search_time']
    elif metrics['grounding_time'] is not None:
        metrics['total_time'] = metrics['grounding_time']
    elif metrics['search_time'] is not None:
        metrics['total_time'] = metrics['search_time']
    # Otherwise total_time remains None
    
    return metrics


# Solvability status constants
SOLVABILITY_SOLVED = "solved"
SOLVABILITY_UNSOLVABLE_STRUCTURAL = "unsolvable_structural"
SOLVABILITY_UNSOLVABLE_RESOURCE = "unsolvable_resource"


def classify_planner_result(return_code: int, plan_exists: bool, stdout: str = "", stderr: str = "") -> str:
    """
    Classify the planner result into solvability categories based on Fast Downward exit codes.
    
    Fast Downward exit codes:
    - 0: SUCCESS (plan found)
    - 11: SEARCH_UNSOLVABLE (task is provably unsolvable)
    - 12: SEARCH_UNSOLVED_INCOMPLETE (search ended without finding a solution)
    - 20: TRANSLATE_OUT_OF_MEMORY
    - 21: TRANSLATE_OUT_OF_TIME
    - 22: SEARCH_OUT_OF_MEMORY
    - 23: SEARCH_OUT_OF_TIME
    - 24: SEARCH_OUT_OF_MEMORY_AND_TIME
    
    Returns:
        str: One of SOLVABILITY_SOLVED, SOLVABILITY_UNSOLVABLE_STRUCTURAL, SOLVABILITY_UNSOLVABLE_RESOURCE
    """
    full_output = stdout + "\n" + stderr
    
    # Check for solution found
    if return_code == 0 and plan_exists:
        return SOLVABILITY_SOLVED
    
    # Check for provably unsolvable (structural)
    # Exit code 11: SEARCH_UNSOLVABLE - Task is provably unsolvable
    # Also check output for "Completely explored state space" which indicates exhaustive search
    if return_code == 11:
        return SOLVABILITY_UNSOLVABLE_STRUCTURAL
    
    # Check for "Completely explored state space" message (indicates exhaustive search proving unsolvability)
    if "Completely explored state space" in full_output:
        return SOLVABILITY_UNSOLVABLE_STRUCTURAL
    
    # Check for resource exhaustion
    # Exit codes 20-24 indicate memory/time exhaustion
    if return_code in [20, 21, 22, 23, 24]:
        return SOLVABILITY_UNSOLVABLE_RESOURCE
    
    # Check output for memory/time indicators
    memory_indicators = ["out of memory", "memory exhausted", "Memory limit exceeded"]
    time_indicators = ["out of time", "timeout", "time limit exceeded"]
    
    for indicator in memory_indicators:
        if indicator.lower() in full_output.lower():
            return SOLVABILITY_UNSOLVABLE_RESOURCE
    
    for indicator in time_indicators:
        if indicator.lower() in full_output.lower():
            return SOLVABILITY_UNSOLVABLE_RESOURCE
    
    # Exit code 12: SEARCH_UNSOLVED_INCOMPLETE - incomplete search (could be resource or structural)
    # Without "Completely explored state space", we treat as resource limitation
    if return_code == 12:
        return SOLVABILITY_UNSOLVABLE_RESOURCE
    
    # Default: treat as resource limitation (conservative)
    return SOLVABILITY_UNSOLVABLE_RESOURCE


def run_planner(plan_path: str, search_algorithm: str = "astar_lmcut", timeout: int = 30) -> Tuple[bool, str, Dict[str, Any], str]:
    """
    Run the planner and return results with solvability classification.
    
    Returns:
        tuple: (success, message, planning_metrics, solvability)
            - success: bool, True if plan was found
            - message: str, human-readable message
            - planning_metrics: dict, metrics from planner output
            - solvability: str, one of SOLVABILITY_SOLVED, SOLVABILITY_UNSOLVABLE_STRUCTURAL, SOLVABILITY_UNSOLVABLE_RESOURCE
    """
    empty_metrics = {'expanded_nodes': None, 'space_used': None, 'solution_length': None, 
                     'search_time': None, 'grounding_time': None, 'total_time': None}
    try:
        # Clear any existing plan file to avoid using stale results
        if os.path.exists(plan_path):
            os.remove(plan_path)
            
        script_dir = os.path.dirname(__file__)
        call_script_path = os.path.join(script_dir, 'call_planner.py')
        
        result = subprocess.run([sys.executable, call_script_path,
                                 '--search', search_algorithm],
                                  capture_output=True, text=True, timeout=timeout)
        
        # Parse additional metrics from Fast Downward output
        planning_metrics = parse_fast_downward_output(result.stdout, result.stderr)
        
        # Check if plan file exists
        plan_exists = os.path.exists(plan_path)
        
        # Classify the result
        solvability = classify_planner_result(result.returncode, plan_exists, result.stdout, result.stderr)
        
        if solvability == SOLVABILITY_SOLVED:
            return True, "Plan found successfully", planning_metrics, solvability
        elif solvability == SOLVABILITY_UNSOLVABLE_STRUCTURAL:
            return False, f"Problem proved unsolvable (return code: {result.returncode})", planning_metrics, solvability
        else:
            return False, f"No solution found - resource limit (return code: {result.returncode})", planning_metrics, solvability
            
    except subprocess.TimeoutExpired:
        return False, "Planning timeout", empty_metrics, SOLVABILITY_UNSOLVABLE_RESOURCE
    except Exception as e:
        return False, f"Planning error: {str(e)}", empty_metrics, SOLVABILITY_UNSOLVABLE_RESOURCE


def parse_plan_file(plan_path: str) -> List[str]:
    """
    Parse a PDDL plan file and extract the sequence of action names.

    Args:
        plan_path: Path to the .plan file.

    Returns:
        List of sanitized action names (base activity names), excluding tau transitions.
    """
    if not os.path.exists(plan_path):
        return []
    
    actions = []
    try:
        with open(plan_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                
                if line.startswith('(') and line.endswith(')'):
                    parts = line[1:-1].split()
                    if parts:
                        action_name_full = parts[0]
                        
                        # Remove exec_ prefix if present
                        if action_name_full.startswith('exec_'):
                            action_name_full = action_name_full[5:]  # Remove 'exec_'
                        
                        # Remove various suffixes
                        action_name_base = action_name_full.split('_DETDUP')[0]
                        action_name_base = action_name_base.split(' ')[0]
                        action_name_base = re.sub(r'_v\d+$', '', action_name_base)
                        # Normalize/sanitize so we can process suffixes uniformly
                        action_name_base = sanitize_name(action_name_base)
                        
                        # Extract base activity name (remove measurement suffixes and fallback)
                        action_name_base = extract_base_activity_name(action_name_base)
                        action_name_base = sanitize_name(action_name_base)
                        
                        # Filter out tau (silent) transitions
                        if not action_name_base.startswith('tau_'):
                            actions.append(action_name_base)
    except Exception as e:
        print(f"Error parsing plan file: {e}")
        return []
    
    return actions


def extract_numerical_attributes(parser: Any) -> List[str]:
    """
    Extract a list of numerical attribute names from the parser.

    Args:
        parser: The XES parser instance.

    Returns:
        List of numerical attribute names.
    """
    numerical_attrs = []
    if hasattr(parser, 'attribute_categories'):
        for attr, category in parser.attribute_categories.items():
            if category == 'numerical':
                numerical_attrs.append(attr)
    return numerical_attrs


def get_goal_attribute_value(parser: Any, event: Dict[str, Any], target_attr: str) -> Optional[float]:
    """
    Extract the numeric value of a target attribute from an event.

    Args:
        parser: The XES parser instance.
        event: The event dictionary.
        target_attr: The name of the target attribute.

    Returns:
        The numeric value as a float, or None if not found or invalid.
    """
    trace_attr_key = find_attribute_key_in_event(event, target_attr)
    
    if trace_attr_key and trace_attr_key in event:
        raw_value = event[trace_attr_key]
        if raw_value is not None and not pd.isna(raw_value):
            try:
                return float(raw_value)
            except (ValueError, TypeError):
                pass
    
    return None


def create_evaluation_directories(script_dir: str) -> str:
    """
    Create the necessary directories for storing evaluation results.

    Args:
        script_dir: The directory containing the current script.

    Returns:
        The path to the created evaluation directory.
    """
    evaluation_dir = os.path.join(script_dir, '..', 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)
    return evaluation_dir


def generate_evaluation_filename(
    pddl_name: str, 
    log_coverage: float, 
    search_algorithm: str, 
    eval_type: str, 
    timestamp: Optional[str] = None
) -> str:
    """
    Generate a standardized filename for evaluation result CSVs.

    Args:
        pddl_name: Name of the PDDL domain.
        log_coverage: Coverage percentage used for filtering.
        search_algorithm: The planning algorithm used.
        eval_type: The type of evaluation (e.g., 'suffix', 'planning').
        timestamp: Optional specific timestamp to use.

    Returns:
        The generated filename string.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coverage_str = f"{log_coverage:.6f}".rstrip('0').rstrip('.')
    return f'{eval_type}_eval_{pddl_name}_cov{coverage_str}_{search_algorithm}_{timestamp}.csv'


def calculate_all_prefix_lengths(test_traces: List[List[str]]) -> List[int]:
    """
    Calculate all unique prefix lengths present in a set of test traces.

    Args:
        test_traces: List of traces, where each trace is a list of activity names.

    Returns:
        A sorted list of unique prefix lengths.
    """
    all_prefix_lengths = set()
    for trace in test_traces:
        sanitized_trace = [sanitize_name(event) for event in trace]
        if sanitized_trace:
            for i in range(1, len(sanitized_trace)):
                all_prefix_lengths.add(i)
    
    return sorted(list(all_prefix_lengths))


def prepare_test_data(parser: Any, max_traces: Optional[int] = None) -> Tuple[List[List[str]], List[List[Dict[str, Any]]]]:
    """
    Prepare test data from parser for evaluation.
    
    Args:
        parser: XES parser instance
        max_traces: Optional maximum number of traces to load from the log (None means all traces)
        
    Returns:
        tuple: (test_traces, test_traces_with_events) where:
            - test_traces: List of activity name sequences
            - test_traces_with_events: List of full event data for each trace
    """
    test_log = parser.test_df
    test_df = pm4py.convert_to_dataframe(test_log)
    
    test_traces = test_df.groupby('case:concept:name')['concept:name'].apply(list).tolist()
    
    test_traces_with_events = []
    for case_id in test_df['case:concept:name'].unique():
        case_events = test_df[test_df['case:concept:name'] == case_id].to_dict('records')
        test_traces_with_events.append(case_events)

    # Apply optional cap on number of traces to read
    if max_traces is not None and isinstance(max_traces, int) and max_traces > 0:
        if len(test_traces) > max_traces:
            test_traces = test_traces[:max_traces]
            test_traces_with_events = test_traces_with_events[:max_traces]

    return test_traces, test_traces_with_events


def read_state_file(file_path: str) -> List[str]:
    """
    Read a state file containing PDDL predicates.

    Args:
        file_path: Path to the state file.

    Returns:
        List of non-empty lines from the file.
    """
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: State file not found at {file_path}. Using empty state.")
        return []


def sanitize_name(name: str) -> str:
    """
    Sanitize an activity or attribute name for use in PDDL.

    Args:
        name: The original name string.

    Returns:
        A sanitized version of the name (lowercase, no special characters).
    """
    # Strip "case:" prefix for cleaner trace attribute names
    if name.lower().startswith("case:"):
        name = name[5:]  # Remove "case:" prefix
    return name.strip().lower().replace(" ", "_").replace(":", "_").replace("-", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")


def convert_interval_to_lte_gte(interval_str: str) -> str:
    """Convert interval notation like (-inf-6.15] to lte-6_15 format."""
    interval = interval_str.strip('()[]')
    if interval.startswith('-inf-'):
        upper_part = interval[5:]  # Remove '-inf-'
        upper_fmt = upper_part.replace('.', '_').replace('-', 'neg')
        return f"lte_{upper_fmt}"
    elif interval.endswith('-inf'):
        lower_part = interval[:-4]  # Remove '-inf'
        lower_fmt = lower_part.replace('.', '_').replace('-', 'neg')
        return f"gte_{lower_fmt}"
    else:
        parts = interval.split('-')
        if len(parts) == 2:
            lower, upper = parts
            lower_fmt = lower.replace('.', '_').replace('-', 'neg')
            upper_fmt = upper.replace('.', '_').replace('-', 'neg')
            return f"gte_{lower_fmt}_lte_{upper_fmt}"
    
    return interval_str


def discretize_value(attr: Union[str, None], value: Union[float, str, None], intervals: Optional[Dict[str, List[float]]] = None) -> str:
    """
    Discretize a numeric or interval value based on defined thresholds.

    Args:
        attr: The attribute name.
        value: The value to discretize.
        intervals: A dictionary mapping attribute names to lists of thresholds.

    Returns:
        The discretized value string (e.g., 'lte_10_0', 'gte_5_0_lte_10_0').
    """
    if (isinstance(value, str) and
            (
                    '-inf' in value or
                    (
                            '-' in value and
                            not value.replace('-', '').replace('.', '').replace('_', '').isalnum()
                    )
            )):
        return convert_interval_to_lte_gte(value)
    else:   # Numeric discretization
        if intervals is None or attr not in intervals or not intervals[attr] or value is None:
            return str(value)
        
        try:
            value = float(value)
            thresholds = sorted(intervals[attr])
            
            if value <= thresholds[0]:
                thresh_str = str(thresholds[0]).replace('.', '_').replace('-', 'neg')
                return f'lte_{thresh_str}'
            elif value >= thresholds[-1]:
                thresh_str = str(thresholds[-1]).replace('.', '_').replace('-', 'neg')
                return f'gte_{thresh_str}'
            else:
                for i in range(len(thresholds) - 1):
                    if thresholds[i] < value <= thresholds[i + 1]:
                        lower_str = str(thresholds[i]).replace('.', '_').replace('-', 'neg')
                        upper_str = str(thresholds[i + 1]).replace('.', '_').replace('-', 'neg')
                        return f'gte_{lower_str}_lte_{upper_str}'
            return str(value)
        except (ValueError, TypeError):
            return str(value)
