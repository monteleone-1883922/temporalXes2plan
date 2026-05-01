import os
import sys
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple, Set, Union
from xes_parser import Parser
from decision_mining import discover_all_decision_rules, get_attribute_domains
import utils


class RDDLDataCollector:
    """
    Collector for gathering data needed to generate RDDL (Relational Dynamic Dependency Language) models.

    Attributes:
        log_name (str): Name of the XES event log file.
        log_path (str): Full path to the XES log.
        parser (Parser): XES parser instance for structural analysis.
        decision_rules (List[Any]): Rules discovered via decision mining.
        intervals (Dict[str, List[float]]): Discretization intervals for numerical attributes.
        decision_samples (List[Dict[str, Any]]): Samples used for decision mining.
        rddl_data (Dict[str, Any]): The compiled RDDL data.
    """

    log_name: str
    log_path: str
    parser: Parser
    decision_rules: List[Any]
    intervals: Dict[str, List[float]]
    decision_samples: List[Dict[str, Any]]
    rddl_data: Dict[str, Any]
    def __init__(
        self, 
        log_name: str, 
        coverage_percentage: float = 1.0, 
        discovery_algorithm: str = 'inductive'
    ) -> None:
        """
        Initialize the RDDL data collector.
        
        Args:
            log_name: Name of the XES event log file
            coverage_percentage: Minimum cumulative coverage percentage for variant filtering
            discovery_algorithm: Algorithm to use for Petri net discovery
        """
        print("="*70)
        print("RDDL DATA COLLECTION STARTED")
        print("="*70)
        
        self.log_name = log_name
        self.log_path = os.path.join(os.path.dirname(__file__), '..', 'logs', log_name)
        
        # Initialize parser to get structure and probabilities
        print("\n[1/4] Initializing XES Parser...")
        self.parser = Parser(log_name, coverage_percentage, discovery_algorithm)
        
        # Run decision mining
        print("\n[2/4] Running Decision Mining...")
        self.decision_rules, self.intervals, self.decision_samples = discover_all_decision_rules(self.log_path)
        
        # Store intervals for numerical attribute discretization
        print(f"   Found intervals for {len(self.intervals)} attributes")
        
        # Collect all data
        print("\n[3/4] Collecting RDDL data...")
        self.rddl_data = self._collect_all_data()
        
        print("\n[4/4] RDDL data collection complete!")
        print("="*70)
    
    def _collect_all_data(self) -> Dict[str, Any]:
        """
        Collect all necessary data for RDDL generation.

        Returns:
            A dictionary containing structural, probability, and data-related information.
        """
        """Collect all necessary data for RDDL generation."""
        data = {
            'structure': self._get_structure(),
            'probabilities': self._get_probabilities(),
            'types_and_values': self._get_types_and_values(),
            'value_transition_probabilities': self._get_value_transition_probabilities(),
            'data_preconditions': self._get_data_preconditions()
        }
        return data
    
    def _get_structure(self) -> Dict[str, Any]:
        """
        Get the control flow structure from the parser.

        Returns:
            Dictionary with predecessors, parallels, and transition graph.
        """
        """Get the control flow structure from the parser."""
        return {
            'predecessors': self.parser.predecessors,
            'parallels': self.parser.parallels,
            'direct_transition_graph': self.parser.direct_transition_graph
        }
    
    def _get_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Get decision point probabilities from the parser.

        Returns:
            Map of decision points to activity branch probabilities.
        """
        """Get decision point probabilities from the parser."""
        # Convert decision_points_probabilities to a more readable format
        probabilities = {}
        for place, transitions_probs in self.parser.decision_points_probabilities.items():
            place_name = f"place_{id(place)}"
            probabilities[place_name] = {}
            for activity_name, prob in transitions_probs.items():
                # activity_name is already a string, no need to convert
                probabilities[place_name][activity_name] = prob
        
        return probabilities
    
    def _get_types_and_values(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all data attribute types and their possible values.
        Only includes data attributes used in decision mining (not activities).
        Returns discretized values for numerical attributes.
        """
        types_and_values = {}
        
        # Get attribute domains from decision mining
        attribute_domains = get_attribute_domains(self.decision_samples)
        
        # Only consider attributes that appear in decision mining samples
        attributes_in_decision_mining = set(attribute_domains.keys())
        
        # Process each attribute that was used in decision mining
        for attr in attributes_in_decision_mining:
            # Determine attribute type from parser if available, otherwise infer from values
            if attr in self.parser.attribute_categories:
                attr_type = self.parser.attribute_categories[attr]
            else:
                # Infer type from the values in attribute_domains
                values_in_domain = attribute_domains[attr]
                if values_in_domain == {True, False} or values_in_domain == {'true', 'false'}:
                    attr_type = 'boolean'
                elif any(v.startswith(('gte_', 'lte_', 'gt_', 'lt_')) for v in values_in_domain if isinstance(v, str)):
                    attr_type = 'numerical'
                else:
                    attr_type = 'categorical'
            
            if attr_type == 'boolean':
                # Boolean attributes always have true/false
                types_and_values[attr] = {
                    'type': 'boolean',
                    'values': ['true', 'false']
                }
            
            elif attr_type == 'categorical':
                # Get categorical values from the log
                values = set()
                for trace in self.parser.full_log:
                    for event in trace:
                        for event_attr, val in event.items():
                            if utils.sanitize_name(event_attr) == attr:
                                sanitized_val = utils.sanitize_name(str(val))
                                values.add(sanitized_val)
                
                types_and_values[attr] = {
                    'type': 'categorical',
                    'values': sorted(list(values))
                }
            
            elif attr_type == 'numerical':
                # Get discretized values from attribute_domains or parser
                values = set()
                
                # Check if we have values from decision mining
                if attr in attribute_domains:
                    values.update(attribute_domains[attr])
                
                # Also check parser's attribute_domains if available
                if hasattr(self.parser, 'attribute_domains') and attr in self.parser.attribute_domains:
                    values.update(self.parser.attribute_domains[attr])
                
                # If still empty, extract from the log and discretize
                if not values:
                    values = self._extract_discretized_numerical_values(attr)
                
                types_and_values[attr] = {
                    'type': 'numerical',
                    'values': sorted(list(values))
                }
        
        return types_and_values
    
    def _extract_discretized_numerical_values(self, attr: str) -> Set[str]:
        """
        Extract and discretize numerical values for an attribute from the log.

        Args:
            attr: The numerical attribute name.

        Returns:
            Set of discretized value strings.
        """
        """
        Extract and discretize numerical values for an attribute from the log.
        """
        values = set()
        raw_values = []
        
        # Collect raw values
        for trace in self.parser.full_log:
            for event in trace:
                for event_attr, val in event.items():
                    if utils.sanitize_name(event_attr) == attr:
                        try:
                            raw_values.append(float(val))
                        except (ValueError, TypeError):
                            pass
        
        if not raw_values:
            return values
        
        # Use intervals from decision mining if available
        if attr in self.intervals:
            thresholds = sorted(self.intervals[attr])
            
            # Create non-overlapping intervals based on thresholds
            for i, threshold in enumerate(thresholds):
                if i == 0:
                    interval_str = f"(-inf-{threshold}]"
                else:
                    interval_str = f"({thresholds[i-1]}-{threshold}]"
                discretized = utils.discretize_value(None, interval_str)
                values.add(discretized)
            
            # Add the last interval
            if thresholds:
                interval_str = f"({thresholds[-1]}-inf)"
                discretized = utils.discretize_value(None, interval_str)
                values.add(discretized)
        else:
            # Fallback: use quartiles for discretization
            quartiles = np.percentile(raw_values, [25, 50, 75])
            thresholds = sorted(set(quartiles))
            
            for i, threshold in enumerate(thresholds):
                if i == 0:
                    interval_str = f"(-inf-{threshold}]"
                else:
                    interval_str = f"({thresholds[i-1]}-{threshold}]"
                discretized = utils.discretize_value(None, interval_str)
                values.add(discretized)
            
            if thresholds:
                interval_str = f"({thresholds[-1]}-inf)"
                discretized = utils.discretize_value(None, interval_str)
                values.add(discretized)
        
        return values
    
    def _get_value_transition_probabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute the probability distribution of new values for each attribute.
        Only considers attributes used in decision mining.
        Counts when an attribute gets a new value (either it didn't exist before, 
        or it changed from a different value).
        
        For numerical attributes, we use the discretized values from the attribute_domains.
        Returns global probabilities across the entire log for discretized values.
        Ensures probabilities sum to 1 for each attribute.
        """
        value_transitions = {}
        
        # Get the known discretized values for each attribute from decision samples
        known_discretized_values = defaultdict(set)
        attributes_in_decision_mining = set()
        
        for sample in self.decision_samples:
            for precond in sample['preconditions']:
                if not precond.startswith('(completed ') and not precond.startswith('(enabled '):
                    # Parse the precondition to extract attribute and value
                    # Format: (attr value) or (not (attr))
                    inner = precond[1:-1]  # Remove outer parentheses
                    if inner.startswith('not ('):
                        # Boolean false
                        attr = inner[5:-1]
                        known_discretized_values[attr].add('false')
                        attributes_in_decision_mining.add(attr)
                    elif ' ' in inner:
                        parts = inner.split(' ', 1)
                        if len(parts) == 2:
                            attr, value = parts
                            known_discretized_values[attr].add(value)
                            attributes_in_decision_mining.add(attr)
                    else:
                        # Boolean true
                        attr = inner
                        known_discretized_values[attr].add('true')
                        attributes_in_decision_mining.add(attr)
        
        # Only process attributes that appear in decision mining
        for attr in attributes_in_decision_mining:
            if attr in self.parser.attribute_categories:
                attr_type = self.parser.attribute_categories[attr]
                
                # Track new value occurrences for this attribute
                value_counts = defaultdict(int)
                total_new_values = 0
                
                for trace in self.parser.full_log:
                    prev_value = None
                    
                    for event in trace:
                        current_value = None
                        
                        # Get current value for this attribute
                        for event_attr, val in event.items():
                            if utils.sanitize_name(event_attr) == attr:
                                if attr_type == 'boolean':
                                    current_value = 'true' if val else 'false'
                                elif attr_type == 'categorical':
                                    current_value = utils.sanitize_name(str(val))
                                elif attr_type == 'numerical':
                                    # For numerical, discretize using known values or compute discretization
                                    current_value = self._discretize_from_known_values(
                                        attr, val, known_discretized_values.get(attr, set())
                                    )
                                break
                        
                        # Count as new value if:
                        # 1. We have a current value, AND
                        # 2. Either there was no previous value, OR the value changed
                        if current_value is not None and current_value != prev_value:
                            value_counts[current_value] += 1
                            total_new_values += 1
                            prev_value = current_value
                        elif current_value is not None:
                            # Value exists but didn't change, keep tracking it
                            prev_value = current_value
                
                # Compute probabilities
                if total_new_values > 0:
                    probabilities = {}
                    for value, count in value_counts.items():
                        probabilities[value] = count / total_new_values
                    
                    # Normalize to ensure sum = 1 (accounting for floating point errors)
                    total_prob = sum(probabilities.values())
                    if total_prob > 0:
                        probabilities = {k: v/total_prob for k, v in probabilities.items()}
                    
                    value_transitions[attr] = {
                        'probabilities': probabilities,
                        'total_new_values': total_new_values
                    }
        
        return value_transitions
    
    def _discretize_from_known_values(self, attr: str, raw_value: Any, known_values: Set[str]) -> Optional[str]:
        """
        Discretize a numerical value based on a set of known discretized value ranges.

        Args:
            attr: The attribute name.
            raw_value: The raw numeric value.
            known_values: Set of known discretized value strings (e.g., 'lte_10_0').

        Returns:
            The matching discretized value string, or None if no match found.
        """
        """
        Discretize a numerical value by checking which known discretized value's range it falls into.
        Uses the reverse approach: parse the discretized value name to understand the range,
        then check if raw_value satisfies that range.
        If multiple ranges match, return the most specific (narrowest) one.
        """
        try:
            float_val = float(raw_value)
        except (ValueError, TypeError):
            return None
        
        if not known_values:
            return None
        
        # Find all matching discretized values
        matching_values = []
        for disc_val in known_values:
            if self._value_matches_discretized_range(float_val, disc_val):
                # Calculate the range width to determine specificity
                range_width = self._get_range_width(disc_val)
                matching_values.append((disc_val, range_width))
        
        if not matching_values:
            return None
        
        # Return the most specific (smallest range) match
        # Sort by range width (ascending) - smaller width = more specific
        matching_values.sort(key=lambda x: x[1])
        return matching_values[0][0]
    
    def _get_range_width(self, discretized_name: str) -> float:
        """
        Calculate the width of the range represented by a discretized value name.

        Args:
            discretized_name: The discretized value string.

        Returns:
            The numerical width of the range (float). Unbounded ranges return infinity.
        """
        """
        Calculate the width of the range represented by a discretized value.
        Smaller width = more specific range.
        Returns infinity for unbounded ranges.
        """
        parts = discretized_name.split('_')
        
        # Handle compound ranges (e.g., gte_28_5_lte_147_5)
        if 'gte' in discretized_name and 'lte' in discretized_name:
            lte_idx = None
            for i, part in enumerate(parts):
                if part == 'lte':
                    lte_idx = i
                    break
            
            if lte_idx and lte_idx >= 2:
                lower_parts = parts[1:lte_idx]
                upper_parts = parts[lte_idx+1:]
                
                try:
                    if len(lower_parts) >= 2 and len(upper_parts) >= 2:
                        lower = float(f"{lower_parts[0]}.{lower_parts[1]}")
                        upper = float(f"{upper_parts[0]}.{upper_parts[1]}")
                        return upper - lower
                except (ValueError, IndexError):
                    pass
        
        # Single-sided ranges have infinite width
        return float('inf')
    
    def _value_matches_discretized_range(self, value: float, discretized_name: str) -> bool:
        """
        Check if a numerical value matches the range represented by a discretized value name.

        Args:
            value: The numeric value to check.
            discretized_name: The discretized range string (e.g., 'gte_10_lte_20').

        Returns:
            True if the value falls within the range, False otherwise.
        """
        """
        Check if a numerical value matches the range represented by a discretized value name.
        E.g., "lte_28_5" means value <= 28.5
              "gte_147_5" means value >= 147.5
              "gte_28_5_lte_147_5" means 28.5 <= value <= 147.5
        """
        # Parse the discretized name to extract operators and thresholds
        parts = discretized_name.split('_')
        
        # Handle compound ranges (e.g., gte_28_5_lte_147_5)
        if 'gte' in discretized_name and 'lte' in discretized_name:
            # Find where lte starts
            lte_idx = None
            for i, part in enumerate(parts):
                if part == 'lte':
                    lte_idx = i
                    break
            
            if lte_idx and lte_idx >= 2:
                # Extract lower bound (gte part)
                lower_parts = parts[1:lte_idx]
                if len(lower_parts) >= 2:
                    try:
                        lower_bound = float(f"{lower_parts[0]}.{lower_parts[1]}")
                    except (ValueError, IndexError):
                        return False
                else:
                    return False
                
                # Extract upper bound (lte part)
                upper_parts = parts[lte_idx+1:]
                if len(upper_parts) >= 2:
                    try:
                        upper_bound = float(f"{upper_parts[0]}.{upper_parts[1]}")
                    except (ValueError, IndexError):
                        return False
                else:
                    return False
                
                return lower_bound <= value <= upper_bound
        
        # Handle single operator ranges
        elif discretized_name.startswith('gte_'):
            # Extract threshold
            threshold_parts = parts[1:]
            if len(threshold_parts) >= 2:
                try:
                    threshold = float(f"{threshold_parts[0]}.{threshold_parts[1]}")
                    return value >= threshold
                except (ValueError, IndexError):
                    return False
        
        elif discretized_name.startswith('lte_'):
            # Extract threshold
            threshold_parts = parts[1:]
            if len(threshold_parts) >= 2:
                try:
                    threshold = float(f"{threshold_parts[0]}.{threshold_parts[1]}")
                    return value <= threshold
                except (ValueError, IndexError):
                    return False
        
        elif discretized_name.startswith('gt_'):
            # Extract threshold
            threshold_parts = parts[1:]
            if len(threshold_parts) >= 2:
                try:
                    threshold = float(f"{threshold_parts[0]}.{threshold_parts[1]}")
                    return value > threshold
                except (ValueError, IndexError):
                    return False
        
        elif discretized_name.startswith('lt_'):
            # Extract threshold
            threshold_parts = parts[1:]
            if len(threshold_parts) >= 2:
                try:
                    threshold = float(f"{threshold_parts[0]}.{threshold_parts[1]}")
                    return value < threshold
                except (ValueError, IndexError):
                    return False
        
        return False
    
    def _get_data_preconditions(self) -> Dict[str, List[Dict[str, List[str]]]]:
        """
        Extract data preconditions coupled with control flow conditions.

        Returns:
            Map of activities to lists of variant-specific preconditions.
        """
        """
        Extract data preconditions coupled with control flow conditions.
        Uses decision mining samples to get data preconditions for each activity variant.
        """
        data_preconditions = defaultdict(list)
        
        for sample in self.decision_samples:
            activity = sample['activity']
            preconditions = sample['preconditions']
            
            # Separate control flow and data preconditions
            control_flow = set()
            data_conds = set()
            
            for precond in preconditions:
                if precond.startswith('(completed ') or precond.startswith('(enabled '):
                    control_flow.add(precond)
                else:
                    data_conds.add(precond)
            
            # Store the combination of control flow and data preconditions
            data_preconditions[activity].append({
                'control_flow': sorted(list(control_flow)),
                'data_preconditions': sorted(list(data_conds))
            })
        
        return dict(data_preconditions)
    
    def get_rddl_data(self) -> Dict[str, Any]:
        """
        Return the collected RDDL data.

        Returns:
            The compiled RDDL data dictionary.
        """
        """Return the collected RDDL data."""
        return self.rddl_data
    
    def print_summary(self) -> None:
        """Print a detailed summary of the collected RDDL data to stdout."""
        """Print detailed statistics in key-value format, one per line."""
        print("\n" + "="*70)
        print("RDDL DATA COLLECTION RESULTS")
        print("="*70)
        
        # 1. STRUCTURE
        print("\n[STRUCTURE]")
        print(f"total_activities: {len(self.rddl_data['structure']['predecessors'])}")
        print(f"parallel_patterns: {len(self.rddl_data['structure']['parallels'])}")
        print(f"direct_transitions: {len(self.rddl_data['structure']['direct_transition_graph'])}")
        
        # 2. DECISION POINT PROBABILITIES
        print("\n[DECISION_POINT_PROBABILITIES]")
        print(f"total_decision_points: {len(self.rddl_data['probabilities'])}")
        for place, probs in self.rddl_data['probabilities'].items():
            for activity, prob in probs.items():
                print(f"{place}_{activity}: {prob:.6f}")
        
        # 3. TYPES AND VALUES
        print("\n[TYPES_AND_VALUES]")
        for attr, info in sorted(self.rddl_data['types_and_values'].items()):
            print(f"{attr}_type: {info['type']}")
            print(f"{attr}_num_values: {len(info['values'])}")
            for val in info['values']:
                print(f"{attr}_value: {val}")
        
        # 4. VALUE PROBABILITIES - show ALL values including those with 0 probability
        print("\n[VALUE_PROBABILITIES]")
        for attr in sorted(self.rddl_data['types_and_values'].keys()):
            if attr in self.rddl_data['value_transition_probabilities']:
                info = self.rddl_data['value_transition_probabilities'][attr]
                print(f"{attr}_total_new_values: {info['total_new_values']}")
                
                # Get all possible values for this attribute
                all_values = self.rddl_data['types_and_values'][attr]['values']
                
                # Print probability for each value (0.0 if not in probabilities dict)
                for val in all_values:
                    prob = info['probabilities'].get(val, 0.0)
                    print(f"{attr}_{val}_probability: {prob:.6f}")
                
                # Verify sum
                total_prob = sum(info['probabilities'].values())
                print(f"{attr}_probability_sum: {total_prob:.6f}")
            else:
                print(f"{attr}_total_new_values: 0")
                print(f"{attr}_probability_sum: 0.000000")
        
        # 5. DATA PRECONDITIONS
        print("\n[DATA_PRECONDITIONS]")
        print(f"total_activities_with_preconditions: {len(self.rddl_data['data_preconditions'])}")
        for activity in sorted(self.rddl_data['data_preconditions'].keys()):
            variants = self.rddl_data['data_preconditions'][activity]
            print(f"{activity}_num_variants: {len(variants)}")
            for i, variant in enumerate(variants):
                variant_id = f"{activity}_variant_{i}"
                print(f"{variant_id}_control_flow: {', '.join(variant['control_flow'])}")
                print(f"{variant_id}_data_preconditions: {', '.join(variant['data_preconditions'])}")
        
        print("\n" + "="*70)


def main() -> Dict[str, Any]:
    """
    Main entry point for running the RDDL data collector from the command line.

    Returns:
        The collected RDDL data.
    """
    if len(sys.argv) < 2:
        print("Usage: python xes_parser_rddl.py <log_name> [coverage_percentage] [discovery_algorithm]")
        print("Example: python xes_parser_rddl.py sepsis.xes 0.8 inductive")
        sys.exit(1)
    
    log_name = sys.argv[1]
    coverage_percentage = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    discovery_algorithm = sys.argv[3] if len(sys.argv) > 3 else 'inductive'
    
    collector = RDDLDataCollector(log_name, coverage_percentage, discovery_algorithm)
    collector.print_summary()
    rddl_data = collector.get_rddl_data()
    
    print("\nRDDL data collected successfully!")
    print("Access the data using: collector.get_rddl_data()")
    print(rddl_data)
    return rddl_data


if __name__ == "__main__":
    main()
