import os
import random
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple, Set, Union
import pm4py
import utils
from xes_parser import Parser


class Encoder:
    """
    Encoder for transforming process models and event logs into PDDL domains and problems.

    Attributes:
        parser (Parser): XES parser instance containing process data.
        domain_name (str): Name of the PDDL domain to generate.
        init (Optional[List[str]]): Custom initial state predicates.
        goal (Optional[List[str]]): Custom goal state predicates.
        min_confidence (float): Minimum confidence for including attribute relationships.
        activities (Set[str]): Set of activities in the process.
        attribute_categories (Dict[str, str]): Mapping of attributes to their categories.
        tau_activities (Set[str]): Set of silent (tau) activities.
        decision_points (Dict[str, Dict[str, float]]): Branch probabilities at decision points.
        decision_preconditions (Dict[str, List[Set[str]]]): Data-driven preconditions for activities.
        decision_attributes (Set[str]): Attributes involved in decision-making.
        parallel_activities (Dict[str, List[str]]): Parallel activity relationships.
        minimal_preconditions (bool): Whether to use minimal preconditions for planning.
    """

    parser: Parser
    domain_name: str
    init: Optional[List[str]]
    goal: Optional[List[str]]
    min_confidence: float
    activities: Set[str]
    attribute_categories: Dict[str, str]
    _value_mappings: Dict[str, Any]
    attr_activity_relationships: Dict[str, Any]
    activity_attr_effects: Dict[str, Any]
    tau_activities: Set[str]
    decision_points: Dict[str, Dict[str, float]]
    decision_preconditions: Dict[str, List[Set[str]]]
    decision_attributes: Set[str]
    parallel_activities: Dict[str, List[str]]
    parallel_activity_map: Dict[str, Set[str]]
    split_actions: Dict[str, Any]
    effect_alternatives: Dict[str, Any]
    minimal_preconditions: bool
    def __init__(
        self, 
        parser: Parser, 
        domain_name: str = "process_domain", 
        min_confidence: float = 0.1, 
        init: Optional[List[str]] = None, 
        goal: Optional[List[str]] = None, 
        minimal_preconditions: bool = False
    ) -> None:
        self.parser = parser
        self.domain_name = domain_name
        self.init = init
        self.goal = goal
        self.min_confidence = min_confidence
        self.activities = self.parser.activities
        self.attribute_categories = parser.attribute_categories
        self._value_mappings = {}
        self.attr_activity_relationships = parser.discover_attribute_activity_relationships()
        self.activity_attr_effects = parser.discover_activity_attribute_effects()
        self.tau_activities = {activity for activity in self.activities if self.is_tau_activity(activity)}
        self.decision_points = parser.decision_points_probabilities
        self.decision_preconditions = defaultdict(list)
        for sample in getattr(parser, 'decision_samples', []):
            activity = sample['activity'].replace('exec_', '')
            preconds = set(p.replace('exec_', '') for p in sample['preconditions'])
            self.decision_preconditions[activity].append(preconds)
        self.decision_attributes = set()
        for preconds_list in self.decision_preconditions.values():
            for preconds in preconds_list:
                for p in preconds:
                    if p.startswith('(') and p.endswith(')'):
                        inner = p[1:-1]
                        parts = inner.split()
                        if len(parts) == 1:
                            attr = parts[0]
                            if attr in self.attribute_categories:
                                self.decision_attributes.add(attr)
                        elif len(parts) == 2:
                            attr, val = parts
                            if attr in self.attribute_categories:
                                self.decision_attributes.add(attr)
        self.parallel_activities = parser.parallels
        self.parallel_activity_map = self._build_parallel_activity_map()
        self._process_decision_points()
        self.split_actions = {}
        self.effect_alternatives = {}
        # When True, preconditions will only require that an action is enabled
        # and attribute-based conditions; completed predecessor preconditions
        # will be skipped. This is useful for suffix-only planning where initial
        # state encodes possible successors but we don't want to enforce
        # completed predecessors in preconditions.
        self.minimal_preconditions = minimal_preconditions
    

    def is_tau_activity(self, activity_name: str) -> bool:
        """Check if an activity is a silent (tau) transition."""
        return activity_name.startswith('tau_')
    

    def _build_parallel_activity_map(self) -> Dict[str, Set[str]]:
        """
        Build a map of activities that can be executed in parallel with each other and
        determine which activities should not be considered as preconditions.
        """
        parallel_map = defaultdict(set)
        for source_activity, parallel_targets in self.parallel_activities.items():
            for activity1 in parallel_targets:
                for activity2 in parallel_targets:
                    if activity1 != activity2:
                        parallel_map[activity1].add(activity2)
                        parallel_map[activity2].add(activity1)
                parallel_map[source_activity].add(activity1)
                parallel_map[activity1].add(source_activity)
        
        return parallel_map
    
    
    def _generate_pddl_predicate_definitions(self) -> str:
        """Generate the predicates section for the PDDL domain."""
        predicates = [
            "    (completed ?a - activity)",
            "    (enabled ?a - activity)"
        ]
        relevant_attrs = set()
        
        # Include attributes that influence or are influenced by actions
        for attr in self.attr_activity_relationships:
            relevant_attrs.add(attr)
        
        # Include attributes affected by actions
        for activity in self.activities:
            variants = self._generate_measurement_action_variants(activity)
            if variants:
                sanitized_action = activity.lower()
                matching_attribute = next((attr for attr in self.attribute_categories 
                                        if attr.lower() == sanitized_action), None)
                if matching_attribute:
                    relevant_attrs.add(matching_attribute)
        
        # Include attributes from decision preconditions
        relevant_attrs |= self.decision_attributes
        
        # Include attributes from decision samples
        for sample in getattr(self.parser, 'decision_samples', []):
            for precond in sample.get('preconditions', []):
                if isinstance(precond, str) and precond.startswith('(') and precond.endswith(')'):
                    inner = precond[1:-1]
                    parts = inner.split()
                    if len(parts) >= 1:
                        attr = parts[0]
                        relevant_attrs.add(attr)
        
        for attr, category in self.attribute_categories.items():
            if attr in relevant_attrs:
                predicate = self._create_attribute_predicate(attr, category)
                if predicate:
                    predicates.append(predicate)

        return "  (:predicates\n" + "\n".join(predicates) + "\n  )"
    
    def _create_attribute_predicate(self, attr: str, category: str) -> Optional[str]:
        """Create a single attribute predicate based on its category."""
        if category == 'boolean':
            return f"    ({attr})"
        elif category == 'numerical':
            return f"    ({attr} ?v - {attr}_type)"
        else:  # categorical
            return f"    ({attr} ?v - {attr}_type)"
    

    def _generate_pddl_type_definitions(self) -> str:
        """Generate PDDL type definitions based on the predicates that will be used."""
        # Collect attributes that will have predicates
        relevant_attrs = set()
        
        # Include attributes that influence or are influenced by actions
        for attr in self.attr_activity_relationships:
            relevant_attrs.add(attr)
        
        # Include attributes affected by actions (measurement actions)
        for activity in self.activities:
            variants = self._generate_measurement_action_variants(activity)
            if variants:
                sanitized_action = activity.lower()
                matching_attribute = next((attr for attr in self.attribute_categories 
                                        if attr.lower() == sanitized_action), None)
                if matching_attribute:
                    relevant_attrs.add(matching_attribute)
        
        # Include attributes from decision preconditions
        relevant_attrs |= self.decision_attributes
        
        # Include attributes from decision samples
        for sample in getattr(self.parser, 'decision_samples', []):
            for precond in sample.get('preconditions', []):
                if isinstance(precond, str) and precond.startswith('(') and precond.endswith(')'):
                    inner = precond[1:-1]
                    parts = inner.split()
                    if len(parts) >= 1:
                        attr = parts[0]
                        relevant_attrs.add(attr)
        types = ["activity"]
        for attr in relevant_attrs:
            category = self.attribute_categories.get(attr)
            if category in ['numerical', 'categorical']:
                types.append(f"{attr}_type")
        
        return (
            "  (:types\n"
            f"    {' '.join(types)}\n"
            "  )"
        )
    

    def _collect_preconditions(self, action_name: str, variant_conditions: Optional[List[str]] = None) -> List[str]:
        """
        Collect preconditions for an action using the direct transition graph.
        
        Args:
            action_name: The name of the activity
            variant_conditions: Optional list of specific conditions to include for this variant
        """
        preconditions = []

        # Always include enabled marker for the action
        preconditions.append(f"(enabled {action_name})")

        # Tau activities can also have action preconditions now
        # if self.is_tau_activity(action_name):
        #     return preconditions
        if action_name in self.parser.start_activities:
            return preconditions
            
        # Find direct predecessors using the direct transition graph
        direct_predecessors = []
        for pred, successors in self.parser.direct_transition_graph.items():
            if action_name in successors:
                direct_predecessors.append(pred)
        
        if not direct_predecessors:
            return preconditions
            
        # By default, include completed predecessor(s) as preconditions to follow
        # structural constraints. However, in minimal_preconditions mode we
        # skip adding predecessor completion requirements and instead rely on
        # the :init enabled markers being set by the evaluation (prefix).
        if not getattr(self, 'minimal_preconditions', False):
            # For now, require at least one direct predecessor to be completed
            # This represents OR logic at decision points
            if len(direct_predecessors) == 1:
                preconditions.append(f"(completed {direct_predecessors[0]})")
            else:
                # For multiple predecessors, we need OR logic
                # Since PDDL doesn't have OR in preconditions directly, we'll use the first one as default
                # This could be improved with action variants for different paths
                preconditions.append(f"(completed {direct_predecessors[0]})")

        # Add attribute-based conditions (keep existing logic)
        attr_values = defaultdict(set)
        for attr, values in self.attr_activity_relationships.items():
            for val, activities in values.items():
                if action_name in activities:
                    stats = activities[action_name]
                    if stats['probability'] >= self.min_confidence:
                        if self.attribute_categories.get(attr) == 'boolean':
                            condition = f"({attr})" if val.lower() == 'true' else f"(not ({attr}))"
                            attr_values[attr].add(condition)
                        else:
                            if attr in self._get_attribute_params(action_name) and action_name not in self.attribute_categories:
                                attr_values[attr].add(f"({attr} ?v)")
        
        for attr, values in attr_values.items():
            values_list = list(values)
            if len(values_list) == 1:
                preconditions.append(values_list[0])
            else:
                if variant_conditions:
                    for value_cond in values_list:
                        if value_cond in variant_conditions:
                            preconditions.append(value_cond)
                            break
                    else:
                        preconditions.append(values_list[0])
                else:
                    preconditions.append(values_list[0])
        
        unique_preconditions = []
        seen = set()
        for precond in preconditions:
            if precond not in seen:
                unique_preconditions.append(precond)
                seen.add(precond)
        
        return unique_preconditions


    def _analyze_input_places(self, action_name: str) -> Dict[str, Any]:
        """
        Analyze the input places of a transition (activity) in the Petri net
        to determine the correct logical relationships for its preconditions.
        
        Returns:
            dict: A dictionary with 'AND' and 'OR' keys, where:
                  - 'AND' contains activities that must ALL be completed
                  - 'OR' contains groups of activities where ANY ONE in each group must be completed
        """
        transition = None
        for t in self.parser.transitions:
            if t.label is not None and utils.sanitize_name(t.label) == action_name:
                transition = t
                break
        if not transition:
            return {}
        
        input_places = []
        for arc in self.parser.edges:
            if isinstance(arc.source, pm4py.objects.petri_net.obj.PetriNet.Place) and \
            arc.target == transition:
                input_places.append(arc.source)
        or_groups = [] # Sort input places into OR groups (places that represent decision points) and AND inputs (places that must all have tokens)
        and_inputs = set()
        for place in input_places:
            incoming_transitions = []
            for arc in self.parser.edges:
                if isinstance(arc.source, pm4py.objects.petri_net.obj.PetriNet.Transition) and \
                arc.target == place and arc.source.label is not None:
                    incoming_transitions.append(arc.source)
            incoming_transitions = [t for t in incoming_transitions if t.label is not None]
            
            if len(incoming_transitions) > 1: # If this place is a decision point output (multiple transitions put tokens here), then the incoming activities form an OR group
                or_group = {utils.sanitize_name(t.label) for t in incoming_transitions}
                if or_group:  # Only add non-empty groups
                    or_groups.append(or_group)
            elif len(incoming_transitions) == 1: # Otherwise, this is an AND input (a single transition puts tokens here)
                for t in incoming_transitions:
                    if t.label is not None:
                        and_inputs.add(utils.sanitize_name(t.label))
        
        result = {}
        if and_inputs:
            result["AND"] = list(and_inputs)
        if or_groups:
            result["OR"] = or_groups
        
        return result


    def _collect_effects(self, action_name: str) -> List[str]:
        """
        Collect effects for an action with additional effects to enable subsequent activities.
        """
        effects = []
        seen_effects = set()
        
        # Initialize alternatives tracking for this action
        if action_name not in self.effect_alternatives:
            self.effect_alternatives[action_name] = []
        
        # Add basic completion effect
        self._add_completion_effect(action_name, effects, seen_effects)
        

        if self.is_tau_activity(action_name):
            self._handle_successor_enabling(action_name, effects, seen_effects)
            self._add_disable_effect(action_name, effects, seen_effects)
            return effects
        if action_name in self.parser.start_activities:
            return self._handle_start_activity_effects(action_name, effects, seen_effects)
        boolean_effects = self._handle_attribute_effects(action_name, effects, seen_effects)
        
        # Handle successor enabling
        self._handle_successor_enabling(action_name, effects, seen_effects)
        
        # Disable current action
        self._add_disable_effect(action_name, effects, seen_effects)
        
        return effects

    def _add_completion_effect(self, action_name: str, effects: List[str], seen_effects: Set[str]) -> None:
        """Add the basic completion effect for an action."""
        completion_effect = f"(completed {action_name})"
        effects.append(completion_effect)
        seen_effects.add(completion_effect)

    def _handle_start_activity_effects(self, action_name: str, effects: List[str], seen_effects: Set[str]) -> List[str]:
        """Handle effects specific to start activities."""
        successors = self._get_successors(action_name)
        for succ in successors:
            enable_effect = f"(enabled {succ})"
            if enable_effect not in seen_effects:
                effects.append(enable_effect)
                seen_effects.add(enable_effect)
        
        self._add_disable_effect(action_name, effects, seen_effects)
        return effects

    def _add_disable_effect(self, action_name: str, effects: List[str], seen_effects: Set[str]) -> None:
        """Add effect to disable the current action."""
        disable_current = f"(not (enabled {action_name}))"
        if disable_current not in seen_effects:
            effects.append(disable_current)
            seen_effects.add(disable_current)

    def _handle_attribute_effects(self, action_name: str, effects: List[str], seen_effects: Set[str]) -> Dict[str, bool]:
        """Handle attribute-related effects for an action."""
        boolean_effects = {}
        sanitized_action = action_name.lower()
        matching_attribute = next((attr for attr in self.attribute_categories 
                                if attr.lower() == sanitized_action), None)
        
        if matching_attribute:
            boolean_effects = self._process_matching_attribute(
                action_name, matching_attribute, effects, seen_effects
            )
        
        # Process activity attribute effects
        self._process_activity_attribute_effects(action_name, effects, seen_effects, boolean_effects)
        
        return boolean_effects

    def _process_matching_attribute(self, action_name: str, matching_attribute: str, effects: List[str], seen_effects: Set[str]) -> Dict[str, bool]:
        """Process effects for attributes that match the action name."""
        boolean_effects = {}
        attr_category = self.attribute_categories[matching_attribute]
        
        if attr_category == 'boolean':
            bool_alternatives = [f"({matching_attribute})", f"(not ({matching_attribute}))"]
            self.effect_alternatives[action_name].append(bool_alternatives)
            boolean_effects[matching_attribute] = True
        else:
            self._handle_non_boolean_matching_attribute(
                action_name, matching_attribute, attr_category, effects, seen_effects
            )
        
        return boolean_effects

    def _handle_non_boolean_matching_attribute(self, action_name: str, matching_attribute: str, attr_category: str, effects: List[str], seen_effects: Set[str]) -> None:
        """Handle non-boolean attributes that match the action name."""
        num_type = f"{matching_attribute}_type"
        remove_effect = (
            f"(forall (?old - {num_type})\n"
            f"    (when ({matching_attribute} ?old)\n"
            f"      (not ({matching_attribute} ?old))\n"
            f"    )\n"
            f"  )"
        )
        
        if remove_effect not in seen_effects:
            effects.append(remove_effect)
            seen_effects.add(remove_effect)
        
        self._add_attribute_alternatives_and_effects(
            action_name, matching_attribute, attr_category, effects, seen_effects
        )

    def _add_attribute_alternatives_and_effects(self, action_name: str, attr: str, attr_category: str, effects: List[str], seen_effects: Set[str]) -> None:
        """Add attribute alternatives and effects."""
        possible_values = self._get_possible_attribute_values(attr)
        
        if possible_values:
            attr_alternatives = [f"({attr} {val})" for val in sorted(possible_values)]
            self.effect_alternatives[action_name].append(attr_alternatives)
            
            # Use most common value as default for base action
            param_index = self._get_param_index_for_attribute(action_name, attr)
            param_name = f"?v{param_index}"
            set_effect = f"({attr} {param_name})"
            if set_effect not in seen_effects:
                effects.append(set_effect)
                seen_effects.add(set_effect)

    def _get_possible_attribute_values(self, attr: str) -> Optional[Set[str]]:
        decision_values = self.parser.attribute_domains.get(attr, set())
        if decision_values:
            if attr in self._value_mappings:
                decision_values = {self._value_mappings[attr].get(val, val) for val in decision_values}
            return decision_values
        
    def _get_param_index_for_attribute(self, action_name: str, attr: str) -> int:
        """Get the parameter index for a given attribute in an action."""
        param_index = 1
        for action_attr in self._get_attribute_params(action_name):
            if action_attr == attr:
                break
            param_index += 1
        return param_index

    def _process_activity_attribute_effects(self, action_name: str, effects: List[str], seen_effects: Set[str], boolean_effects: Dict[str, bool]) -> None:
        """Process activity attribute effects from the discovered relationships."""
        if action_name not in self.activity_attr_effects:
            return
        
        conditional_effects = []
        
        # Process boolean conditional effects
        self._process_boolean_conditional_effects(conditional_effects, boolean_effects, effects, seen_effects)
        
        # Add boolean effects
        self._add_boolean_effects(boolean_effects, effects, seen_effects)

    def _handle_successor_enabling(self, action_name: str, effects: List[str], seen_effects: Set[str]) -> None:
        """Handle enabling of successor activities."""
        successors = self._get_successors(action_name)
        decision_point_probs = self._get_decision_point_for_activity(action_name)
        
        if decision_point_probs:
            self._handle_decision_point_successors(action_name, decision_point_probs, successors, effects, seen_effects)
        else:
            self._enable_all_successors(successors, effects, seen_effects)

    def _handle_decision_point_successors(self, action_name: str, decision_point_probs: Dict[str, float], successors: List[str], effects: List[str], seen_effects: Set[str]) -> None:
        """Handle successor enabling for decision points."""
        valid_successors = {succ: prob for succ, prob in decision_point_probs.items() 
                           if prob >= 0.01 and succ in successors}
        
        if valid_successors:
            for key in self.decision_points_conditions:
                self._process_decision_point_conditions(key, action_name, effects)

    def _enable_all_successors(self, successors: List[str], effects: List[str], seen_effects: Set[str]) -> None:
        """Enable all successor activities."""
        for succ in successors:
            enable_effect = f"(enabled {succ})"
            if enable_effect not in seen_effects:
                effects.append(enable_effect)
                seen_effects.add(enable_effect)

    def _process_decision_point_conditions(self, key: Any, action_name: str, effects: List[str]) -> None:
        """Process conditions for decision points."""
        attributes_pre_xor = self.decision_points_conditions[key]
        for attribute in attributes_pre_xor:
            values_attr_pre_xor = self.decision_points_conditions[key][attribute]
            for val in values_attr_pre_xor:
                transitions_val_pre_xor = self.decision_points_conditions[key][attribute][val]
                if len(transitions_val_pre_xor.keys()) > 1:
                    self._handle_multiple_transitions(action_name, attribute, val, transitions_val_pre_xor, effects)
                else:
                    self._handle_single_transition(attribute, val, transitions_val_pre_xor, effects)

    def _handle_multiple_transitions(self, action_name: str, attribute: str, val: Any, transitions_val_pre_xor: Dict[str, Any], effects: List[str]) -> None:
        """Handle multiple transitions in decision points."""
        decision_alternatives = []
        for transition in transitions_val_pre_xor.keys():
            decision_alternatives.append(f"(enabled {transition})")
        
        conditional_alt_obj = {
            'condition': f"({attribute} {val})",
            'alternatives': decision_alternatives
        }
        self.effect_alternatives[action_name].append([conditional_alt_obj])
        
        # Use first transition for base action
        first_transition = list(transitions_val_pre_xor.keys())[0]
        effects.append(f"(when ({attribute} {val})\n        (enabled {first_transition}))")

    def _handle_single_transition(self, attribute: str, val: Any, transitions_val_pre_xor: Dict[str, Any], effects: List[str]) -> None:
        """Handle single transition in decision points."""
        for transition in transitions_val_pre_xor.keys():
            effects.append(f"(when ({attribute} {val})\n        (enabled {transition}))")

    def _process_boolean_conditional_effects(self, conditional_effects: List[Dict[str, Any]], boolean_effects: Dict[str, bool], effects: List[str], seen_effects: Set[str]) -> None:
        """Process conditional effects for boolean attributes."""
        conditional_by_attr = defaultdict(list)
        for effect in conditional_effects:
            conditional_by_attr[effect['attribute']].append(effect)
        
        for attr, cond_effects in conditional_by_attr.items():
            if attr in boolean_effects:
                continue
            for cond_effect in cond_effects:
                effect_str = f"({attr})" if cond_effect['value'] else f"(not ({attr}))"
                when_effect = f"(when {cond_effect['condition']}\n        {effect_str}\n      )"
                if when_effect not in seen_effects:
                    effects.append(when_effect)
                    seen_effects.add(when_effect)

    def _add_boolean_effects(self, boolean_effects: Dict[str, bool], effects: List[str], seen_effects: Set[str]) -> None:
        """Add boolean effects to the effects list."""
        for attr, is_true in boolean_effects.items():
            effect_str = f"({attr})" if is_true else f"(not ({attr}))"
            if effect_str not in seen_effects:
                effects.append(effect_str)
                seen_effects.add(effect_str)

    def _generate_measurement_variant_effects(self, action_name: str, variant: Dict[str, Any]) -> List[str]:
        """
        Generate effects for a specific measurement outcome variant.
        This avoids PDDL semantic issues by having deterministic effects
        based on the chosen measurement outcome.
        """
        effects = []
        seen_effects = set()
        
        # Basic completion and enabling effects
        completion_effect = f"(completed {action_name})"
        effects.append(completion_effect)
        seen_effects.add(completion_effect)
        
        attribute = variant['attribute']
        outcome_value = variant['outcome_value']
        attr_category = self.attribute_categories[attribute]
        
        # Remove old value and set new value
        num_type = f"{attribute}_type"
        remove_effect = (
            f"(forall (?old - {num_type})\n"
            f"       (when ({attribute} ?old)\n"
            f"         (not ({attribute} ?old))))"
        )
        effects.append(remove_effect)
        seen_effects.add(remove_effect)
        
        # Set the specific outcome value
        set_effect = f"({attribute} {outcome_value})"
        effects.append(set_effect)
        seen_effects.add(set_effect)
        
        # Handle enabling of successor activities based on the measurement outcome
        successors = self._get_successors(action_name)
        
        # For measurement activities, enable successors based on the specific outcome
        # This replaces the conditional logic in the original approach
        for succ in successors:
            enable_effect = f"(enabled {succ})"
            if enable_effect not in seen_effects:
                effects.append(enable_effect)
                seen_effects.add(enable_effect)
        
        # Disable current action
        disable_current = f"(not (enabled {action_name}))"
        if disable_current not in seen_effects:
            effects.append(disable_current)
            seen_effects.add(disable_current)
        
        return effects

    def _generate_measurement_action_variants(self, action_name: str) -> List[Dict[str, Any]]:
        """
        Generate separate actions for each measurement outcome to avoid
        PDDL semantic issues with simultaneous effects.
        """
        variants = []
        
        # Check if this action corresponds to a measurement activity
        sanitized_action = action_name.lower()
        matching_attribute = next((attr for attr in self.attribute_categories 
                                if attr.lower() == sanitized_action), None)
        
        if not matching_attribute:
            return []
            
        attr_category = self.attribute_categories[matching_attribute]
        
        # Only create variants for non-boolean attributes (measurements)
        if attr_category == 'boolean':
            return []
            
        # Get possible values for this attribute
        possible_values = self.parser.attribute_domains.get(matching_attribute, set())
        if matching_attribute in self._value_mappings:
            possible_values = {self._value_mappings[matching_attribute].get(val, val) for val in possible_values}
        
        if not possible_values:
            return []
            
        # Generate a separate action for each possible measurement outcome
        for value in sorted(possible_values):
            variant_name = f"{action_name}-{value}"
            variant = {
                'name': variant_name,
                'attribute': matching_attribute,
                'outcome_value': value,
                'base_action': action_name
            }
            variants.append(variant)
        
        return self._deduplicate_variants(variants, action_name)

    def _get_attribute_params(self, action_name: str) -> Set[str]:
        """Determine which attributes should be parameters for an action."""
        param_attributes = set()
        
        # Check if this is a measurement action variant (like crp_high, leucocytes_low)
        # These should NOT have parameters as they represent specific observations
        discretized_suffixes = ['_very_low', '_low', '_medium', '_high', '_true', '_false']
        if any(action_name.endswith(suffix) for suffix in discretized_suffixes):
            return param_attributes  # Return empty set - no parameters needed
        
        # Also check for categorical value suffixes (for attributes like diagnose)
        base_name = action_name
        for attr in self.attribute_categories:
            if action_name.startswith(attr + '_'):
                # This is likely a measurement variant like crp_high or diagnose_a
                return param_attributes  # No parameters needed
        
        # Original logic for other types of actions that might need parameters
        sanitized_action = action_name.lower()
        for attr in self.attribute_categories:
            # Only add parameters for actions that manipulate attribute values generically,
            # not for specific measurement actions
            if attr.lower() == sanitized_action and self.attribute_categories[attr] != 'boolean':
                # Skip if this looks like a measurement action (has variants)
                if not any(f"{attr}_{suffix}" in [a for a in self.activities] for suffix in ['low', 'medium', 'high', 'very_low']):
                    param_attributes.add(attr)
                    return param_attributes
        
        # Check for actions that introduce new attribute values
        if action_name in self.activity_attr_effects:
            for attr, from_values in self.activity_attr_effects[action_name].items():
                if 'NEW' in from_values and self.attribute_categories.get(attr) != 'boolean':
                    # Only add parameter if this is not a measurement action setting specific values
                    if not any(action_name.endswith(suffix) for suffix in discretized_suffixes):
                        if not any(attr in attrs and any(from_val != 'NEW' and to_vals for from_val, to_vals in attrs[attr].items())
                                  for act_name, attrs in self.activity_attr_effects.items() if attr in attrs):
                            param_attributes.add(attr)
        
        return param_attributes


    def _get_decision_point_for_activity(self, action_name: str) -> Optional[Dict[str, float]]:
        if action_name in self.processed_decision_points:
            return self.processed_decision_points[action_name]
        return None


    def _find_or_preconditions(self, action_name: str) -> List[List[str]]:
        """
        Find all OR conditions in the preconditions of an action.
        Returns a list of lists, where each inner list represents one OR condition.
        """
        # If minimal preconditions are enabled, avoid generating OR-condition
        # variants - preconditions will rely on :init enabled markers.
        if getattr(self, 'minimal_preconditions', False):
            return []
        or_conditions = []
        input_groups = self._analyze_input_places(action_name)       # Check activity OR conditions from Petri net structure
        if input_groups and "OR" in input_groups:
            for or_group in input_groups["OR"]:
                if len(or_group) > 1:  # Only consider actual OR conditions (more than one option)
                    or_conditions.append([f"(completed {act})" for act in or_group])
        
        for attr, values in self.attr_activity_relationships.items(): # Check attribute OR conditions
            attr_or_values = []
            for val, activities in values.items():
                if action_name in activities:
                    stats = activities[action_name]
                    if stats['probability'] >= self.min_confidence:
                        if self.attribute_categories.get(attr) == 'boolean':
                            condition = f"({attr})" if val.lower() == 'true' else f"(not ({attr}))"
                            attr_or_values.append(condition)
                        else:
                            if attr in self._get_attribute_params(action_name):
                                attr_or_values.append(f"({attr} ?v)")
            if len(attr_or_values) > 1:  # Only consider actual OR conditions
                or_conditions.append(attr_or_values)
        
        return or_conditions


    def _generate_action_variants(self, action_name: str) -> List[Dict[str, Any]]:
        """ Generate variants of an action to replace OR conditions with separate actions. """
        or_conditions = self._find_or_preconditions(action_name)
        
        if not or_conditions:
            return []  # No variants needed
        
        def generate_combinations(or_lists, current_combo=None, index=0): # Generate all combinations of preconditions
            if current_combo is None:
                current_combo = []
            if index >= len(or_lists):
                return [current_combo]
            result = []
            for item in or_lists[index]:
                result.extend(generate_combinations(or_lists, current_combo + [item], index + 1))
            
            return result
        
        combinations = generate_combinations(or_conditions)
        variants = []
        for i, combo in enumerate(combinations):
            variant_name = f"{action_name}_v{i}"
            variants.append({
                'name': variant_name,
                'conditions': combo,
                'base_action': action_name,
                'index': i
            })
        
        return self._deduplicate_variants(variants, action_name)


    def _generate_pddl_action_definition(self, action_name: str) -> str:
        """Generate PDDL action definition for a specific activity."""
        # First check for measurement variants (separate actions per outcome)
        measurement_variants = self._generate_measurement_action_variants(action_name)
        if measurement_variants:
            all_actions = []
            decision_preconds = self.decision_preconditions.get(action_name, [])
            for variant in measurement_variants:
                variant_effects = self._generate_measurement_variant_effects(action_name, variant)
                if decision_preconds:
                    for i, preconds in enumerate(decision_preconds):
                        variant_name = f"{variant['name']}_v{i+1}"
                        all_actions.append(self._generate_single_action_definition(
                            action_name, 
                            variant_name=variant_name,
                            deterministic_effects=variant_effects,
                            decision_preconditions=preconds
                        ))
                else:
                    all_actions.append(self._generate_single_action_definition(
                        action_name, 
                        variant_name=variant['name'],
                        deterministic_effects=variant_effects
                    ))
            return "\n\n".join(all_actions)
        
        # Check for decision variants
        if action_name in self.decision_preconditions:
            all_actions = []
            decision_variants = self.decision_preconditions[action_name]
            for i, preconds in enumerate(decision_variants):
                variant_name = f"{action_name}_v{i+1}"
                all_actions.append(self._generate_single_action_definition(
                    action_name,
                    variant_name=variant_name,
                    decision_preconditions=preconds
                ))
            # Add a fallback variant without categorical preconditions
            # This allows traces with attribute values not covered by decision rules to still be solvable
            fallback_name = f"{action_name}_fallback"
            all_actions.append(self._generate_single_action_definition(
                action_name,
                variant_name=fallback_name,
                decision_preconditions=set()  # No categorical preconditions
            ))
            return "\n\n".join(all_actions)
        
        # Check for OR precondition variants
        variants = self._generate_action_variants(action_name)
        if variants:
            self.split_actions[action_name] = variants
            all_actions = []
            for variant in variants:
                all_actions.append(self._generate_single_action_definition(
                    action_name, 
                    variant_name=variant['name'],
                    variant_conditions=variant['conditions']
                ))
            
            return "\n\n".join(all_actions)
        
        # No variants needed, generate a single action
        return self._generate_single_action_definition(action_name)


    def _generate_single_action_definition(
        self, 
        action_name: str, 
        variant_name: Optional[str] = None, 
        variant_conditions: Optional[List[str]] = None, 
        deterministic_effects: Optional[List[str]] = None, 
        decision_preconditions: Optional[Set[str]] = None
    ) -> str:
        """Generate a single PDDL action definition, optionally for a variant."""
        is_starting_action = action_name in self.parser.start_activities
        display_name = variant_name if variant_name else action_name
        
        # Use the display name (variant name) to determine parameters, not the base action name
        if is_starting_action:
            param_attributes = set()
        else:
            param_attributes = self._get_attribute_params(display_name)
        
        action_def = [f"  (:action exec_{display_name}"]
        parameters = []
        param_counts = {}
        for attr in param_attributes:
            param_counts['v'] = param_counts.get('v', 0) + 1
            param_type = self._get_parameter_type(attr)
            parameters.append(f"?v{param_counts['v']} - {param_type}")
        
        if parameters:
            action_def.append(f"   :parameters ({' '.join(parameters)})")
        else:
            action_def.append("   :parameters ()")
        
        # Add preconditions
        self._add_action_preconditions(action_def, action_name, variant_conditions, decision_preconditions, is_starting_action)
        
        # Add effects
        self._add_action_effects(action_def, action_name, deterministic_effects)
        
        action_def.append("  )")
        return "\n".join(action_def)

    def _get_parameter_type(self, attr: str) -> str:
        """Get the parameter type for an attribute."""
        return f"{attr}_type"

    def _add_action_preconditions(
        self, 
        action_def: List[str], 
        action_name: str, 
        variant_conditions: Optional[List[str]], 
        extra_preconditions: Optional[Set[str]] = None, 
        is_starting_action: bool = False
    ) -> None:
        """Add preconditions to action definition."""
        preconditions = self._collect_preconditions(action_name, variant_conditions)
        if extra_preconditions:
            if is_starting_action:
                filtered = [p for p in extra_preconditions if not p.startswith('(completed ')]
                preconditions.extend(filtered)
            else:
                preconditions.extend(extra_preconditions)
        preconditions = list(set(preconditions))
        if preconditions:
            if len(preconditions) == 1:
                action_def.append(f"   :precondition {preconditions[0]}")
            else:
                action_def.append("   :precondition (and\n      " + "\n      ".join(preconditions) + "\n   )")
        else:
            action_def.append("   :precondition ()")

    def _add_action_effects(self, action_def: List[str], action_name: str, deterministic_effects: Optional[List[str]]) -> None:
        """Add effects to action definition."""
        if deterministic_effects is not None:
            effects = deterministic_effects
        else:
            effects = self._collect_effects(action_name)
            
        if len(effects) == 1:
            action_def.append(f"   :effect {effects[0]}")
        else:
            action_def.append("   :effect (and\n      " + "\n      ".join(effects) + "\n   )")


    def generate_domain(self, output_path: Optional[str] = None) -> str:
        """Generate PDDL domain string and optionally save to file."""
        requirements = ":strips :typing :universal-preconditions :conditional-effects :negative-preconditions"
        domain = [
            f"(define (domain {self.domain_name})", f"  (:requirements {requirements})", "",
            self._generate_pddl_type_definitions(), "",
            self._generate_pddl_constants_definitions(), "",
            self._generate_pddl_predicate_definitions(), ""
        ]
        
        for action in sorted(self.activities):
            domain.append(self._generate_pddl_action_definition(action))
            domain.append("")
        domain.append(")")
        domain_content = "\n".join(domain)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(domain_content)
        
        return domain_content
    

    def generate_problem(self, problem_name: str = "process_problem", output_path: Optional[str] = None) -> str:
        """
        Generate PDDL problem file with initial state and goal.
        
        Args:
            problem_name: Name of the problem
            output_path: Path to save the problem file
        """
        custom_init = self.init
        custom_goal = self.goal
        
        problem = [
            f"(define (problem {problem_name}_prediction)",
            f"  (:domain {self.domain_name})",
            ""
        ]
        
        # Add init section
        problem.append(self._generate_init_section(custom_init))
        problem.append("")
        
        # Add goal section
        problem.append(self._generate_goal_section(custom_goal))
        problem.append("")
        problem.append(")")
        
        problem_content = "\n".join(problem)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(problem_content)
        
        return problem_content

    def _add_attribute_constants(self, constants: List[str], relevant_attrs: Set[str]) -> None:
        """Add constants for numerical and categorical attributes."""
        for attr, category in self.attribute_categories.items():
            if category in ['numerical', 'categorical'] and attr in relevant_attrs and attr in self.parser.attribute_domains:
                vals = self.parser.attribute_domains[attr]
                if vals:
                    constants.append(f"    {' '.join(sorted(vals))} - {attr}_type")

    def _generate_pddl_constants_definitions(self) -> str:
        """Generate the constants section of the PDDL domain."""
        constants = ["  (:constants"]
        
        # Collect attributes that influence or are influenced by actions
        relevant_attrs = set()
        for attr in self.attr_activity_relationships:
            relevant_attrs.add(attr)
        for activity, effects in self.activity_attr_effects.items():
            for effect in effects:
                if 'attribute' in effect:
                    relevant_attrs.add(effect['attribute'])
        
        # Also include attributes for measurement actions
        for activity in self.activities:
            variants = self._generate_measurement_action_variants(activity)
            if variants:
                sanitized_action = activity.lower()
                matching_attribute = next((attr for attr in self.attribute_categories 
                                        if attr.lower() == sanitized_action), None)
                if matching_attribute:
                    relevant_attrs.add(matching_attribute)
        
        # Include attributes from decision preconditions
        relevant_attrs |= self.decision_attributes
        print(f"dECISION attributes: {self.decision_attributes}")
        
        # Add activity constants
        activity_names = sorted(self.activities)
        if activity_names:
            constants.append(f"    {' '.join(activity_names)} - activity")
        
        # Add attribute constants
        self._add_attribute_constants(constants, relevant_attrs)
        
        constants.append("  )")
        return "\n".join(constants)

    def _ensure_value_mappings(self, attr: str, vals: Set[Any], prefix: str) -> None:
        """Ensure value mappings exist for categorical attributes."""
        if not hasattr(self, '_value_mappings'):
            self._value_mappings = {}
        self._value_mappings[attr] = {val: utils.sanitize_name(val) for val in vals}

    def _generate_init_section(self, custom_init: Optional[List[str]]) -> str:
        """Generate the initial state section of the PDDL problem."""
        init = ["  (:init"]
        
        # Collect attributes that influence or are influenced by actions
        relevant_attrs = set()
        for attr in self.attr_activity_relationships:
            relevant_attrs.add(attr)
        for activity, effects in self.activity_attr_effects.items():
            for effect in effects:
                if 'attribute' in effect:
                    relevant_attrs.add(effect['attribute'])
        
        # Also include attributes for measurement actions
        for activity in self.activities:
            variants = self._generate_measurement_action_variants(activity)
            if variants:
                sanitized_action = activity.lower()
                matching_attribute = next((attr for attr in self.attribute_categories 
                                        if attr.lower() == sanitized_action), None)
                if matching_attribute:
                    relevant_attrs.add(matching_attribute)
        
        if custom_init:
            for condition in custom_init:
                init.append(f"    {condition}")
        else:
            if self.parser.start_activities:
                first_start_activity = next(iter(self.parser.start_activities))
                init.append(f"    (enabled {first_start_activity})")
        
        init.append("  )")
        return "\n".join(init)
    
    def _generate_goal_section(self, custom_goal: Optional[List[str]]) -> str:
        """Generate the goal section of the PDDL problem."""
        goal = ["  (:goal", "    (and"]
        
        if custom_goal:
            for condition in custom_goal:
                goal.append(f"      {condition}")
        else:
            goal_condition = self._create_default_goal()
            goal.append(f"      {goal_condition}")
        
        goal.append("    ) )")
        return "\n".join(goal)

    def _create_default_goal(self) -> str:
        """Create a default goal condition."""
        if self.parser.end_activities:
            # Filter end activities to only include those that are in the discovered activities
            valid_end_activities = [activity for activity in self.parser.end_activities.keys() if activity in self.activities]
            if valid_end_activities:
                random_end = random.choice(valid_end_activities)
                return f"(completed {random_end})"
        return "()"  # No valid end activities found


    def _get_successors(self, action_name: str) -> List[str]:
        """Get successors using the direct transition graph."""
        return self.parser.direct_transition_graph.get(action_name, [])
    

    def _process_decision_points(self) -> None:
        """Process and normalize decision point probabilities for internal mapping."""
        self.processed_decision_points = {}
        for key_str, probs in self.decision_points.items():
            if isinstance(probs, dict):
                if isinstance(key_str, str):
                    activity_name = self._extract_activity_name(key_str)
                    if activity_name:
                        self.processed_decision_points[activity_name] = probs
            elif isinstance(key_str, str) and not key_str.startswith("{"):
                self.processed_decision_points[key_str] = probs
            elif isinstance(key_str, str) and key_str.startswith("{"):
                try:
                    source_part = key_str.split(",")[0].replace("{(", "").replace("{'", "").replace("'}", "")
                    sources = [s.strip().strip("'") for s in source_part.split(",")]
                    for source in sources:
                        source_activity = utils.sanitize_name(source)
                        if source_activity:
                            self.processed_decision_points[source_activity] = probs
                except Exception as e:
                    print(f"Error processing complex key {key_str}: {e}")


    def _normalize_action_signature(self, preconditions: List[str], effects: List[str]) -> str:
        """Create a normalized string signature for an action to aid deduplication."""
        normalized_preconditions = sorted(set(preconditions)) if preconditions else []
        normalized_effects = sorted(set(effects)) if effects else []
        precond_str = " AND ".join(normalized_preconditions)
        effect_str = " AND ".join(normalized_effects)
        
        return f"PRECOND[{precond_str}]_EFFECT[{effect_str}]"

    def _deduplicate_variants(self, variants: List[Dict[str, Any]], action_name: str) -> List[Dict[str, Any]]:
        """Remove redundant action variants based on their preconditions and effects."""
        if not variants:
            return variants
            
        seen_signatures = {}
        unique_variants = []
        for variant in variants:
            if 'name' in variant:
                variant_name = variant['name']
                if 'conditions' in variant:  # OR condition variants
                    preconditions = self._collect_preconditions(action_name, variant['conditions'])
                else:
                    preconditions = self._collect_preconditions(action_name)
                
                if 'effects' in variant:  # Deterministic variants
                    effects = variant['effects']
                elif 'attribute' in variant:  # Measurement variants
                    effects = self._generate_measurement_variant_effects(action_name, variant)
                else:
                    effects = self._collect_effects(action_name)
                
                signature = self._normalize_action_signature(preconditions, effects)
                if signature in seen_signatures:
                    print(f"Skipping duplicate variant '{variant_name}' - identical to '{seen_signatures[signature]}'")
                    continue
                else:
                    seen_signatures[signature] = variant_name
                    unique_variants.append(variant)
            else:
                unique_variants.append(variant)
        
        if len(unique_variants) != len(variants):
            print(f"Removed {len(variants) - len(unique_variants)} duplicate variants for action '{action_name}'")
            
        return unique_variants

    def _extract_activity_name(self, key_str: str) -> Optional[str]:
        """Extract a valid activity name from a potentially complex dictionary key string."""
        key_lower = str(key_str).lower()
        for activity in self.activities:
            if activity is not None and activity.lower() in key_lower:
                return activity
                
        return None