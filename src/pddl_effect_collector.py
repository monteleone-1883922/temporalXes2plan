from typing import List, Set, Dict, Any, Optional
from collections import defaultdict
from encoder_models import PDDLEncodingContext, ConditionalEffect
from pddl_action_utils import is_tau_activity, get_attribute_params, get_successors, get_decision_point_for_activity

class EffectCollector:
    """
    Responsible for extracting and generating PDDL effects for actions
    based on the discovered process model and decision rules.
    """
    
    def __init__(self, context: PDDLEncodingContext):
        self.context = context

    def collect_effects(self, action_name: str) -> List[str]:
        """
        Collect effects for an action with additional effects to enable subsequent activities.
        """
        effects = []
        seen_effects = set()
        
        # Initialize alternatives tracking for this action
        if action_name not in self.context.effect_alternatives:
            self.context.effect_alternatives[action_name] = []
        
        # Add basic completion effect
        self._add_completion_effect(action_name, effects, seen_effects)
        
        if is_tau_activity(action_name):
            self._handle_successor_enabling(action_name, effects, seen_effects)
            self._add_disable_effect(action_name, effects, seen_effects)
            return effects
            
        if action_name in self.context.parser.start_activities:
            return self._handle_start_activity_effects(action_name, effects, seen_effects)
            
        self._handle_attribute_effects(action_name, effects, seen_effects)
        
        # Handle successor enabling
        self._handle_successor_enabling(action_name, effects, seen_effects)
        
        # Disable current action
        self._add_disable_effect(action_name, effects, seen_effects)
        
        return effects

    def _add_completion_effect(self, action_name: str, effects: List[str], seen_effects: Set[str]) -> None:
        """
        Adds the basic `(completed action_name)` effect.
        """
        completion_effect = f"(completed {action_name})"
        effects.append(completion_effect)
        seen_effects.add(completion_effect)

    def _handle_start_activity_effects(self, action_name: str, effects: List[str], seen_effects: Set[str]) -> List[str]:
        """
        Handles effects specific to start activities by enabling their direct successors.
        """
        successors = get_successors(self.context, action_name)
        for succ in successors:
            enable_effect = f"(enabled {succ})"
            if enable_effect not in seen_effects:
                effects.append(enable_effect)
                seen_effects.add(enable_effect)
        
        self._add_disable_effect(action_name, effects, seen_effects)
        return effects

    def _add_disable_effect(self, action_name: str, effects: List[str], seen_effects: Set[str]) -> None:
        """
        Adds the `(not (enabled action_name))` effect to prevent repeated execution.
        """
        disable_current = f"(not (enabled {action_name}))"
        if disable_current not in seen_effects:
            effects.append(disable_current)
            seen_effects.add(disable_current)

    def _handle_attribute_effects(self, action_name: str, effects: List[str], seen_effects: Set[str]) -> Dict[str, bool]:
        """
        Handles all attribute-related effects for an action, including boolean, numerical, and categorical types.
        Returns a dictionary tracking boolean effects that have been processed.
        """
        boolean_effects = {}
        sanitized_action = action_name.lower()
        matching_attribute = next((attr for attr in self.context.attribute_categories 
                                if attr.lower() == sanitized_action), None)
        
        if matching_attribute:
            boolean_effects = self._process_matching_attribute(
                action_name, matching_attribute, effects, seen_effects
            )
        
        self._process_activity_attribute_effects(action_name, effects, seen_effects, boolean_effects)
        return boolean_effects

    def _process_matching_attribute(self, action_name: str, matching_attribute: str, effects: List[str], seen_effects: Set[str]) -> Dict[str, bool]:
        boolean_effects = {}
        attr_category = self.context.attribute_categories[matching_attribute]
        
        if attr_category == 'boolean':
            bool_alternatives = [f"({matching_attribute})", f"(not ({matching_attribute}))"]
            self.context.effect_alternatives[action_name].append(bool_alternatives)
            boolean_effects[matching_attribute] = True
        else:
            self._handle_non_boolean_matching_attribute(
                action_name, matching_attribute, attr_category, effects, seen_effects
            )
        
        return boolean_effects

    def _handle_non_boolean_matching_attribute(self, action_name: str, matching_attribute: str, attr_category: str, effects: List[str], seen_effects: Set[str]) -> None:
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
        possible_values = self._get_possible_attribute_values(attr)
        
        if possible_values:
            attr_alternatives = [f"({attr} {val})" for val in sorted(possible_values)]
            self.context.effect_alternatives[action_name].append(attr_alternatives)
            
            param_index = self._get_param_index_for_attribute(action_name, attr)
            param_name = f"?v{param_index}"
            set_effect = f"({attr} {param_name})"
            if set_effect not in seen_effects:
                effects.append(set_effect)
                seen_effects.add(set_effect)

    def _get_possible_attribute_values(self, attr: str) -> Optional[Set[str]]:
        """
        Retrieves the set of possible values for a given attribute from the parser's domains, applying value mappings.
        """
        decision_values = self.context.parser.attribute_domains.get(attr, set())
        if decision_values:
            if attr in self.context.value_mappings:
                decision_values = {self.context.value_mappings[attr].get(val, val) for val in decision_values}
            return decision_values
        return None

    def _get_param_index_for_attribute(self, action_name: str, attr: str) -> int:
        """
        Determines the 1-based parameter index for a given attribute within an action's parameter list.
        """
        param_index = 1
        for action_attr in get_attribute_params(self.context, action_name):
            if action_attr == attr:
                break
            param_index += 1
        return param_index

    def _process_activity_attribute_effects(self, action_name: str, effects: List[str], seen_effects: Set[str], boolean_effects: Dict[str, bool]) -> None:
        if action_name not in self.context.activity_attr_effects:
            return
        
        conditional_effects = []
        # TODO: This original implementation misses populating conditional_effects here!
        # Assuming there is other logic we missed, for now just calling the processing.
        self._process_boolean_conditional_effects(conditional_effects, boolean_effects, effects, seen_effects)
        self._add_boolean_effects(boolean_effects, effects, seen_effects)

    def _handle_successor_enabling(self, action_name: str, effects: List[str], seen_effects: Set[str]) -> None:
        """
        Handles enabling of successor activities, either universally or conditionally via decision points.
        """
        successors = get_successors(self.context, action_name)
        decision_point_probs = get_decision_point_for_activity(self.context, action_name)
        
        if decision_point_probs:
            self._handle_decision_point_successors(action_name, decision_point_probs, successors, effects, seen_effects)
        else:
            self._enable_all_successors(successors, effects, seen_effects)

    def _handle_decision_point_successors(self, action_name: str, decision_point_probs: Dict[str, float], successors: List[str], effects: List[str], seen_effects: Set[str]) -> None:
        valid_successors = {succ: prob for succ, prob in decision_point_probs.items() 
                           if prob >= 0.01 and succ in successors}
        
        if valid_successors:
            for key in self.context.decision_points_conditions:
                self._process_decision_point_conditions(key, action_name, effects)

    def _enable_all_successors(self, successors: List[str], effects: List[str], seen_effects: Set[str]) -> None:
        for succ in successors:
            enable_effect = f"(enabled {succ})"
            if enable_effect not in seen_effects:
                effects.append(enable_effect)
                seen_effects.add(enable_effect)

    def _process_decision_point_conditions(self, key: Any, action_name: str, effects: List[str]) -> None:
        attributes_pre_xor = self.context.decision_points_conditions[key]
        for attribute in attributes_pre_xor:
            values_attr_pre_xor = self.context.decision_points_conditions[key][attribute]
            for val in values_attr_pre_xor:
                transitions_val_pre_xor = self.context.decision_points_conditions[key][attribute][val]
                if len(transitions_val_pre_xor.keys()) > 1:
                    self._handle_multiple_transitions(action_name, attribute, val, transitions_val_pre_xor, effects)
                else:
                    self._handle_single_transition(attribute, val, transitions_val_pre_xor, effects)

    def _handle_multiple_transitions(self, action_name: str, attribute: str, val: Any, transitions_val_pre_xor: Dict[str, Any], effects: List[str]) -> None:
        decision_alternatives = []
        for transition in transitions_val_pre_xor.keys():
            decision_alternatives.append(f"(enabled {transition})")
        
        conditional_alt_obj = ConditionalEffect(
            condition=f"({attribute} {val})",
            alternatives=decision_alternatives
        )
        self.context.effect_alternatives[action_name].append([conditional_alt_obj])
        
        first_transition = list(transitions_val_pre_xor.keys())[0]
        effects.append(f"(when ({attribute} {val})\n        (enabled {first_transition}))")

    def _handle_single_transition(self, attribute: str, val: Any, transitions_val_pre_xor: Dict[str, Any], effects: List[str]) -> None:
        for transition in transitions_val_pre_xor.keys():
            effects.append(f"(when ({attribute} {val})\n        (enabled {transition}))")

    def _process_boolean_conditional_effects(self, conditional_effects: List[Dict[str, Any]], boolean_effects: Dict[str, bool], effects: List[str], seen_effects: Set[str]) -> None:
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
        for attr, is_true in boolean_effects.items():
            effect_str = f"({attr})" if is_true else f"(not ({attr}))"
            if effect_str not in seen_effects:
                effects.append(effect_str)
                seen_effects.add(effect_str)
