from typing import List, Dict, Any, Optional, Set, Union
from .encoder_models import PDDLEncodingContext, ActionVariant, MeasurementVariant
from .pddl_precondition_collector import PreconditionCollector
from .pddl_effect_collector import EffectCollector
from .pddl_action_utils import get_attribute_params, get_successors

class ActionBuilder:
    """
    Builds the PDDL action definitions by orchestrating the precondition
    and effect collectors, and handling the generation of action variants.
    """
    def __init__(self, context: PDDLEncodingContext, precondition_collector: PreconditionCollector, effect_collector: EffectCollector):
        self.context = context
        self.precondition_collector = precondition_collector
        self.effect_collector = effect_collector

    def generate_pddl_action_definition(self, action_name: str) -> str:
        """
        Generate the full PDDL action definition for a given activity.
        This handles parameters, preconditions (structural and data-driven), 
        and effects (completion, enabling successors, and attribute updates).
        """
        # First check for measurement variants (separate actions per outcome)
        measurement_variants = self._generate_measurement_action_variants(action_name)
        if measurement_variants:
            all_actions = []
            decision_preconds = self.context.decision_preconditions.get(action_name, [])
            for variant in measurement_variants:
                variant_effects = self._generate_measurement_variant_effects(action_name, variant)
                if decision_preconds:
                    for i, preconds in enumerate(decision_preconds):
                        variant_name = f"{variant.name}_v{i+1}"
                        all_actions.append(self._generate_single_action_definition(
                            action_name, 
                            variant_name=variant_name,
                            deterministic_effects=variant_effects,
                            decision_preconditions=preconds
                        ))
                else:
                    all_actions.append(self._generate_single_action_definition(
                        action_name, 
                        variant_name=variant.name,
                        deterministic_effects=variant_effects
                    ))
            return "\n\n".join(all_actions)
        
        # Check for decision variants
        if action_name in self.context.decision_preconditions:
            all_actions = []
            decision_variants = self.context.decision_preconditions[action_name]
            for i, preconds in enumerate(decision_variants):
                variant_name = f"{action_name}_v{i+1}"
                all_actions.append(self._generate_single_action_definition(
                    action_name,
                    variant_name=variant_name,
                    decision_preconditions=preconds
                ))
            # Add a fallback variant without categorical preconditions
            fallback_name = f"{action_name}_fallback"
            all_actions.append(self._generate_single_action_definition(
                action_name,
                variant_name=fallback_name,
                decision_preconditions=set()
            ))
            return "\n\n".join(all_actions)
        
        # Check for OR precondition variants
        variants = self._generate_action_variants(action_name)
        if variants:
            self.context.split_actions[action_name] = variants
            all_actions = []
            for variant in variants:
                all_actions.append(self._generate_single_action_definition(
                    action_name, 
                    variant_name=variant.name,
                    variant_conditions=variant.conditions
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
        """
        Generates a single PDDL action definition.
        
        It determines the parameters for the action, formats them, and calls helpers 
        to add the preconditions and effects blocks.
        
        Args:
            action_name (str): The base name of the action.
            variant_name (Optional[str]): Name of the variant, used as the action's display name.
            variant_conditions (Optional[List[str]]): Specific conditions for this variant.
            deterministic_effects (Optional[List[str]]): Pre-computed effects to apply directly.
            decision_preconditions (Optional[Set[str]]): Additional preconditions from decision rules.
            
        Returns:
            str: The formatted PDDL `:action` block.
        """
        is_starting_action = action_name in self.context.parser.start_activities
        display_name = variant_name if variant_name else action_name
        
        if is_starting_action:
            param_attributes = set()
        else:
            param_attributes = get_attribute_params(self.context, display_name)
        
        action_def = [f"  (:action exec_{display_name}"]
        parameters = []
        param_counts = {}
        for attr in param_attributes:
            param_counts['v'] = param_counts.get('v', 0) + 1
            param_type = f"{attr}_type"
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

    def _add_action_preconditions(
        self, 
        action_def: List[str], 
        action_name: str, 
        variant_conditions: Optional[List[str]], 
        extra_preconditions: Optional[Set[str]] = None, 
        is_starting_action: bool = False
    ) -> None:
        """
        Appends the `:precondition` block to the action definition.
        
        Args:
            action_def (List[str]): The lines of the action definition being built.
            action_name (str): The base action name.
            variant_conditions (Optional[List[str]]): Specific variant conditions to include.
            extra_preconditions (Optional[Set[str]]): Additional preconditions to add.
            is_starting_action (bool): True if the action is a starting activity.
        """
        preconditions = self.precondition_collector.collect_preconditions(action_name, variant_conditions)
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
        """
        Appends the `:effect` block to the action definition.
        
        Args:
            action_def (List[str]): The lines of the action definition being built.
            action_name (str): The base action name.
            deterministic_effects (Optional[List[str]]): Pre-computed effects to apply directly.
        """
        if deterministic_effects is not None:
            effects = deterministic_effects
        else:
            effects = self.effect_collector.collect_effects(action_name)
            
        if len(effects) == 1:
            action_def.append(f"   :effect {effects[0]}")
        else:
            action_def.append("   :effect (and\n      " + "\n      ".join(effects) + "\n   )")

    def _generate_measurement_action_variants(self, action_name: str) -> List[MeasurementVariant]:
        """
        Generates separate action variants for each measurement outcome.
        
        This avoids PDDL semantic issues by creating distinct actions with deterministic 
        effects based on the chosen measurement outcome.
        
        Args:
            action_name (str): The base measurement action name.
            
        Returns:
            List[MeasurementVariant]: A list of generated measurement variants.
        """
        #TODO: rework since this is based only on action name to discriminate measurement actions
        # more general should be actions introducing new attributes
        variants = []
        sanitized_action = action_name.lower()
        matching_attribute = next((attr for attr in self.context.attribute_categories 
                                if attr.lower() == sanitized_action), None)
        
        if not matching_attribute:
            return []

        # boolean attr requires no variants
        attr_category = self.context.attribute_categories[matching_attribute]
        if attr_category == 'boolean':
            return []
            
        possible_values = self.context.parser.attribute_domains.get(matching_attribute, set())
        if matching_attribute in self.context.value_mappings:
            possible_values = {self.context.value_mappings[matching_attribute].get(val, val) for val in possible_values}
        
        if not possible_values:
            return []
            
        for value in sorted(possible_values):
            variant_name = f"{action_name}-{value}"
            variants.append(MeasurementVariant(
                name=variant_name,
                attribute=matching_attribute,
                outcome_value=value,
                base_action=action_name
            ))
        
        return self._deduplicate_measurement_variants(variants, action_name)

    def _generate_measurement_variant_effects(self, action_name: str, variant: MeasurementVariant) -> List[str]:
        """
        Generates effects for a specific measurement outcome variant.
        
        Args:
            action_name (str): The base measurement action name.
            variant (MeasurementVariant): The specific measurement outcome variant.
            
        Returns:
            List[str]: A list of effect strings representing the deterministic outcome.
        """
        effects = []
        seen_effects = set()
        
        completion_effect = f"(completed {action_name})"
        effects.append(completion_effect)
        seen_effects.add(completion_effect)
        
        attribute = variant.attribute
        outcome_value = variant.outcome_value
        
        num_type = f"{attribute}_type"
        remove_effect = (
            f"(forall (?old - {num_type})\n"
            f"       (when ({attribute} ?old)\n"
            f"         (not ({attribute} ?old))))"
        )
        effects.append(remove_effect)
        seen_effects.add(remove_effect)
        
        set_effect = f"({attribute} {outcome_value})"
        effects.append(set_effect)
        seen_effects.add(set_effect)
        
        successors = get_successors(self.context, action_name)
        for succ in successors:
            enable_effect = f"(enabled {succ})"
            if enable_effect not in seen_effects:
                effects.append(enable_effect)
                seen_effects.add(enable_effect)
        
        disable_current = f"(not (enabled {action_name}))"
        if disable_current not in seen_effects:
            effects.append(disable_current)
            seen_effects.add(disable_current)
        
        return effects

    def _generate_action_variants(self, action_name: str) -> List[ActionVariant]:
        """
        Generates variants of an action to replace complex OR conditions with separate actions.
        
        Since standard PDDL struggles natively with complex OR preconditions, this method 
        breaks them down into combinations, creating a distinct action variant for each combination.
        
        Args:
            action_name (str): The base action name.
            
        Returns:
            List[ActionVariant]: A list of generated action variants.
        """
        or_conditions = self.precondition_collector.find_or_preconditions(action_name)
        
        if not or_conditions:
            return []
        
        def generate_combinations(or_lists, current_combo=None, index=0):
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
            variants.append(ActionVariant(
                name=variant_name,
                conditions=combo,
                base_action=action_name,
                index=i
            ))
        
        return self._deduplicate_action_variants(variants, action_name)

    def _normalize_action_signature(self, preconditions: List[str], effects: List[str]) -> str:
        """
        Creates a normalized string signature for an action to aid deduplication.
        
        Args:
            preconditions (List[str]): List of precondition strings.
            effects (List[str]): List of effect strings.
            
        Returns:
            str: A unique signature representing the action's preconditions and effects.
        """
        normalized_preconditions = sorted(set(preconditions)) if preconditions else []
        normalized_effects = sorted(set(effects)) if effects else []
        precond_str = " AND ".join(normalized_preconditions)
        effect_str = " AND ".join(normalized_effects)
        return f"PRECOND[{precond_str}]_EFFECT[{effect_str}]"

    def _deduplicate_action_variants(self, variants: List[ActionVariant], action_name: str) -> List[ActionVariant]:
        """
        Removes redundant action variants based on their semantic signatures (preconditions + effects).
        
        Args:
            variants (List[ActionVariant]): The original list of variants.
            action_name (str): The base action name.
            
        Returns:
            List[ActionVariant]: A deduplicated list of unique variants.
        """
        if not variants:
            return variants
            
        seen_signatures = {}
        unique_variants = []
        for variant in variants:
            preconditions = self.precondition_collector.collect_preconditions(action_name, variant.conditions)
            effects = self.effect_collector.collect_effects(action_name)
            
            signature = self._normalize_action_signature(preconditions, effects)
            if signature in seen_signatures:
                continue
            else:
                seen_signatures[signature] = variant.name
                unique_variants.append(variant)
                
        return unique_variants

    def _deduplicate_measurement_variants(self, variants: List[MeasurementVariant], action_name: str) -> List[MeasurementVariant]:
        """
        Removes redundant measurement variants based on their semantic signatures.
        
        Args:
            variants (List[MeasurementVariant]): The original list of measurement variants.
            action_name (str): The base action name.
            
        Returns:
            List[MeasurementVariant]: A deduplicated list of unique measurement variants.
        """
        if not variants:
            return variants
            
        seen_signatures = {}
        unique_variants = []
        for variant in variants:
            preconditions = self.precondition_collector.collect_preconditions(action_name)
            effects = self._generate_measurement_variant_effects(action_name, variant)
            
            signature = self._normalize_action_signature(preconditions, effects)
            if signature in seen_signatures:
                continue
            else:
                seen_signatures[signature] = variant.name
                unique_variants.append(variant)
                
        return unique_variants
