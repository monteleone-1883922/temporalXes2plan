import os
from typing import Optional, Set
from .encoder_models import PDDLEncodingContext
from .pddl_action_builder import ActionBuilder

class DomainBuilder:
    """
    Responsible for generating the high-level PDDL domain structure,
    including types, constants, and predicates.
    """
    def __init__(self, context: PDDLEncodingContext, action_builder: ActionBuilder):
        self.context = context
        self.action_builder = action_builder

    def generate_domain(self, output_path: Optional[str] = None) -> str:
        """Generate PDDL domain string and optionally save to file."""
        requirements = ":strips :typing :universal-preconditions :conditional-effects :negative-preconditions"
        domain = [
            f"(define (domain {self.context.domain_name})", f"  (:requirements {requirements})", "",
            self._generate_pddl_type_definitions(), "",
            self._generate_pddl_constants_definitions(), "",
            self._generate_pddl_predicate_definitions(), ""
        ]
        
        for action in sorted(self.context.activities):
            domain.append(self.action_builder.generate_pddl_action_definition(action))
            domain.append("")
        domain.append(")")
        domain_content = "\n".join(domain)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(domain_content)
        
        return domain_content

    def _generate_pddl_predicate_definitions(self) -> str:
        """
        Generates the `:predicates` section for the PDDL domain.
        
        This section defines all the logical statements (predicates) that can be true or false 
        in the planning problem. It inherently includes structural predicates like `(completed ?a)` 
        and `(enabled ?a)`. Additionally, it dynamically includes predicates for variables based 
        on relationships discovered in the event log (e.g., categorical or numerical attributes).
        
        Returns:
            str: The formatted PDDL `:predicates` block.
        """
        predicates = [
            "    (completed ?a - activity)",
            "    (enabled ?a - activity)"
        ]
        relevant_attrs = set()
        
        for attr in self.context.attr_activity_relationships:
            relevant_attrs.add(attr)
        
        for activity in self.context.activities:
            variants = self.action_builder._generate_measurement_action_variants(activity)
            if variants:
                sanitized_action = activity.lower()
                matching_attribute = next((attr for attr in self.context.attribute_categories 
                                        if attr.lower() == sanitized_action), None)
                if matching_attribute:
                    relevant_attrs.add(matching_attribute)
        
        relevant_attrs |= self.context.decision_attributes
        
        for sample in getattr(self.context.parser, 'decision_samples', []):
            for precond in sample.get('preconditions', []):
                if isinstance(precond, str) and precond.startswith('(') and precond.endswith(')'):
                    inner = precond[1:-1]
                    parts = inner.split()
                    if len(parts) >= 1:
                        attr = parts[0]
                        relevant_attrs.add(attr)
        
        for attr, category in self.context.attribute_categories.items():
            if attr in relevant_attrs:
                predicate = self._create_attribute_predicate(attr, category)
                if predicate:
                    predicates.append(predicate)

        return "  (:predicates\n" + "\n".join(predicates) + "\n  )"

    def _create_attribute_predicate(self, attr: str, category: str) -> Optional[str]:
        """
        Creates a single attribute predicate declaration based on its data category.
        
        Args:
            attr (str): The name of the attribute.
            category (str): The data type category ('boolean', 'numerical', or 'categorical').
            
        Returns:
            Optional[str]: The formatted predicate string, e.g., `(attr_name ?v - attr_name_type)`, 
                           or None if not applicable.
        """
        if category == 'boolean':
            return f"    ({attr})"
        elif category == 'numerical':
            return f"    ({attr} ?v - {attr}_type)"
        else:
            return f"    ({attr} ?v - {attr}_type)"

    def _generate_pddl_type_definitions(self) -> str:
        """
        Generates the `:types` section for the PDDL domain based on used predicates.
        
        Collects all attributes that influence or are influenced by actions, attributes 
        affected by measurement actions, and those found in decision preconditions.
        It then formats them into a PDDL `:types` declaration, categorizing them appropriately 
        (e.g., assigning custom types like `attr_name_type` to numerical or categorical data).
        
        Returns:
            str: The formatted PDDL `:types` block.
        """
        relevant_attrs = {attr for attr in self.context.attr_activity_relationships}
        
        for activity in self.context.activities:
            variants = self.action_builder._generate_measurement_action_variants(activity)
            if variants:
                sanitized_action = activity.lower()
                matching_attribute = next((attr for attr in self.context.attribute_categories 
                                        if attr.lower() == sanitized_action), None)
                if matching_attribute:
                    relevant_attrs.add(matching_attribute)
        
        relevant_attrs |= self.context.decision_attributes
        
        for sample in getattr(self.context.parser, 'decision_samples', []):
            for precond in sample.get('preconditions', []):
                if isinstance(precond, str) and precond.startswith('(') and precond.endswith(')'):
                    inner = precond[1:-1]
                    parts = inner.split()
                    if len(parts) >= 1:
                        attr = parts[0]
                        relevant_attrs.add(attr)
        types = ["activity"]
        for attr in relevant_attrs:
            category = self.context.attribute_categories.get(attr)
            if category in ['numerical', 'categorical']:
                types.append(f"{attr}_type")
        
        return (
            "  (:types\n"
            f"    {' '.join(types)}\n"
            "  )"
        )

    def _generate_pddl_constants_definitions(self) -> str:
        """
        Generates the `:constants` section of the PDDL domain.
        
        Constants represent objects that exist in all problem instances for this domain.
        This includes all activities (as they are constants of type `activity`) and specific 
        possible values for categorical and numerical attributes discovered from the parser's 
        attribute domains.
        
        Returns:
            str: The formatted PDDL `:constants` block.
        """
        constants = ["  (:constants"]
        
        relevant_attrs = set()
        for attr in self.context.attr_activity_relationships:
            relevant_attrs.add(attr)
        for activity, effects in self.context.activity_attr_effects.items():
            for effect in effects:
                if 'attribute' in effect:
                    relevant_attrs.add(effect['attribute'])
        
        for activity in self.context.activities:
            variants = self.action_builder._generate_measurement_action_variants(activity)
            if variants:
                sanitized_action = activity.lower()
                matching_attribute = next((attr for attr in self.context.attribute_categories 
                                        if attr.lower() == sanitized_action), None)
                if matching_attribute:
                    relevant_attrs.add(matching_attribute)
        
        relevant_attrs |= self.context.decision_attributes
        
        activity_names = sorted(self.context.activities)
        if activity_names:
            constants.append(f"    {' '.join(activity_names)} - activity")
        
        self._add_attribute_constants(constants, relevant_attrs)
        
        constants.append("  )")
        return "\n".join(constants)

    def _add_attribute_constants(self, constants: list, relevant_attrs: set) -> None:
        """
        Appends discovered attribute values to the constants list.
        
        Iterates over the context's attribute categories and the set of relevant attributes, 
        and appends their domain values as constants of their specific type to the provided list.
        
        Args:
            constants (list): The list of PDDL constant string lines being built.
            relevant_attrs (set): The set of attributes that are relevant to the domain.
        """
        for attr, category in self.context.attribute_categories.items():
            if category in ['numerical', 'categorical'] and attr in relevant_attrs and attr in self.context.parser.attribute_domains:
                vals = self.context.parser.attribute_domains[attr]
                if vals:
                    constants.append(f"    {' '.join(sorted(vals))} - {attr}_type")
