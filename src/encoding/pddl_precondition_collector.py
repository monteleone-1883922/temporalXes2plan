from typing import List, Dict, Any, Optional
import pm4py
import core_utils as utils
from collections import defaultdict
from .encoder_models import PDDLEncodingContext
from .pddl_action_utils import get_attribute_params

class PreconditionCollector:
    """
    Responsible for extracting and generating PDDL preconditions for actions
    based on the discovered process model and decision rules.
    """
    
    def __init__(self, context: PDDLEncodingContext):
        self.context = context

    def collect_preconditions(self, action_name: str, variant_conditions: Optional[List[str]] = None) -> List[str]:
        """
        Collect preconditions for an action using the direct transition graph.
        
        Args:
            action_name: The name of the activity
            variant_conditions: Optional list of specific conditions to include for this variant
        """
        preconditions = set()

        # Always include enabled marker for the action
        preconditions.add(f"(enabled {action_name})")

        if action_name in self.context.parser.start_activities:
            return list(preconditions)
            
        # Find direct predecessors using the direct transition graph
        direct_predecessors = []
        for pred, successors in self.context.parser.direct_transition_graph.items():
            if action_name in successors:
                direct_predecessors.append(pred)
        
        if not direct_predecessors:
            return list(preconditions)
        # TODO return the params for the action
        action_attributes_param = get_attribute_params(self.context, action_name)
            
        # By default, include completed predecessor(s) as preconditions to follow
        # structural constraints. However, in minimal_preconditions mode we
        # skip adding predecessor completion requirements and instead rely on
        # the :init enabled markers being set by the evaluation (prefix).
        if not self.context.minimal_preconditions:
            # For now, require at least one direct predecessor to be completed
            # This represents OR logic at decision points
            if len(direct_predecessors) == 1:
                preconditions.add(f"(completed {direct_predecessors[0]})")
            else:
                #TODO: implement OR logic for multiple predecessors
                # Since PDDL doesn't have OR in preconditions directly, we'll use the first one as default
                # This could be improved with action variants for different paths
                preconditions.add(f"(completed {direct_predecessors[0]})")

        # Add attribute-based conditions (keep existing logic)

        attr_values = defaultdict(set)
        #iterate on dict that contains for each attr and value how likely a certain action will happen
        for attr, values in self.context.attr_activity_relationships.items():
            for val, activities in values.items():
                if action_name in activities:
                    stats = activities[action_name]
                    if stats['probability'] >= self.context.min_confidence:
                        if self.context.attribute_categories.get(attr) == 'boolean':
                            condition = f"({attr})" if val.lower() == 'true' else f"(not ({attr}))"
                            attr_values[attr].add(condition)
                        else:
                            if attr in action_attributes_param and action_name not in self.context.attribute_categories:
                                attr_values[attr].add(f"({attr} ?v)")
        
        for attr, values in attr_values.items():
            values_list = list(values)
            if len(values_list) == 1:
                preconditions.add(values_list[0])
            else:
                if variant_conditions:
                    for value_cond in values_list:
                        if value_cond in variant_conditions:
                            preconditions.add(value_cond)
                            break
                    else:
                        preconditions.add(values_list[0])
                else:
                    preconditions.add(values_list[0])
        
        return list(preconditions)

    def analyze_input_places(self, action_name: str) -> Dict[str, Any]:
        """
        Analyze the input places of a transition (activity) in the Petri net
        to determine the correct logical relationships for its preconditions.
        
        Returns:
            dict: A dictionary with 'AND' and 'OR' keys, where:
                  - 'AND' contains activities that must ALL be completed
                  - 'OR' contains groups of activities where ANY ONE in each group must be completed
        """
        transition = None
        for t in self.context.parser.transitions:
            if t.label is not None and utils.sanitize_name(t.label) == action_name:
                transition = t
                break
        if not transition:
            return {}
        
        input_places = []
        for arc in self.context.parser.edges:
            if isinstance(arc.source, pm4py.objects.petri_net.obj.PetriNet.Place) and \
            arc.target == transition:
                input_places.append(arc.source)
        
        or_groups = [] 
        and_inputs = set()
        for place in input_places:
            incoming_transitions = []
            for arc in self.context.parser.edges:
                if isinstance(arc.source, pm4py.objects.petri_net.obj.PetriNet.Transition) and \
                arc.target == place and arc.source.label is not None:
                    incoming_transitions.append(arc.source)
            incoming_transitions = [t for t in incoming_transitions if t.label is not None]
            
            if len(incoming_transitions) > 1:
                or_group = {utils.sanitize_name(t.label) for t in incoming_transitions}
                if or_group:
                    or_groups.append(or_group)
            elif len(incoming_transitions) == 1:
                for t in incoming_transitions:
                    if t.label is not None:
                        and_inputs.add(utils.sanitize_name(t.label))
        
        result = {}
        if and_inputs:
            result["AND"] = list(and_inputs)
        if or_groups:
            result["OR"] = or_groups
        
        return result

    def find_or_preconditions(self, action_name: str) -> List[List[str]]:
        """
        Find all OR conditions in the preconditions of an action.
        Returns a list of lists, where each inner list represents one OR condition.
        """
        if self.context.minimal_preconditions:
            return []
        
        or_conditions = []
        input_groups = self.analyze_input_places(action_name)
        if input_groups and "OR" in input_groups:
            for or_group in input_groups["OR"]:
                if len(or_group) > 1:
                    or_conditions.append([f"(completed {act})" for act in or_group])
        
        for attr, values in self.context.attr_activity_relationships.items():
            attr_or_values = []
            for val, activities in values.items():
                if action_name in activities:
                    stats = activities[action_name]
                    if stats['probability'] >= self.context.min_confidence:
                        if self.context.attribute_categories.get(attr) == 'boolean':
                            condition = f"({attr})" if val.lower() == 'true' else f"(not ({attr}))"
                            attr_or_values.append(condition)
                        else:
                            if attr in get_attribute_params(self.context, action_name):
                                attr_or_values.append(f"({attr} ?v)")
            if len(attr_or_values) > 1:
                or_conditions.append(attr_or_values)
        
        return or_conditions
