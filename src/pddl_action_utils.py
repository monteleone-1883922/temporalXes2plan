from typing import Set, List, Optional, Dict
from encoder_models import PDDLEncodingContext

def is_tau_activity(activity_name: str) -> bool:
    """Check if an activity is a silent (tau) transition."""
    return activity_name.startswith('tau_')

def get_attribute_params(context: PDDLEncodingContext, action_name: str) -> Set[str]:
    """
    Determine which attributes should be parameters for an action.
    """
    param_attributes = set()
    
    # Check if this is a measurement action variant (like crp_high, leucocytes_low)
    # These should NOT have parameters as they represent specific observations
    discretized_suffixes = ['_very_low', '_low', '_medium', '_high', '_true', '_false']
    if any(action_name.endswith(suffix) for suffix in discretized_suffixes):
        return param_attributes  # Return empty set - no parameters needed
    
    # Also check for categorical value suffixes (for attributes like diagnose)
    sanitized_action = action_name.lower()
    for attr in context.attribute_categories:
        if action_name.startswith(attr + '_'):
            # This is likely a measurement variant like crp_high or diagnose_a
            return param_attributes  # No parameters needed
        # Only add parameters for actions that manipulate attribute values generically,
        # not for specific measurement actions
        if attr.lower() == sanitized_action and context.attribute_categories[attr] != 'boolean' and \
                not any(f"{attr}_{suffix}" in context.activities for suffix in
                        ['low', 'medium', 'high', 'very_low']):
            # Skip if this looks like a measurement action (has variants)
            param_attributes.add(attr)
            return param_attributes
    
    # Check for actions that introduce new attribute values
    if action_name in context.activity_attr_effects:
        for attr, from_values in context.activity_attr_effects[action_name].items():
            # Only add parameter if this is not a measurement action setting specific values
            if 'NEW' in from_values and context.attribute_categories.get(attr) != 'boolean' and \
                    not any(
                        any(
                            from_val != 'NEW' and to_vals for from_val, to_vals in attrs[attr].items()
                        )
                            for act_name, attrs in context.activity_attr_effects.items() if attr in attrs
                    ):
                param_attributes.add(attr)
    return param_attributes

def get_successors(context: PDDLEncodingContext, action_name: str) -> List[str]:
    """Get successors using the direct transition graph."""
    return context.parser.direct_transition_graph.get(action_name, [])

def get_decision_point_for_activity(context: PDDLEncodingContext, action_name: str) -> Optional[Dict[str, float]]:
    """
    Retrieve the branch probabilities for an activity if it follows a decision point.
    """
    if action_name in context.processed_decision_points:
        return context.processed_decision_points[action_name]
    return None
