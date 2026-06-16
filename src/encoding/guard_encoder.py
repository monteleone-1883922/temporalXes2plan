"""Utilities for converting Guard objects to PDDLCondition instances."""
from typing import List

from models import Guard
from encoding.pddl_model import PDDLCondition


def guard_to_condition(guard: Guard) -> PDDLCondition:
    """Convert a single Guard to a structured PDDLCondition.

    Args:
        guard: A Guard produced by the decision tree mining phase.

    Returns:
        A PDDLCondition ready to be added to an action's precondition set.

    Examples:
        Guard("diagnosis", "flu",  negated=False) → PDDLCondition.attr_is("diagnosis", "flu")
        Guard("diagnosis", "flu",  negated=True)  → PDDLCondition.attr_is_not("diagnosis", "flu")
        Guard("urgent",    None,   negated=False) → PDDLCondition.attr_true("urgent")
        Guard("urgent",    None,   negated=True)  → PDDLCondition.attr_false("urgent")
    """
    if guard.value is None:
        if guard.negated:
            return PDDLCondition.attr_false(guard.attribute)
        return PDDLCondition.attr_true(guard.attribute)
    if guard.negated:
        return PDDLCondition.attr_is_not(guard.attribute, guard.value)
    return PDDLCondition.attr_is(guard.attribute, guard.value)


def and_clause_to_conditions(guards: List[Guard]) -> List[PDDLCondition]:
    """Convert a list of AND-ed guards to a list of PDDLCondition instances.

    Args:
        guards: A single AND-clause from a SOP guard structure.

    Returns:
        One PDDLCondition per guard, suitable for adding to a precondition set.
    """
    return [guard_to_condition(g) for g in guards]
