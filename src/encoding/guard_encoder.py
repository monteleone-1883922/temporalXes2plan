"""Utilities for converting Guard objects to PDDL predicate strings."""
from typing import List

from models import Guard


def guard_to_pddl(guard: Guard) -> str:
    """Convert a single Guard to its PDDL predicate string.

    Args:
        guard: A Guard produced by the decision tree mining phase.

    Returns:
        A PDDL predicate string ready to be used as a precondition.

    Examples:
        Guard("diagnosis", "flu",  negated=False) -> "(diagnosis_is flu)"
        Guard("diagnosis", "flu",  negated=True)  -> "(diagnosis_is_not flu)"
        Guard("urgent",    None,   negated=False) -> "(urgent_true)"
        Guard("urgent",    None,   negated=True)  -> "(urgent_false)"
    """
    if guard.value is None:
        suffix = "false" if guard.negated else "true"
        return f"({guard.attribute}_{suffix})"
    if guard.negated:
        return f"({guard.attribute}_is_not {guard.value})"
    return f"({guard.attribute}_is {guard.value})"


def and_clause_to_pddl(guards: List[Guard]) -> List[str]:
    """Convert a list of AND-ed guards to a list of PDDL predicate strings.

    Args:
        guards: A single AND-clause from a SOP guard structure.

    Returns:
        One PDDL string per guard, suitable for adding to a precondition list.
    """
    return [guard_to_pddl(g) for g in guards]
