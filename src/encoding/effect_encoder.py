"""Standalone helpers for generating PDDL effect strings from attribute values."""
from typing import Any, List, Set

from models import AttributeCatalogEntry


def boolean_effect(
    attr_name: str, value: Any, negated_attributes: Set[str]
) -> List[str]:
    """PDDL effects for a boolean attribute change.

    Args:
        attr_name: Sanitized attribute name.
        value: The boolean value (True/False or string equivalent).
        negated_attributes: Attributes that need negative predicates.

    Returns:
        List of PDDL effect strings.
    """
    needs_negative = attr_name in negated_attributes
    if value is True or str(value).lower() == "true":
        effects = [f"({attr_name}_true)"]
        if needs_negative:
            effects.append(f"(not ({attr_name}_false))")
    else:
        if needs_negative:
            effects = [f"({attr_name}_false)", f"(not ({attr_name}_true))"]
        else:
            effects = [f"(not ({attr_name}_true))"]
    return effects


def categorical_effect(
    attr_name: str,
    value: Any,
    catalog_entry: AttributeCatalogEntry,
    negated_attributes: Set[str],
) -> List[str]:
    """PDDL effects for a categorical/numerical attribute change.

    Args:
        attr_name: Sanitized attribute name.
        value: The categorical value to set.
        catalog_entry: Catalog entry with possible_values.
        negated_attributes: Attributes that need negative predicates.

    Returns:
        List of PDDL effect strings.
    """
    val_str = str(value)
    needs_negative = attr_name in negated_attributes
    effects = [f"({attr_name}_is {val_str})"]

    for other_val in sorted(str(v) for v in catalog_entry.possible_values):
        if other_val == val_str:
            if needs_negative:
                effects.append(f"(not ({attr_name}_is_not {other_val}))")
        else:
            if needs_negative:
                effects.append(f"({attr_name}_is_not {other_val})")
            effects.append(f"(not ({attr_name}_is {other_val}))")

    return effects


def value_to_pddl_effects(
    attr_name: str,
    value: Any,
    catalog_entry: AttributeCatalogEntry,
    negated_attributes: Set[str],
) -> List[str]:
    """Generate PDDL effect strings for an attribute value assignment.

    Dispatches to boolean_effect or categorical_effect based on the
    catalog entry's attribute_type.

    Args:
        attr_name: Sanitized attribute name.
        value: The value to assign.
        catalog_entry: Catalog entry for the attribute.
        negated_attributes: Attributes requiring negative predicates.

    Returns:
        List of PDDL effect strings.
    """
    if catalog_entry.attribute_type == "boolean":
        return boolean_effect(attr_name, value, negated_attributes)
    return categorical_effect(attr_name, value, catalog_entry, negated_attributes)
