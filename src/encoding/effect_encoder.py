"""Standalone helpers for generating PDDLEffect instances from attribute values."""
from typing import Any, List, Set

from models import AttributeCatalogEntry
from encoding.pddl_model import PDDLEffect


def boolean_effect(
    attr_name: str, value: Any, negated_attributes: Set[str]
) -> List[PDDLEffect]:
    """PDDLEffect list for a boolean attribute change.

    Args:
        attr_name: Sanitized attribute name.
        value: The boolean value (True/False or string equivalent).
        negated_attributes: Attributes that need negative predicates.

    Returns:
        List of PDDLEffect objects encoding the state change.
    """
    needs_negative = attr_name in negated_attributes
    if value is True or str(value).lower() == "true":
        effects = [PDDLEffect.set_attr_true(attr_name)]
        if needs_negative:
            effects.append(PDDLEffect.clear_attr_false(attr_name))
    else:
        if needs_negative:
            effects = [
                PDDLEffect.set_attr_false(attr_name),
                PDDLEffect.clear_attr_true(attr_name),
            ]
        else:
            effects = [PDDLEffect.clear_attr_true(attr_name)]
    return effects


def categorical_effect(
    attr_name: str,
    value: Any,
    catalog_entry: AttributeCatalogEntry,
    negated_attributes: Set[str],
) -> List[PDDLEffect]:
    """PDDLEffect list for a categorical/numerical attribute change.

    Args:
        attr_name: Sanitized attribute name.
        value: The categorical value to set.
        catalog_entry: Catalog entry with possible_values.
        negated_attributes: Attributes that need negative predicates.

    Returns:
        List of PDDLEffect objects encoding the state change.
    """
    val_str = str(value)
    needs_negative = attr_name in negated_attributes
    effects: List[PDDLEffect] = [PDDLEffect.set_attr_is(attr_name, val_str)]

    for other_val in sorted(str(v) for v in catalog_entry.possible_values):
        if other_val == val_str:
            if needs_negative:
                effects.append(PDDLEffect.clear_attr_is_not(attr_name, other_val))
        else:
            if needs_negative:
                effects.append(PDDLEffect.set_attr_is_not(attr_name, other_val))
            effects.append(PDDLEffect.clear_attr_is(attr_name, other_val))

    return effects


def value_to_pddl_effects(
    attr_name: str,
    value: Any,
    catalog_entry: AttributeCatalogEntry,
    negated_attributes: Set[str],
) -> List[PDDLEffect]:
    """Generate PDDLEffect instances for an attribute value assignment.

    Dispatches to boolean_effect or categorical_effect based on the
    catalog entry's attribute_type.

    Args:
        attr_name: Sanitized attribute name.
        value: The value to assign.
        catalog_entry: Catalog entry for the attribute.
        negated_attributes: Attributes requiring negative predicates.

    Returns:
        List of PDDLEffect objects.
    """
    if catalog_entry.attribute_type == "boolean":
        return boolean_effect(attr_name, value, negated_attributes)
    return categorical_effect(attr_name, value, catalog_entry, negated_attributes)
