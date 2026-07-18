"""PDDL static schema (types, constants, predicates) for build_domain_from_prepared_info.

These three pieces only depend on the attribute catalog and the graph's
places/transitions — not on any transition's preconditions/effects — so they
are built once, up front, independently of the per-transition action loop.
"""
from typing import Dict, Set

import core_utils
from encoding.pddl_model import (
    PDDLConstants, PDDLObject, PDDLPredicate, PDDLPredicates, PDDLType, PDDLTypes,
)
from encoding.prepared_input import PreparedDomainInput, PLACE_NODE_TYPES
from models import AttributeCatalogEntry


def _prepared_types(attribute_catalog: Dict[str, AttributeCatalogEntry]) -> PDDLTypes:
    """Build the PDDL type hierarchy: petri_element -> place, plus one
    value-type per categorical/numerical attribute (e.g. "risk_val").

    Transitions are never PDDL objects: standard Petri net semantics only
    ever marks places (see claude_plans/standard_petri_net_marking_plan.md)
    -- "petri_element" exists purely so "marked" can stay a single generic
    predicate rather than one per place-producing context.
    """
    types = PDDLTypes(items=[
        PDDLType("petri_element"),
        PDDLType("place", parent="petri_element"),
    ])
    for attr_name, entry in sorted(attribute_catalog.items()):
        if entry.attribute_type in ("categorical", "numerical"):
            types.add(PDDLType(f"{attr_name}_val"))
    return types


def _prepared_constants(prepared: PreparedDomainInput) -> PDDLConstants:
    """Build the PDDL constants: one object per place (never per
    transition -- see _prepared_types), and per possible value of each
    categorical/numerical attribute."""
    constants = PDDLConstants()

    for label in sorted(
        n.label for n in prepared.nodes if n.type in PLACE_NODE_TYPES
    ):
        constants.add(PDDLObject(label, "place"))

    for attr_name, entry in sorted(prepared.attribute_catalog.items()):
        if entry.attribute_type in ("categorical", "numerical"):
            type_name = f"{attr_name}_val"
            for val in sorted(str(v) for v in entry.possible_values):
                constants.add(PDDLObject(core_utils.sanitize_value(attr_name, val), type_name))

    return constants


def _prepared_predicates(
    attribute_catalog: Dict[str, AttributeCatalogEntry],
    negated_attributes: Set[str],
) -> PDDLPredicates:
    """Build the PDDL predicates: "marked", plus one is/true predicate per
    attribute (and its negative counterpart, only when actually needed)."""
    predicates = PDDLPredicates()
    predicates.add(PDDLPredicate("marked", [("?x", "petri_element")]))

    for attr_name, entry in sorted(attribute_catalog.items()):
        if entry.attribute_type in ("categorical", "numerical"):
            type_name = f"{attr_name}_val"
            predicates.add(PDDLPredicate(f"{attr_name}_is", [("?v", type_name)]))
            if attr_name in negated_attributes:
                predicates.add(PDDLPredicate(f"{attr_name}_is_not", [("?v", type_name)]))
        elif entry.attribute_type == "boolean":
            predicates.add(PDDLPredicate(f"{attr_name}_true"))
            if attr_name in negated_attributes:
                predicates.add(PDDLPredicate(f"{attr_name}_false"))

    return predicates
