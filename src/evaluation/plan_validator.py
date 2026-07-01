"""Forward-simulate a plan against the serialized Petri net to validate it.

The validator replays each action in the plan in two layers:

1. **Token flow** (always): every input place must hold a token; tokens are moved
   from input to output places.  Silent actions (tau_*, mark_places_from_*) are
   skipped.

2. **Attribute state** (when ``variant_effects`` is supplied): the exact attribute
   effects of each variant action are applied and preconditions are checked.
   OPTIC writes the full variant name (e.g. ``exec_register_v1``) in the plan,
   so the lookup is exact — no guesswork.  When ``variant_effects`` is ``None``
   the attribute layer is skipped entirely and ``attribute_checked`` is ``False``.

Both layers run in a single forward pass over the plan steps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from encoding.action_registry import ActionRegistry


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Result of a plan validation run.

    Attributes:
        valid: True if the plan passed all checks that were run.
        attribute_checked: True when variant_effects was supplied and
            precondition / effect tracking ran with exact variant data.
            False when only the token-flow layer ran.
        error_step: 0-based index of the first failing action, or None.
        error_action: Base name of the first failing action, or None.
        error_reason: Human-readable description of the failure, or None.
        final_places: Token marking after the last successfully executed action.
        final_attributes: Attribute state after the last action (populated only
            when attribute_checked is True).
        steps_executed: Non-silent actions successfully executed before failure
            (or total if valid).
    """

    valid: bool
    attribute_checked: bool
    error_step: Optional[int]
    error_action: Optional[str]
    error_reason: Optional[str]
    final_places: Set[str]
    final_attributes: Dict[str, str]
    steps_executed: int


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_plan(
    plan_steps: List[str],
    init_places: List[str],
    serialized: Dict[str, Any],
    init_attributes: Optional[List[Dict[str, Any]]] = None,
    variant_effects: Optional[Dict[str, Dict[str, str]]] = None,
) -> ValidationReport:
    """Simulate a plan forward against the serialized Petri net.

    Args:
        plan_steps: Ordered list of action name strings as returned by the
            planner (may include exec_ prefix and _vN version suffix).
        init_places: Initial token marking (list of place names).
        serialized: current.json dict (output of serialize_parse_result).
        init_attributes: Optional list of {"attribute": str, "value": str}
            dicts representing the initial attribute state.  Only used when
            ``variant_effects`` is provided.
        variant_effects: Optional dict mapping the full planner action name
            (e.g. ``"exec_register_v1"``) to a flat ``{attr: value}`` dict of
            the attribute effects that variant applies.  Built by
            ``EvalAPI.build_domain_with_variant_effects()``.  When ``None``,
            only the token-flow layer runs.

    Returns:
        ValidationReport describing the outcome of the simulation.
    """
    transitions: Dict[str, Any] = serialized.get("transitions", {})

    marking: Set[str] = set(init_places)
    attributes: Dict[str, str] = {}
    if variant_effects is not None and init_attributes:
        for item in init_attributes:
            attributes[item["attribute"]] = str(item["value"])

    non_silent_executed = 0

    for step_idx, raw_action in enumerate(plan_steps):
        if _is_silent(raw_action):
            continue

        # Token-flow lookup uses the base name (strips exec_ and _vN).
        base_name = _normalise_token_flow(raw_action)
        t_info = transitions.get(base_name)

        if t_info is None:
            return ValidationReport(
                valid=False,
                attribute_checked=variant_effects is not None,
                error_step=step_idx,
                error_action=base_name,
                error_reason=f"Action '{base_name}' not found in transition catalog.",
                final_places=marking,
                final_attributes=attributes,
                steps_executed=non_silent_executed,
            )

        input_places: List[str] = t_info.get("input_places", [])
        output_places: List[str] = t_info.get("output_places", [])

        # Layer 1: token availability.
        missing = [p for p in input_places if p not in marking]
        if missing:
            return ValidationReport(
                valid=False,
                attribute_checked=variant_effects is not None,
                error_step=step_idx,
                error_action=base_name,
                error_reason=(
                    f"Action '{base_name}' requires tokens in {missing}, "
                    f"but current marking is {sorted(marking)}."
                ),
                final_places=marking,
                final_attributes=attributes,
                steps_executed=non_silent_executed,
            )

        # Layer 2: attribute preconditions (exact, only when variant_effects supplied).
        if variant_effects is not None:
            preconditions: List[List[Dict[str, Any]]] = t_info.get("preconditions", [])
            if preconditions and not _check_preconditions(preconditions, attributes):
                return ValidationReport(
                    valid=False,
                    attribute_checked=True,
                    error_step=step_idx,
                    error_action=base_name,
                    error_reason=(
                        f"Action '{base_name}' preconditions not satisfied. "
                        f"Current attributes: {attributes}."
                    ),
                    final_places=marking,
                    final_attributes=attributes,
                    steps_executed=non_silent_executed,
                )

        # Fire transition: move tokens.
        marking -= set(input_places)
        marking |= set(output_places)

        # Apply exact attribute effects (raw_action key preserves _vN suffix).
        if variant_effects is not None:
            attributes.update(variant_effects.get(raw_action, {}))

        non_silent_executed += 1

    return ValidationReport(
        valid=True,
        attribute_checked=variant_effects is not None,
        error_step=None,
        error_action=None,
        error_reason=None,
        final_places=marking,
        final_attributes=attributes,
        steps_executed=non_silent_executed,
    )


# ---------------------------------------------------------------------------
# Transition lookup
# ---------------------------------------------------------------------------

def resolve_action_to_transition(
    action_name: str,
    serialized: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Look up an action name in the transition catalog using the base name.

    Args:
        action_name: Raw action name from the planner output (may include
            exec_ prefix and _vN suffix).
        serialized: current.json dict.

    Returns:
        Transition info dict, or None if not found.
    """
    transitions: Dict[str, Any] = serialized.get("transitions", {})
    return transitions.get(_normalise_token_flow(action_name))


# ---------------------------------------------------------------------------
# Variant effects extraction
# ---------------------------------------------------------------------------

def build_variant_effects_lookup(
    registry: "ActionRegistry",
) -> Dict[str, Dict[str, str]]:
    """Build a lookup from full variant action name to its attribute effects.

    Extracts positive (non-clear) attribute-setting effects from each
    PDDLBaseAction variant in the registry.

    Args:
        registry: ActionRegistry returned by DomainBuilder.build_with_registry().

    Returns:
        Dict mapping full variant name (e.g. "exec_register_v1") to a flat
        {attribute: value} dict of effects applied by that variant.
        Variants with no attribute effects map to an empty dict.
    """
    lookup: Dict[str, Dict[str, str]] = {}
    for variants in registry._variants.values():
        for variant in variants:
            effects: Dict[str, str] = {}
            for eff in variant.effects:
                if eff.clear:
                    continue
                if eff.kind == "attr_is" and eff.value is not None:
                    effects[eff.attribute] = eff.value
                elif eff.kind == "attr_true":
                    effects[eff.attribute] = "true"
                elif eff.kind == "attr_false":
                    effects[eff.attribute] = "false"
            lookup[variant.name] = effects
    return lookup


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SILENT_PREFIXES = ("tau_", "mark_places_from_")


def _is_silent(action_name: str) -> bool:
    return any(action_name.startswith(p) for p in _SILENT_PREFIXES)


def _normalise_token_flow(name: str) -> str:
    """Strip exec_ prefix and version/DETDUP suffixes for token-flow lookup.

    Used only to locate the base transition in serialized["transitions"].
    The raw action name is preserved separately for the variant-effects lookup.

    Transformations applied in order:
    1. Strip leading ``exec_`` prefix.
    2. Strip trailing ``_DETDUP`` suffix and everything after it.
    3. Strip trailing ``_v<digits>`` version suffix.

    Args:
        name: Raw action name string from the planner.

    Returns:
        Base transition name suitable for looking up in serialized["transitions"].
    """
    if name.startswith("exec_"):
        name = name[5:]
    name = re.sub(r"_DETDUP.*$", "", name)
    name = re.sub(r"_v\d+$", "", name)
    return name


def _check_preconditions(
    preconditions: List[List[Dict[str, Any]]],
    attributes: Dict[str, str],
) -> bool:
    """Evaluate a SOP precondition against the current attribute state.

    Args:
        preconditions: List of AND-clauses (OR of ANDs).
        attributes: Current attribute → value mapping.

    Returns:
        True if at least one AND-clause is fully satisfied.
    """
    for and_clause in preconditions:
        if _check_and_clause(and_clause, attributes):
            return True
    return False


def _check_and_clause(
    conditions: List[Dict[str, Any]],
    attributes: Dict[str, str],
) -> bool:
    for cond in conditions:
        attr = cond["attribute"]
        predicate = cond.get("predicate", "=")
        value = str(cond.get("value", ""))
        current = attributes.get(attr)

        if predicate == "=":
            if current != value:
                return False
        else:  # "<>"
            if current == value:
                return False
    return True
