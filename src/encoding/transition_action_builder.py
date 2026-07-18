"""Builds the PDDL action variants for one transition (3-axis Cartesian product).

A single PreparedTransition can expand into several PDDLBaseAction variants,
one per combination of:
    Axis A — OR clauses in the transition's structural preconditions
    Axis B — OR clauses in the XOR-branch conditions routing into it
    Axis C — one slot per OR clause in each effect group's guard
Contradictory combinations (e.g. Axis A asserting risk=high while Axis C's
guard asserts risk=low) are detected and pruned via Guard.first_conflict,
logging which two conditions (and from which axis) caused the skip.
"""
import dataclasses
import math
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import core_utils as utils
from encoding.effect_encoder import value_to_pddl_effects
from encoding.guard_encoder import and_clause_to_conditions
from encoding.pddl_model import (
    PDDLAction, PDDLBaseAction, PDDLCondition, PDDLDurativeAction, PDDLEffect,
)
from encoding.prepared_input import (
    PreparedEffectGroup, PreparedTransition, PreparedXorBranch, VariantInfo,
)
from models import AttributeCatalogEntry, Guard

logger = utils.get_logger(__name__)


def _dedupe_sop(sop: List[List[Guard]]) -> List[List[Guard]]:
    """Remove duplicate AND-clauses from a SOP (order-independent comparison)."""
    seen: Set[FrozenSet[Guard]] = set()
    result: List[List[Guard]] = []
    for clause in sop:
        key = frozenset(clause)
        if key in seen:
            continue
        seen.add(key)
        result.append(clause)
    return result


def _combine_xor_sop(
    sop_a: List[List[Guard]], sop_b: List[List[Guard]],
) -> List[List[Guard]]:
    """AND (Cartesian product) of two XOR-branch SOPs.

    Only collapses exact-duplicate merged clauses (via set); does NOT filter
    contradictions here — that check happens once, generically, in the main
    3-axis loop (_build_prepared_transition_actions), together with axis A and C.
    """
    if not sop_a:
        return sop_b
    if not sop_b:
        return sop_a
    combined = {
        frozenset(clause_a) | frozenset(clause_b)
        for clause_a in sop_a
        for clause_b in sop_b
    }
    return [list(c) for c in combined]


def _combine_xor_branches(
    xor_branches: Optional[List[PreparedXorBranch]],
) -> Tuple[List[List[Guard]], float]:
    """Combine every XOR branch feeding into one transition.

    A transition can be downstream of several independent XOR splits.
    Branches with DT guards combine via _combine_xor_sop (their conditions
    must all hold simultaneously — independent routing decisions along the
    same path). Branches with no guards (cascade_level 2/3) contribute only
    their probability, multiplied together (independent probabilistic
    discounts).

    Returns:
        (combined_conditions, combined_unguarded_probability).
        combined_conditions is [] when no branch carries DT guards.
    """
    if not xor_branches:
        return [], 1.0
    combined_sop: List[List[Guard]] = []
    combined_prob = 1.0
    for branch in xor_branches:
        if branch.conditions:
            combined_sop = _combine_xor_sop(combined_sop, _dedupe_sop(branch.conditions))
        else:
            combined_prob *= branch.probability
    return combined_sop, combined_prob


def _deduplicate_variants(
    variants: List[Tuple[PDDLBaseAction, VariantInfo]],
) -> List[Tuple[PDDLBaseAction, VariantInfo]]:
    """Remove duplicate variants, keeping the one with the highest effect_probability.

    Operates on (action, info) pairs so the VariantInfo metadata (§2.2 of
    claude_plans/trace_replayer_implementation_plan.md) survives dedup
    alongside the action it describes.
    """
    groups: Dict[tuple, List[Tuple[PDDLBaseAction, VariantInfo]]] = {}
    for v, info in variants:
        key = (
            frozenset(_get_preconditions(v)),
            frozenset(v.effect_attributes),
            frozenset(_get_effects(v)),
        )
        groups.setdefault(key, []).append((v, info))
    return [
        max(group, key=lambda pair: pair[0].effect_probability)
        for group in groups.values()
    ]


def _get_preconditions(action: PDDLBaseAction) -> Set[PDDLCondition]:
    # TODO: for durative actions the merge key is the union of all three condition
    # slots, so two actions with identical conditions distributed differently across
    # slots (e.g. X in at_start vs X in over_all) would be considered equivalent
    # and incorrectly merged.  Fix by using a per-slot tuple key instead.
    if isinstance(action, PDDLDurativeAction):
        return (
            action.conditions_at_start
            | action.conditions_over_all
            | action.conditions_at_end
        )
    if isinstance(action, PDDLAction):
        return action.preconditions
    return set()


def _get_effects(action: PDDLBaseAction) -> List[PDDLEffect]:
    if isinstance(action, PDDLDurativeAction):
        return action.effects_at_start + action.effects_at_end
    if isinstance(action, PDDLAction):
        return action.effects
    return []


def _build_prepared_transition_actions(
    act_name: str,
    transition: PreparedTransition,
    attribute_catalog: Dict[str, AttributeCatalogEntry],
    negated_attributes: Set[str],
    use_durative: bool,
    xor_branches: Optional[List[PreparedXorBranch]] = None,
    lower_bound_prob_actions: float = 1e-3,
    out_places: Optional[List[str]] = None,
) -> Tuple[List[PDDLBaseAction], Dict[str, VariantInfo]]:
    """Build every PDDLBaseAction variant for one transition.

    Enumerates the Cartesian product of axis A (preconditions) x axis B (XOR
    routing) x axis C (effect-group guards), skipping any combination that is
    internally contradictory, then deduplicates and names the survivors.

    Each surviving variant is a SINGLE PDDL action modeling the whole real
    Petri net transition firing: unmark input places, mark output places
    (out_places, the transition's real structural outputs — never a
    transition-as-place marker; standard Petri net semantics only ever
    marks places, see claude_plans/standard_petri_net_marking_plan.md) and
    apply the chosen effect group's attribute assignments, all together.
    For durative actions the unmark happens at_start and the mark+attribute
    effects at_end; for instantaneous PDDLAction both halves are one flat
    effects list.

    Returns:
        (variants, variant_map) where variant_map maps each survivor's final
        name to a VariantInfo capturing exactly which combined guard (axes
        A+B+C, as a single-clause SOP) and effect group produced it — the
        lookup table a plan replayer needs to identify "which effect group
        did the planner's chosen variant apply" by name alone, without
        regenerating this Cartesian product at runtime (see
        docs/trace_replayer_analysis.md §6.3).
    """
    in_places = transition.input_places
    out_places = out_places or []

    # Axis A — OR in preconditions, deduplicated
    prec_clauses: List[List[Guard]] = _dedupe_sop(transition.preconditions) or [[]]

    # Axis B — OR in XOR-branch conditions, combined across every branch this
    # transition is downstream of, deduplicated
    combined_conditions, unguarded_xor_prob = _combine_xor_branches(xor_branches)
    xor_clauses: List[List[Guard]] = _dedupe_sop(combined_conditions) or [[]]
    # Branches without a DT guard only carry a bare probability, encoded here
    # as an extra flat cost (no precondition to attach it to).
    xor_cost = 0.0
    if 0.0 < unguarded_xor_prob != 1.0:
        xor_cost = -math.log(max(unguarded_xor_prob, lower_bound_prob_actions))

    # Axis C — one slot per OR clause in each effect group's guard, clauses
    # deduplicated per-group (groups remain alternatives to each other, not
    # deduplicated against one another: two groups can share an identical
    # guard while assigning different attribute values)
    effect_axis: List[Tuple[Optional[PreparedEffectGroup], Optional[List[Guard]]]] = []
    for group in transition.effect_groups:
        guard = _dedupe_sop(group.guard) if group.guard else []
        if not guard:
            effect_axis.append((group, None))
        else:
            for and_clause in guard:
                effect_axis.append((group, and_clause))
    if not effect_axis:
        effect_axis = [(None, None)]

    base_cost = transition.cost + xor_cost
    variants: List[Tuple[PDDLBaseAction, VariantInfo]] = []

    _AXIS_A = "axis A (structural preconditions)"
    _AXIS_B = "axis B (XOR branch)"
    _AXIS_C = "axis C (effect_group guard)"

    def _log_and_skip(labeled_guards: List[Tuple[Guard, str]]) -> bool:
        """Return True (and log) if labeled_guards contains a conflict."""
        conflict = Guard.first_conflict(g for g, _ in labeled_guards)
        if conflict is None:
            return False
        g1, g2 = conflict
        l1 = next(label for guard, label in labeled_guards if guard == g1)
        l2 = next(label for guard, label in labeled_guards if guard == g2)
        logger.debug(
            "Skipping action combination for '%s': %s (%s) conflicts with %s (%s)",
            act_name, g1, l1, g2, l2,
        )
        return True

    for p_clause in prec_clauses:
        p_labeled = [(g, _AXIS_A) for g in p_clause]
        if _log_and_skip(p_labeled):
            continue  # axis A alone is already inconsistent

        for x_clause in xor_clauses:
            x_labeled = [(g, _AXIS_B) for g in x_clause]
            px_labeled = p_labeled + x_labeled
            if _log_and_skip(px_labeled):
                continue  # pruned here: avoids iterating all of axis C for nothing

            for (eff_group, guard_clause) in effect_axis:
                c_labeled = [(g, _AXIS_C) for g in (guard_clause or [])]
                combined_labeled = px_labeled + c_labeled
                if _log_and_skip(combined_labeled):
                    continue

                combined_guards = frozenset(g for g, _ in combined_labeled)
                preconds: Set[PDDLCondition] = {PDDLCondition.marked(p) for p in in_places}
                preconds.update(and_clause_to_conditions(list(combined_guards)))

                cost = base_cost
                # An effect group without its own guard is statistical — its
                # probability wasn't explained by any precondition above, so
                # it's charged here as a -log(probability) cost instead.
                if eff_group is not None and guard_clause is None:
                    eff_prob = eff_group.probability
                    if 0.0 < eff_prob != 1.0:
                        cost += -math.log(max(eff_prob, lower_bound_prob_actions))

                start_effects: List[PDDLEffect] = [PDDLEffect.unmarking(p) for p in in_places]
                end_effects: List[PDDLEffect] = [PDDLEffect.marking(p) for p in sorted(out_places)]
                if eff_group is not None:
                    for attr, value in eff_group.assignments:
                        catalog_entry = attribute_catalog[attr]
                        end_effects.extend(value_to_pddl_effects(
                            attr, value, catalog_entry, negated_attributes,
                        ))

                duration = transition.duration
                # Placeholder name — the real name is assigned after
                # deduplication below, since the survivor count is only known
                # once conflicting/duplicate combinations have been skipped.
                if use_durative and duration is not None:
                    action: PDDLBaseAction = PDDLDurativeAction(
                        name=f"execute_{act_name}",
                        duration_min=duration.effective_min,
                        duration_max=duration.effective_max,
                        conditions_at_start=preconds,
                        effects_at_start=start_effects,
                        effects_at_end=end_effects,
                        base_cost=cost if cost != 0.0 else None,
                    )
                else:
                    action = PDDLAction(
                        name=f"execute_{act_name}",
                        preconditions=preconds,
                        effects=start_effects + end_effects,
                        base_cost=cost if cost != 0.0 else None,
                    )
                info = VariantInfo(
                    activity_name=act_name,
                    preconditions=[list(combined_guards)],
                    effect_group=eff_group,
                )
                variants.append((action, info))

    variants = _deduplicate_variants(variants)
    # Only give variants a "_v{i}" suffix when there's more than one — a
    # single surviving variant keeps the plain "execute_{act_name}" name.
    if len(variants) == 1:
        action, info = variants[0]
        named = [(dataclasses.replace(action, name=f"execute_{act_name}"), info)]
    else:
        named = [
            (dataclasses.replace(action, name=f"execute_{act_name}_v{i}"), info)
            for i, (action, info) in enumerate(variants)
        ]

    variant_map = {action.name: info for action, info in named}
    return [action for action, _ in named], variant_map
