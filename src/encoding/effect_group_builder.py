"""Derives PreparedEffectGroup combinations directly from TransitionInfo.effects.

Ports the same combinatorial semantics as the old EffectDuplicator (appearance
phase, then value phase, per attribute), but works on abstract (attribute,
value) descriptors instead of mutating PDDLBaseAction variants, since this
runs before any PDDL action exists — it builds the INPUT to the shared build
engine (encoding/transition_action_builder.py), not its output.
"""
import dataclasses
from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple

import core_utils as utils
from encoding.prepared_input import PreparedEffectGroup
from encoding.transition_action_builder import _dedupe_sop
from models import EffectInfo, Guard, TransitionInfo

logger = utils.get_logger(__name__)


def _is_fully_deterministic(effect: EffectInfo) -> bool:
    """True when an effect always appears with exactly one possible value.

    Such effects need no branching at all: the (attribute, value) pair is
    added unconditionally to every partial group.
    """
    if effect.appearance_level != 1 or effect.value_level != 1:
        return False
    if effect.appearance_guards is not None or effect.value_guards is not None:
        return False
    return len(effect.value_probabilities) == 1


@dataclasses.dataclass
class _PartialGroup:
    """One in-progress (assignments, guard, probability) branch while building
    PreparedEffectGroup combinations in _prepared_effect_groups.

    guard_clause is a single accumulated AND-clause, not a full SOP: OR
    branching across DT guard clauses is represented by multiple separate
    _PartialGroup entries in the working list. They are only merged into one
    PreparedEffectGroup (with a real multi-clause SOP guard) at the end, when
    several partial groups happen to end up with identical assignments.
    """
    assignments: List[Tuple[str, str]]
    guard_clause: List[Guard]
    probability: float


def _prepared_has_incompatible(
    info: TransitionInfo, attr_name: str, present_attrs: Set[str]
) -> bool:
    """True when attr_name is marked incompatible with an attribute already
    present in this partial group (info.incompatible_effects), meaning
    attr_name must NOT appear alongside it."""
    for pair in info.incompatible_effects:
        pair_set = set(pair)
        if attr_name in pair_set and (pair_set - {attr_name}) & present_attrs:
            return True

    return False


def _prepared_has_related(
    info: TransitionInfo, attr_name: str, present_attrs: Set[str],
) -> bool:
    """True when attr_name is marked related to an attribute already present
    in this partial group (info.related_effects), meaning attr_name is forced
    to appear whenever that other attribute does (they co-occur)."""
    for pair in info.related_effects:
        pair_set = set(pair)
        if attr_name in pair_set and (pair_set - {attr_name}) & present_attrs:
            return True
    return False


def _prepared_appearance_phase(
    info: TransitionInfo,
    attr_name: str,
    effect: EffectInfo,
    groups: List[_PartialGroup],
    track_cost: bool,
    log_and_skip,
) -> Tuple[List[_PartialGroup], List[_PartialGroup]]:
    """Returns (with_effect, without_effect) partial-group lists."""
    # always appear but not always with same value
    if effect.appearance_level == 1 and effect.appearance_guards is None:
        with_eff: List[_PartialGroup] = []
        without: List[_PartialGroup] = []
        for g in groups:
            present = {a for a, _ in g.assignments}
            # since it has to appear always, we only add to without if there are incompatible assignments
            if _prepared_has_incompatible(info, attr_name, present):
                without.append(g)
            else:
                with_eff.append(g)
        return with_eff, without

    # conditionally appear
    if effect.appearance_level == 1 and effect.appearance_guards is not None:
        appears_clauses = effect.appearance_guards.guards.get("appears", [])
        if not appears_clauses:
            #TODO: analyze this case
            return [], list(groups)
        with_eff = []
        without = []
        for g in groups:
            present = {a for a, _ in g.assignments}
            # if at least 1 incompatible in the group then it cannot appear
            if _prepared_has_incompatible(info, attr_name, present):
                without.append(g)
                continue
            #checks if at least 1 in that group is strongly related to the attribute (they often appear together)
            forced_related = _prepared_has_related(info, attr_name, present)
            # if there isn't a related attribute in the group, then it is possible to have that group with the attribute
            if not forced_related:
                without.append(g)
            # since attribute has guards in sop then we need a group for each end clause
            for clause in appears_clauses:
                if log_and_skip(info.activity_name, g.guard_clause, clause):
                    continue
                with_eff.append(_PartialGroup(
                    assignments=list(g.assignments),
                    guard_clause=g.guard_clause + list(clause),
                    probability=g.probability,
                ))
        return with_eff, without

    # probably appear
    if effect.appearance_level == 2:
        with_eff = []
        without = []
        for g in groups:
            present = {a for a, _ in g.assignments}
            if _prepared_has_incompatible(info, attr_name, present):
                without.append(g)
                continue
            with_prob = g.probability
            without_prob = g.probability
            if track_cost:
                # Statistical split: no guard explains this decision, so the
                # branching probability itself becomes the group's weight
                # (later turned into a -log(probability) cost).
                with_prob = g.probability * effect.presence_probability
                without_prob = g.probability * (1 - effect.presence_probability)
            with_eff.append(_PartialGroup(
                assignments=list(g.assignments), guard_clause=list(g.guard_clause), probability=with_prob,
            ))
            without.append(_PartialGroup(
                assignments=list(g.assignments), guard_clause=list(g.guard_clause), probability=without_prob,
            ))
        return with_eff, without

    # appearance_level == 3 (or unrecognised): no data -> effect never applied
    return [], list(groups)


def _prepared_value_phase(
    info: TransitionInfo,
    attr_name: str,
    effect: EffectInfo,
    groups: List[_PartialGroup],
    track_cost: bool,
    log_and_skip,
) -> List[_PartialGroup]:
    """Assigns a value to attr_name for every group that got the effect in
    the appearance phase (groups here are already "with effect")."""
    if not groups:
        return []

    # deterministic value (almost always that attribute change its value to this)
    if effect.value_level == 1 and effect.value_guards is None:
        value, _ = max(effect.value_probabilities.items(), key = lambda val_prob: val_prob[1], default=(None, 0.0) )
        if value:
            for g in groups:
                g.assignments.append((attr_name, value))
            return groups
        else:
            logger.warning("No value found for attribute %s. This attribute is claimed to have a deterministic value", attr_name)

    # conditionally assume the value
    if effect.value_level == 1 and effect.value_guards is not None:
        result: List[_PartialGroup] = []
        for g in groups:
            for value_str, or_clauses in sorted(effect.value_guards.guards.items()):
                # for values on which the dt failed to find reliable guards and the fallback policy was used
                if not or_clauses:
                    prob = effect.value_probabilities[value_str]
                    probability = g.probability * prob if track_cost else g.probability
                    result.append(_PartialGroup(
                        assignments=list(g.assignments) + [(attr_name, value_str)],
                        guard_clause=g.guard_clause,
                        probability=probability,
                    ))
                for clause in or_clauses:
                    if log_and_skip(info.activity_name, g.guard_clause, clause):
                        continue
                    result.append(_PartialGroup(
                        assignments=list(g.assignments) + [(attr_name, value_str)],
                        guard_clause=g.guard_clause + list(clause),
                        probability=g.probability,
                    ))
        return result

    if effect.value_level == 2:
        result = []
        for g in groups:
            for value, prob in sorted(effect.value_probabilities.items(), key=lambda kv: str(kv[0])):
                probability = g.probability * prob if track_cost else g.probability
                result.append(_PartialGroup(
                    assignments=list(g.assignments) + [(attr_name, value)],
                    guard_clause=list(g.guard_clause),
                    probability=probability,
                ))
        return result

    # value_level == 3 / no data: attribute "appears" but no value assigned;
    # drop the assignment for this attribute, leave group otherwise unchanged.
    return groups


def _prepared_effect_groups(
    info: TransitionInfo,
    track_appearance_cost: bool = True,
    track_value_cost: bool = True,
) -> List[PreparedEffectGroup]:
    """Derive PreparedEffectGroup combinations directly from TransitionInfo.effects.

    Ports the same combinatorial semantics as EffectDuplicator
    (encoding/effect_duplicator.py), but works on abstract (attribute, value)
    descriptors instead of mutating PDDLBaseAction variants, since this runs
    before any PDDL action exists (it builds the INPUT to the shared build
    engine, not its output).
    """
    # Start from a single "empty" group and progressively branch/annotate it
    # one attribute at a time (sorted for determinism across runs).
    groups: List[_PartialGroup] = [_PartialGroup(assignments=[], guard_clause=[], probability=1.0)]

    def _log_and_skip(act_name: str, existing: List[Guard], new_clause: List[Guard]) -> bool:
        """Return True (and log) if combining existing+new_clause would be
        internally contradictory — used to prune impossible branches early."""
        labeled = [(g, "existing") for g in existing] + [(g, "new") for g in new_clause]
        conflict = Guard.first_conflict(g for g, _ in labeled)
        if conflict is None:
            return False
        g1, g2 = conflict
        l1 = next(label for guard, label in labeled if guard == g1)
        l2 = next(label for guard, label in labeled if guard == g2)
        logger.debug(
            "Skipping effect-group branch for '%s': %s (%s) conflicts with %s (%s)",
            act_name, g1, l1, g2, l2,
        )
        return True

    for attr_name, effect in sorted(info.effects.items()):
        if _is_fully_deterministic(effect):
            # No branching needed: add the single possible value to every
            # surviving group and move on to the next attribute.
            value = next(iter(effect.value_probabilities.keys()))
            for g in groups:
                g.assignments.append((attr_name, value))
            continue

        with_groups, without_groups = _prepared_appearance_phase(
            info, attr_name, effect, groups, track_appearance_cost, _log_and_skip
        )
        valued = _prepared_value_phase(
            info, attr_name, effect, with_groups, track_value_cost, _log_and_skip
        )
        # "without" groups never got the attribute, so they pass through
        # unchanged into the next attribute's iteration alongside the newly
        # valued "with" groups.
        groups = without_groups + valued

    # Final merge: several partial groups may have converged on the exact
    # same assignments via different guard paths (e.g. two DT branches that
    # both happen to assign the same values) — collapse those into one
    # PreparedEffectGroup whose guard is the OR (SOP) of their guard clauses.
    merged: Dict[FrozenSet[Tuple[str, str]], List[_PartialGroup]] = defaultdict(list)
    for g in groups:
        merged[frozenset(g.assignments)].append(g)

    result: List[PreparedEffectGroup] = []
    for assignment_set, partials in merged.items():
        sop = _dedupe_sop([pg.guard_clause for pg in partials])
        # A SOP made of a single empty clause means "always true" — normalize
        # it to [] to match the "no guard" convention used everywhere else.
        if sop == [[]]:
            sop = []
        # Keep the highest probability among merged branches (mirrors the old
        # EffectDuplicator's "keep highest effect_probability" dedup policy).
        probability = round(max(pg.probability for pg in partials), 4)
        assignments = sorted(assignment_set)
        result.append(PreparedEffectGroup(
            assignments=assignments, guard=sop, probability=probability,
        ))
    return result
