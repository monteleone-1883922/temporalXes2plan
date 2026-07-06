import dataclasses
import math
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import core_utils
import core_utils as utils
from encoding.prepared_input import (
    GraphNode, PreparedDomainInput, PreparedTransition, PreparedXorBranch, PreparedEffectGroup,
)
from models import AnalysisConfig, AttributeCatalogEntry, Guard, ParseResult, PetriNetModel
from encoding.pddl_model import (
    PDDLAction, PDDLBaseAction, PDDLCondition, PDDLConstants, PDDLDomain,
    PDDLDurativeAction, PDDLEffect, PDDLObject, PDDLPredicate, PDDLPredicates,
    PDDLType, PDDLTypes,
)
from encoding.action_builder import ActionBuilder
from encoding.action_registry import ActionRegistry
from encoding.effect_duplicator import EffectDuplicator, _deduplicate_variants
from encoding.effect_encoder import value_to_pddl_effects
from encoding.guard_encoder import and_clause_to_conditions
from encoding.xor_branch_processor import XorBranchProcessor

logger = utils.get_logger(__name__)

_PLACE_NODE_TYPES = frozenset({"place", "xor_split"})
_TRANS_NODE_TYPES = frozenset({"transition", "and_split", "silent"})


class DomainBuilder:
    """Builds a PDDLDomain from a ParseResult."""



    def _build_prepared_domain_input(self, parse_result: ParseResult, config: Optional[AnalysisConfig] = None) -> PreparedDomainInput:
        xor_branches: Dict[str, List[PreparedXorBranch]] = defaultdict(list)
        for trans_name, info in sorted(parse_result.transitions.items()):
            if info.xor_branches:
                for xor_branch in info.xor_branches:
                    xor_branches[xor_branch.place_name].append(
                        PreparedXorBranch(trans_name, xor_branch.guards, xor_branch.probability)
                    )


        prepare_domain_input = PreparedDomainInput(
            xor_branches=xor_branches,
            attribute_catalog=parse_result.attribute_catalog,

        )
        prepare_domain_input.build_graph_from_petri_net(parse_result.petri_net_model)
        return prepare_domain_input




    def _build_internal(
        self,
        parse_result: ParseResult,
        domain_name: str,
        use_durative: bool,
        use_costs: bool,
        config: Optional[AnalysisConfig],
    ) -> tuple[PDDLDomain, "ActionRegistry"]:
        if config is None:
            config = AnalysisConfig()

        types = self._build_types(parse_result.attribute_catalog)
        constants = self._build_constants(parse_result)
        predicates = self._build_predicates(
            parse_result.attribute_catalog, parse_result.negated_attributes
        )

        action_builder = ActionBuilder(parse_result, use_durative=use_durative)
        base_actions = action_builder.build_all()

        registry = ActionRegistry.from_base_actions(base_actions)
        XorBranchProcessor(parse_result).apply(registry)
        EffectDuplicator(parse_result, config).apply(registry)
        actions = registry.all_actions()

        requirements = [":strips", ":typing"]
        if any(isinstance(a, PDDLDurativeAction) for a in actions):
            requirements += [":durative-actions", ":duration-inequalities"]

        has_costs = use_costs
        if has_costs:
            requirements += [":action-costs", ":numeric-fluents"]

        effective_deadline = use_durative and any(
            isinstance(a, PDDLDurativeAction) for a in actions
        )
        domain = PDDLDomain(
            name=domain_name,
            requirements=requirements,
            types=types,
            constants=constants,
            predicates=predicates,
            actions=actions,
            has_costs=has_costs,
            has_deadline=effective_deadline,
        )
        return domain, registry


# ---------------------------------------------------------------------------
# Shared build engine: PreparedDomainInput -> PDDLDomain
# ---------------------------------------------------------------------------
#
# Typed port of the 3-axis Cartesian-product algorithm already used by
# encoding/domain_rebuilder.py::DomainRebuilder.rebuild (which operates on the
# raw current.json dict). Working on typed PreparedDomainInput/PreparedTransition/
# PreparedXorBranch/Guard/AttributeCatalogEntry objects lets this reuse
# guard_encoder.and_clause_to_conditions and effect_encoder.value_to_pddl_effects
# directly (no dict-to-object adapter needed), and adds variant deduplication
# (reused from effect_duplicator._deduplicate_variants) that DomainRebuilder
# does not do today.
#
# No name/value sanitization happens here: names pass through PreparedDomainInput
# exactly as they arrive; sanitization stays confined to encoding/pddl_model.py's
# rendering methods.


def build_domain_from_prepared_info(
    prepared: PreparedDomainInput,
    config: Optional[AnalysisConfig] = None,
    domain_name: str = "test_process",
    use_durative: bool = False,
    use_costs: bool = False,
    has_deadline: bool = False,
) -> PDDLDomain:
    """Build a PDDLDomain from an already-resolved PreparedDomainInput.

    Args:
        prepared: Fully resolved input (graph, transitions, XOR branches,
            attribute catalog) — the same shape regardless of whether it came
            from the XES pipeline or from the web GUI's current.json.
        domain_name: Name for the PDDL domain.
        use_durative: Encode transitions with duration data as durative
            actions when True.
        use_costs: Include action cost effects and functions section when True.
        has_deadline: Add deadline_exceeded predicate and over-all condition
            when True and at least one durative action is produced.

    Returns:
        A fully populated PDDLDomain.
    """
    config = config or AnalysisConfig()
    out_places_by_label, in_places_by_label = _prepared_graph_index(prepared)
    xor_branch_of = _prepared_xor_branch_of(prepared)
    negated_attributes = _prepared_negated_attributes(prepared)

    types = _prepared_types(prepared.attribute_catalog)
    constants = _prepared_constants(prepared)
    predicates = _prepared_predicates(prepared.attribute_catalog, negated_attributes)

    actions: List[PDDLBaseAction] = []
    for act_name, transition in prepared.transitions.items():
        actions.extend(_build_prepared_transition_actions(
            act_name, transition, prepared.attribute_catalog,
            negated_attributes, use_durative, xor_branch_of.get(act_name), config.lower_bound_prob_actions
        ))

    covered = set(prepared.transitions.keys())
    for node in prepared.nodes:
        if node.type == "silent" and node.label and node.label not in covered:
            tau = _build_prepared_tau_action(node, in_places_by_label)
            if tau:
                actions.append(tau)

    actions.extend(_build_prepared_place_marking_actions(out_places_by_label))

    requirements = [":strips", ":typing"]
    if any(isinstance(a, PDDLDurativeAction) for a in actions):
        requirements += [":durative-actions", ":duration-inequalities"]

    has_costs = use_costs
    if has_costs:
        requirements += [":action-costs", ":numeric-fluents"]

    effective_deadline = has_deadline and any(
        isinstance(a, PDDLDurativeAction) for a in actions
    )

    return PDDLDomain(
        name=domain_name,
        requirements=requirements,
        types=types,
        constants=constants,
        predicates=predicates,
        actions=actions,
        has_costs=has_costs,
        has_deadline=effective_deadline,
    )


# ------------------------------------------------------------------
# Phase 1 — Graph indexes
# ------------------------------------------------------------------

def _prepared_graph_index(
    prepared: PreparedDomainInput,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build transition-label -> output/input place-label lookups.

    Returns:
        out_places_by_label: transition label -> output place labels.
        in_places_by_label: transition label -> input place labels.
    """
    node_by_id: Dict[str, GraphNode] = {n.id: n for n in prepared.nodes}

    out_places_by_label: Dict[str, List[str]] = {}
    in_places_by_label: Dict[str, List[str]] = {}

    for n in prepared.nodes:
        if n.type in _TRANS_NODE_TYPES and n.label:
            out_places_by_label.setdefault(n.label, [])
            in_places_by_label.setdefault(n.label, [])

    for edge in prepared.edges:
        src_node = node_by_id.get(edge.source)
        tgt_node = node_by_id.get(edge.target)
        if not src_node or not tgt_node:
            continue

        if src_node.type not in _PLACE_NODE_TYPES:
            # transition -> place
            if src_node.label in out_places_by_label:
                out_places_by_label[src_node.label].append(tgt_node.label)
        else:
            # place -> transition
            if tgt_node.label in in_places_by_label:
                in_places_by_label[tgt_node.label].append(src_node.label)

    return out_places_by_label, in_places_by_label


# ------------------------------------------------------------------
# Phase 2 — XOR-branch lookup and negated attributes
# ------------------------------------------------------------------

def _prepared_xor_branch_of(
    prepared: PreparedDomainInput,
) -> Dict[str, List[PreparedXorBranch]]:
    """Flatten xor_branches (keyed by place) into activity_name -> branches.

    A transition can be downstream of several independent XOR splits, so it
    may receive more than one PreparedXorBranch.
    """
    result: Dict[str, List[PreparedXorBranch]] = defaultdict(list)
    for branches in prepared.xor_branches.values():
        for branch in branches:
            result[branch.activity_name].append(branch)
    return dict(result)


def _prepared_negated_attributes(prepared: PreparedDomainInput) -> Set[str]:
    """Collect attribute names that appear with Guard.negated=True anywhere."""
    negated: Set[str] = set()

    def _scan_sop(sop: List[List[Guard]]) -> None:
        for clause in sop:
            for guard in clause:
                if guard.negated:
                    negated.add(guard.attribute)

    for transition in prepared.transitions.values():
        _scan_sop(transition.preconditions)
        for group in transition.effect_groups:
            _scan_sop(group.guard)

    for branches in prepared.xor_branches.values():
        for branch in branches:
            _scan_sop(branch.conditions)

    return negated


# ------------------------------------------------------------------
# Phase 3 — Types, constants, predicates
# ------------------------------------------------------------------

def _prepared_types(attribute_catalog: Dict[str, AttributeCatalogEntry]) -> PDDLTypes:
    types = PDDLTypes(items=[
        PDDLType("petri_element"),
        PDDLType("place", parent="petri_element"),
        PDDLType("transition", parent="petri_element"),
    ])
    for attr_name, entry in sorted(attribute_catalog.items()):
        if entry.attribute_type in ("categorical", "numerical"):
            types.add(PDDLType(f"{attr_name}_val"))
    return types


def _prepared_constants(prepared: PreparedDomainInput) -> PDDLConstants:
    constants = PDDLConstants()

    for label in sorted(
        n.label for n in prepared.nodes if n.type in _PLACE_NODE_TYPES
    ):
        constants.add(PDDLObject(label, "place"))

    for label in sorted(
        n.label for n in prepared.nodes if n.type in _TRANS_NODE_TYPES and n.label
    ):
        constants.add(PDDLObject(label, "transition"))

    for attr_name, entry in sorted(prepared.attribute_catalog.items()):
        if entry.attribute_type in ("categorical", "numerical"):
            type_name = f"{attr_name}_val"
            for val in sorted(str(v) for v in entry.possible_values):
                constants.add(PDDLObject(val, type_name))

    return constants


def _prepared_predicates(
    attribute_catalog: Dict[str, AttributeCatalogEntry],
    negated_attributes: Set[str],
) -> PDDLPredicates:
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


# ------------------------------------------------------------------
# Phase 4 — Labeled transition actions (3-axis Cartesian product)
# ------------------------------------------------------------------

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
    3-axis loop (_build_prepared_transition_actions), together with asse A e C.
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


def _build_prepared_transition_actions(
    act_name: str,
    transition: PreparedTransition,
    attribute_catalog: Dict[str, AttributeCatalogEntry],
    negated_attributes: Set[str],
    use_durative: bool,
    xor_branches: Optional[List[PreparedXorBranch]] = None,
    lower_bound_prob_actions: float = 1e-3
) -> List[PDDLBaseAction]:
    in_places = transition.input_places

    # Asse A — OR in preconditions, deduplicated
    prec_clauses: List[List[Guard]] = _dedupe_sop(transition.preconditions) or [[]]

    # Asse B — OR in XOR-branch conditions, combined across every branch this
    # transition is downstream of, deduplicated
    combined_conditions, unguarded_xor_prob = _combine_xor_branches(xor_branches)
    xor_clauses: List[List[Guard]] = _dedupe_sop(combined_conditions) or [[]]
    xor_cost = 0.0
    if 0.0 < unguarded_xor_prob != 1.0:
        xor_cost = -math.log(max(unguarded_xor_prob, lower_bound_prob_actions))

    # Asse C — one slot per OR clause in each effect group's guard, clauses
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
    variants: List[PDDLBaseAction] = []

    _ASSE_A = "asse A (precondizioni strutturali)"
    _ASSE_B = "asse B (branch XOR)"
    _ASSE_C = "asse C (guardia effect_group)"

    def _log_and_skip(labeled_guards: List[Tuple[Guard, str]]) -> bool:
        """Return True (and log) if labeled_guards contains a conflict."""
        conflict = Guard.first_conflict(labeled_guards)
        if conflict is None:
            return False
        (g1, l1), (g2, l2) = conflict
        logger.debug(
            "Skipping action combination for '%s': %s (%s) conflicts with %s (%s)",
            act_name, g1, l1, g2, l2,
        )
        return True

    for p_clause in prec_clauses:
        p_labeled = [(g, _ASSE_A) for g in p_clause]
        if _log_and_skip(p_labeled):
            continue  # asse A da solo già incoerente

        for x_clause in xor_clauses:
            x_labeled = [(g, _ASSE_B) for g in x_clause]
            px_labeled = p_labeled + x_labeled
            if _log_and_skip(px_labeled):
                continue  # scartata qui: evita di iterare tutto l'asse C invano

            for (eff_group, guard_clause) in effect_axis:
                c_labeled = [(g, _ASSE_C) for g in (guard_clause or [])]
                combined_labeled = px_labeled + c_labeled
                if _log_and_skip(combined_labeled):
                    continue

                combined_guards = frozenset(g for g, _ in combined_labeled)
                preconds: Set[PDDLCondition] = {PDDLCondition.marked(p) for p in in_places}
                preconds.update(and_clause_to_conditions(list(combined_guards)))

                cost = base_cost + xor_cost
                if eff_group is not None and guard_clause is None:
                    eff_prob = eff_group.probability
                    if 0.0 < eff_prob != 1.0:
                        cost += -math.log(max(eff_prob, lower_bound_prob_actions))

                effects: List[PDDLEffect] = [PDDLEffect.unmarking(p) for p in in_places]
                effects.append(PDDLEffect.marking(act_name))
                if eff_group is not None:
                    for attr, value in eff_group.assignments:
                        catalog_entry = attribute_catalog[attr]
                        effects.extend(value_to_pddl_effects(
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
                        effects_at_end=effects,
                        base_cost=cost if cost != 0.0 else None,
                    )
                else:
                    action = PDDLAction(
                        name=f"execute_{act_name}",
                        preconditions=preconds,
                        effects=effects,
                        base_cost=cost if cost != 0.0 else None,
                    )
                variants.append(action)

    variants = _deduplicate_variants(variants)
    if len(variants) == 1:
        variants = [dataclasses.replace(variants[0], name=f"execute_{act_name}")]
    else:
        variants = [
            dataclasses.replace(v, name=f"execute_{act_name}_v{i}")
            for i, v in enumerate(variants)
        ]

    return variants


# ------------------------------------------------------------------
# Phase 5 — Place-marking actions (one per transition)
# ------------------------------------------------------------------

def _build_prepared_place_marking_actions(
    out_places_by_label: Dict[str, List[str]]
) -> List[PDDLAction]:
    """One action per transition: consumes transition marker, marks ALL output places."""
    actions: List[PDDLAction] = []
    for trans_label, out_places in sorted(out_places_by_label.items()):
        if not out_places:
            continue
        effects = [PDDLEffect.unmarking(trans_label)]
        for p in sorted(out_places):
            effects.append(PDDLEffect.marking(p))
        actions.append(PDDLAction(
            name=f"mark_places_from_{trans_label}",
            preconditions={PDDLCondition.marked(trans_label)},
            effects=effects,
        ))
    return actions


# ------------------------------------------------------------------
# Phase 6 — Silent (tau) transition actions
# ------------------------------------------------------------------

def _build_prepared_tau_action(
    node: GraphNode,
    in_places_by_label: Dict[str, List[str]],
) -> Optional[PDDLAction]:
    label = node.label
    if not label:
        return None

    in_places = in_places_by_label.get(label, [])

    preconds: Set[PDDLCondition] = {PDDLCondition.marked(p) for p in in_places}
    effects: List[PDDLEffect] = [PDDLEffect.unmarking(p) for p in in_places]
    effects.append(PDDLEffect.marking(label))

    return PDDLAction(
        name=f"execute_{label}",
        preconditions=preconds,
        effects=effects,
    )
