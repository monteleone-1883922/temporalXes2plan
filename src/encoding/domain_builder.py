"""Builds a PDDLDomain from either a ParseResult (XES pipeline) or an
already-resolved PreparedDomainInput (web GUI's current.json).

Typed port of the 3-axis Cartesian-product algorithm: PreparedDomainInput/
PreparedTransition/PreparedXorBranch/Guard/AttributeCatalogEntry objects let
this reuse guard_encoder.and_clause_to_conditions and
effect_encoder.value_to_pddl_effects directly (no dict-to-object adapter
needed), plus variant deduplication (encoding/transition_action_builder.py).

No name/value sanitization happens here: names pass through PreparedDomainInput
exactly as they arrive; sanitization stays confined to encoding/pddl_model.py's
rendering methods.

The heavy lifting is delegated to sibling modules, one per build phase:
    encoding/prepared_graph_utils.py     — graph indexing, XOR-branch lookup
    encoding/prepared_schema_builder.py  — PDDL types/constants/predicates
    encoding/effect_group_builder.py     — PreparedEffectGroup derivation
    encoding/transition_action_builder.py — the 3-axis action-variant engine
This file only orchestrates them and assembles the final PDDLDomain.
"""
from typing import Dict, List, Optional, Tuple

import core_utils as utils
from encoding.effect_group_builder import _prepared_effect_groups
from encoding.prepared_graph_utils import (
    _prepared_graph_index, _prepared_negated_attributes, _prepared_xor_branch_of,
)
from encoding.prepared_input import (
    GraphNode, PreparedDomainInput, PreparedTransition, PreparedXorBranch, VariantInfo,
)
from encoding.prepared_schema_builder import _prepared_constants, _prepared_predicates, _prepared_types
from encoding.pddl_model import PDDLAction, PDDLBaseAction, PDDLCondition, PDDLDomain, PDDLDurativeAction, PDDLEffect
from encoding.transition_action_builder import _build_prepared_transition_actions
from models import AnalysisConfig, ParseResult

logger = utils.get_logger(__name__)

# Fast Downward represents durative-action numeric values (durations, costs)
# internally as integers scaled by 1000 (confirmed by FD's own "rounding
# numeric constants ... to an accuracy of 0.001" warning) — so the largest
# representable value is INT32_MAX / 1000. Domains built from real event-log
# timing data (durations spanning months/years, in raw seconds) can exceed
# this by a wide margin, which silently overflows FD's internal state during
# search instead of raising a clean error (see _maybe_rescale_durations).
_FD_MAX_DURATION_VALUE = 2147483.647

# Candidate rescale units, finest first — _maybe_rescale_durations picks the
# smallest factor (finest unit) that brings the domain's max duration back
# under _FD_MAX_DURATION_VALUE, to lose as little precision as possible.
_DURATION_RESCALE_UNITS: List[Tuple[float, str]] = [
    (60.0, "minutes"),
    (3600.0, "hours"),
    (86400.0, "days"),
    (604800.0, "weeks"),
]


def _maybe_rescale_durations(domain: PDDLDomain) -> None:
    """Rescale every durative action's duration bounds when the domain's max
    duration exceeds Fast Downward's internal numeric limit, so the planner
    doesn't silently corrupt its own search state (see _FD_MAX_DURATION_VALUE).

    Sets domain.duration_scale_factor (1.0 if no rescale was needed) and logs
    a WARNING describing the unit shift whenever one is applied — callers
    that build a deadline (TIL) for this domain must divide it by the same
    factor to stay on the same PDDL time axis as the (now rescaled) actions.
    """
    durative_actions = [a for a in domain.actions if isinstance(a, PDDLDurativeAction)]
    max_duration = max((a.duration_max for a in durative_actions), default=0.0)

    if max_duration <= _FD_MAX_DURATION_VALUE:
        domain.duration_scale_factor = 1.0
        logger.debug(
            "Duration rescale check: max action duration %.3fs is within "
            "Fast Downward's internal limit (%.3fs) — no shift needed.",
            max_duration, _FD_MAX_DURATION_VALUE,
        )
        return

    factor: Optional[float] = None
    unit_name = ""
    for candidate_factor, candidate_unit in _DURATION_RESCALE_UNITS:
        if max_duration / candidate_factor <= _FD_MAX_DURATION_VALUE:
            factor, unit_name = candidate_factor, candidate_unit
            break
    if factor is None:
        # Extreme fallback: not even weeks are enough — compute a custom
        # factor with a safety margin instead of leaving the domain broken.
        factor = max_duration / (_FD_MAX_DURATION_VALUE * 0.9)
        unit_name = f"custom(/{factor:.6g})"

    for action in durative_actions:
        action.duration_min /= factor
        action.duration_max /= factor
    domain.duration_scale_factor = factor

    logger.warning(
        "Duration unit shift applied: max action duration %.1fs exceeds "
        "Fast Downward's internal limit (~%.1fs) — rescaled all durative "
        "action bounds by /%.4g (seconds -> %s) to avoid numeric overflow "
        "during search. duration_scale_factor=%.6g stored on the domain; "
        "any deadline (TIL) built for this domain must be divided by this "
        "same factor to stay on the same time axis.",
        max_duration, _FD_MAX_DURATION_VALUE, factor, unit_name, factor,
    )


class DomainBuilder:
    """Builds a PDDLDomain from a ParseResult."""

    def build_prepared_input(
        self, parse_result: ParseResult, config: Optional[AnalysisConfig] = None,
    ) -> PreparedDomainInput:
        """Convert a XES-pipeline ParseResult into a PreparedDomainInput.

        This is the ParseResult-side "meeting point" adapter: it derives
        xor_branches (grouped by place) and, per transition, a
        PreparedTransition whose effect_groups are computed directly from
        TransitionInfo.effects (see encoding/effect_group_builder.py) — the
        same shape the web GUI produces from current.json, so both feed into
        the same build_domain_from_prepared_info below.
        """
        config = config or AnalysisConfig()
        track_appearance_cost = config.effect_appearance_mode != "duplicate_no_cost"
        track_value_cost = config.effect_value_mode != "duplicate_no_cost"

        xor_branches: Dict[str, List[PreparedXorBranch]] = {}
        transitions: Dict[str, PreparedTransition] = {}
        for trans_name, info in sorted(parse_result.transitions.items()):
            if info.xor_branches:
                for xor_branch in info.xor_branches:
                    xor_branches.setdefault(xor_branch.place_name, []).append(
                        PreparedXorBranch(trans_name, xor_branch.guards, xor_branch.probability)
                    )

            transitions[trans_name] = PreparedTransition(
                activity_name=trans_name,
                input_places=info.input_places,
                preconditions=[],
                effect_groups=_prepared_effect_groups(
                    info, track_appearance_cost, track_value_cost,
                ),
                cost=0.0,
                duration=info.duration,
            )

        nodes, edges = PreparedDomainInput.build_graph_from_petri_net(parse_result.petri_net_model)
        return PreparedDomainInput(
            nodes=nodes,
            edges=edges,
            transitions=transitions,
            xor_branches=xor_branches,
            attribute_catalog=parse_result.attribute_catalog,
        )


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
    domain, _ = build_domain_with_variant_map(
        prepared, config=config, domain_name=domain_name,
        use_durative=use_durative, use_costs=use_costs, has_deadline=has_deadline,
    )
    return domain


def build_domain_with_variant_map(
    prepared: PreparedDomainInput,
    config: Optional[AnalysisConfig] = None,
    domain_name: str = "test_process",
    use_durative: bool = False,
    use_costs: bool = False,
    has_deadline: bool = False,
) -> Tuple[PDDLDomain, Dict[str, VariantInfo]]:
    """Same build as build_domain_from_prepared_info, plus the variant lookup.

    variant_map maps every generated action's exact name (e.g.
    "execute_register_v2") to the VariantInfo (combined precondition guard +
    effect group) that produced it — the lookup a plan replayer needs to
    identify exactly which effect group the planner's chosen variant applies,
    without regenerating the axis A x B x C Cartesian product at runtime (see
    docs/trace_replayer_analysis.md §6.3,
    claude_plans/trace_replayer_implementation_plan.md §2.2-2.3).

    build_domain_from_prepared_info is a thin wrapper around this function
    that discards variant_map, kept as the stable public entry point for
    callers that only need the PDDLDomain.
    """
    config = config or AnalysisConfig()
    out_places_by_label, in_places_by_label = _prepared_graph_index(prepared)
    xor_branch_of = _prepared_xor_branch_of(prepared)
    negated_attributes = _prepared_negated_attributes(prepared)

    types = _prepared_types(prepared.attribute_catalog)
    constants = _prepared_constants(prepared)
    predicates = _prepared_predicates(prepared.attribute_catalog, negated_attributes)

    actions: List[PDDLBaseAction] = []
    variant_map: Dict[str, VariantInfo] = {}
    for act_name, transition in prepared.transitions.items():
        trans_actions, trans_variant_map = _build_prepared_transition_actions(
            act_name, transition, prepared.attribute_catalog,
            negated_attributes, use_durative, xor_branch_of.get(act_name), config.lower_bound_prob_actions,
            out_places_by_label.get(act_name, []),
        )
        actions.extend(trans_actions)
        variant_map.update(trans_variant_map)

    # Silent (tau) transitions that never appear as a keyed PreparedTransition
    # (e.g. pure routing steps with no effects) still need a direct
    # place-transfer action to keep the net's token flow connected.
    covered = set(prepared.transitions.keys())
    for node in prepared.nodes:
        if node.type == "silent" and node.label and node.label not in covered:
            tau = _build_prepared_tau_action(node, in_places_by_label, out_places_by_label)
            if tau:
                actions.append(tau)

    requirements = [":strips", ":typing"]
    if any(isinstance(a, PDDLDurativeAction) for a in actions):
        requirements += [":durative-actions", ":duration-inequalities"]

    has_costs = use_costs
    if has_costs:
        requirements += [":action-costs", ":numeric-fluents"]

    effective_deadline = has_deadline and any(
        isinstance(a, PDDLDurativeAction) for a in actions
    )
    if effective_deadline:
        requirements += [":timed-initial-literals"]

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
    _maybe_rescale_durations(domain)
    return domain, variant_map


# ------------------------------------------------------------------
# Phase 5 — Silent (tau) transition actions
# ------------------------------------------------------------------
#
# There used to be a separate "Phase 5" here building one
# mark_places_from_{transition} action per transition (consuming a
# transition-as-place marker, produced by execute_*, to propagate tokens
# onward) -- removed: standard Petri net semantics never marks a
# transition, only places, so execute_*/the tau action below now unmark
# their input places and mark their real output places directly, in one
# single action (see claude_plans/standard_petri_net_marking_plan.md).

def _build_prepared_tau_action(
    node: GraphNode,
    in_places_by_label: Dict[str, List[str]],
    out_places_by_label: Dict[str, List[str]],
) -> Optional[PDDLAction]:
    """Direct place-transfer action for a silent transition: unmark its
    input places, mark its real output places -- a single action modeling
    the whole firing, same as _build_prepared_transition_actions."""
    label = node.label
    if not label:
        return None

    in_places = in_places_by_label.get(label, [])
    out_places = out_places_by_label.get(label, [])

    preconds: set = {PDDLCondition.marked(p) for p in in_places}
    effects: List[PDDLEffect] = [PDDLEffect.unmarking(p) for p in in_places]
    effects.extend(PDDLEffect.marking(p) for p in sorted(out_places))

    return PDDLAction(
        name=f"execute_{label}",
        preconditions=preconds,
        effects=effects,
    )
