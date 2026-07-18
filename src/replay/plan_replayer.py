"""Caso 4 of docs/trace_replayer_analysis.md §2: validate a planner-produced
plan (ordered action names, already including the exact _vN variant suffix)
against the PDDL domain that generated it.

Supersedes evaluation/plan_validator.py, which is unusable today (its
TYPE_CHECKING import points at encoding.action_registry.ActionRegistry,
deleted in an earlier session — see docs/trace_replayer_analysis.md §1/§6.4).
Unlike the log-replay modes in trace_replayer.py, there is no ambiguity to
resolve here: the planner already picked the exact variant name, so a direct
lookup in variant_map (encoding.domain_builder.build_domain_with_variant_map)
tells us precisely which effect group it committed to — no backtracking, no
tau resolution. If the chosen variant's preconditions don't hold, that is a
bug in the plan (or a stale domain), not an ambiguity: replay stops at that
step.

Every PDDLBaseAction the domain builder emits is one of:
  - "execute_{activity}"[_v{i}]: unmarks the activity's input places, marks
    its real output places, and (when the activity carries effect_groups)
    applies the chosen group's attribute assignments — all in a SINGLE
    action, modeling one whole real Petri net transition firing (standard
    semantics: only places are ever marked, never the transition itself —
    see claude_plans/standard_petri_net_marking_plan.md). This is also
    where variant_map has an entry when the activity carries effect_groups.
  - "execute_{tau_label}" for a tau with no PreparedTransition of its own
    (encoding/domain_builder.py::_build_prepared_tau_action) — same
    single-action shape, pure token-flow, never in variant_map.
A full transition firing in a plan is therefore exactly one step; this
module simulates token flow and (when present) attribute effects on that
single step.
"""
import re
from typing import Dict, List, Set

from encoding.prepared_graph_utils import _prepared_graph_index
from encoding.prepared_input import PreparedDomainInput, VariantInfo
from models import sop_holds

from replay.trace_replayer import ReplayOutcome, ReplayStep

_EXECUTE_PREFIX = "execute_"
_VARIANT_SUFFIX_RE = re.compile(r"_v\d+$")


def _base_activity_name(raw_action: str) -> str:
    """Strip the execute_ prefix and _vN variant suffix, if present.

    Used only to resolve token-flow (in/out places); the exact raw_action
    name (suffix included) is what variant_map is keyed by for the
    attribute-effect lookup — see replay_plan.
    """
    name = raw_action
    if name.startswith(_EXECUTE_PREFIX):
        name = name[len(_EXECUTE_PREFIX):]
    return _VARIANT_SUFFIX_RE.sub("", name)


def replay_plan(
    plan_steps: List[str],
    init_marking: Set[str],
    init_attributes: Dict[str, str],
    prepared: PreparedDomainInput,
    variant_map: Dict[str, VariantInfo],
) -> ReplayOutcome:
    """Forward-simulate plan_steps, stopping at the first infeasible step.

    Args:
        plan_steps: Ordered action names exactly as returned by the planner
            (e.g. "execute_register_v1").
        init_marking: Place names holding a token in the initial state.
        init_attributes: Initial attribute -> value state.
        prepared: The PreparedDomainInput the plan's domain was built from.
        variant_map: From encoding.domain_builder.build_domain_with_variant_map
            — must come from building the same PreparedDomainInput, or the
            lookup below is meaningless.

    Returns:
        ReplayOutcome — is_replayable is False at the first step whose
        token-flow or attribute preconditions fail; steps records every
        firing, mirroring trace_replayer.ReplayStep.
    """
    out_places_by_label, in_places_by_label = _prepared_graph_index(prepared)

    marking: Set[str] = set(init_marking)
    attributes: Dict[str, str] = dict(init_attributes)
    steps: List[ReplayStep] = []

    for step_idx, raw_action in enumerate(plan_steps):
        base_name = _base_activity_name(raw_action)
        input_places = in_places_by_label.get(base_name, [])
        missing = [p for p in input_places if p not in marking]
        if missing:
            return ReplayOutcome(
                is_replayable=False, error_step=step_idx,
                error_reason=(
                    f"Action '{raw_action}' requires tokens in {missing}, "
                    f"current marking is {sorted(marking)}."
                ),
                steps=steps, final_marking=marking,
                final_attributes=attributes, warnings=[],
            )

        # variant_info is None for tau actions (no PreparedTransition, no
        # effect data at all) — those just move tokens, nothing to verify or
        # apply at the attribute level, hence is_tau=(variant_info is None)
        # below.
        variant_info = variant_map.get(raw_action)
        applied: Dict[str, str] = {}
        if variant_info is not None:
            # Exact lookup, no ambiguity: the planner already named the
            # precise variant, so this either holds or the plan/domain is
            # stale — no alternative to retry (unlike trace_replayer.py's
            # backtracking over unresolved log ambiguity).
            if not sop_holds(variant_info.preconditions, attributes):
                return ReplayOutcome(
                    is_replayable=False, error_step=step_idx,
                    error_reason=(
                        f"Action '{raw_action}' preconditions not satisfied. "
                        f"Current attributes: {attributes}."
                    ),
                    steps=steps, final_marking=marking,
                    final_attributes=attributes, warnings=[],
                )
            if variant_info.effect_group is not None:
                applied = dict(variant_info.effect_group.assignments)
                attributes.update(applied)

        marking.difference_update(input_places)
        marking.update(out_places_by_label.get(base_name, []))
        steps.append(ReplayStep(
            activity_name=base_name, is_tau=(variant_info is None),
            attributes_applied=applied,
        ))

    return ReplayOutcome(
        is_replayable=True, error_step=None, error_reason=None,
        steps=steps, final_marking=marking, final_attributes=attributes,
        warnings=[],
    )
