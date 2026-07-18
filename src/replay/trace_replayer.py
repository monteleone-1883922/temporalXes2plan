"""Phase 2 of the trace replayer: walk an already-resolved FiringStep sequence
(Phase 1, see _log_alignment.py) verifying PDDL-level executability — XOR
routing guards (axis B) and effect-group guards (axis C) — and applying the
chosen effect group's assignments to an accumulated attribute state.

Covers the 3 log-based use cases of docs/trace_replayer_analysis.md §2 (a 4th,
plan validation, lives in plan_replayer.py and doesn't touch this module):

1. replay_full_trace: replay the entire trace, verify it is fully executable.
   Phase 1 + Phase 2 (this module's guard/effect-group walk) from the start.
2. replay_evaluation_split: for evaluation. Phase 1 always runs on the WHOLE
   trace (never truncated — see _log_alignment's docstring and the module
   history: truncating before Phase 1 is unsound, especially for
   config.replay_engine == "alignments", since the optimal alignment of a
   truncated prefix is not guaranteed to be a prefix of the optimal alignment
   of the full trace). A cheap, unambiguous first pass (_build_snapshots)
   extracts the raw log-observed state (marking + attributes, no
   guard/effect-group interpretation) at the cut point and at the true end.
   Phase 2 then walks only the tail (cut -> end) verifying: it completes
   without blocking, its own resulting attribute state matches the real
   final snapshot exactly (retrying alternate effect-group choices at any
   ambiguous split if it doesn't), and optionally the accumulated duration
   of that tail against a deadline.
3. replay_partial_trace: the trace itself is genuinely incomplete (no
   artificial cut, there is nothing after it). Only Phase 1 + the raw final
   snapshot are needed — Phase 2's guard/effect-group walk is not run at all
   for this case.

The only real choice point Phase 2 ever resolves is "which effect group did
this labeled transition actually apply" (the log does not say) — tau
ambiguity is already fully resolved by Phase 1. When more than one group's
guard is satisfiable, a snapshot is recorded (state + ordered candidate list)
before applying the best-guess group, mirroring PartialTraceReplayer's
TauSplitPoint/_backtrack mechanism, generalized to effect-group choice. For
replay_evaluation_split, a mismatch against the target final state at the end
of the walk is treated exactly like any other block: it triggers the same
backtrack to the next untried candidate.
"""
import dataclasses
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from encoding.prepared_graph_utils import _prepared_xor_branch_of
from encoding.prepared_input import PreparedDomainInput, PreparedEffectGroup, PreparedTransition
from encoding.transition_action_builder import _combine_xor_branches
from models import AnalysisConfig, AttributeCatalogEntry, FiringStep, Guard, PetriNetModel, sop_holds

import core_utils as utils
from replay._log_alignment import align_single_trace

logger = utils.get_logger(__name__)

# pm4py event attributes that are not case data — same set as
# parsing/partial_trace_replayer.py's _PM4PY_META_KEYS.
_PM4PY_META_KEYS = frozenset({
    "concept:name",
    "time:timestamp",
    "lifecycle:transition",
    "org:resource",
    "org:group",
    "org:role",
    "@@index",
    "@@classifier",
})


@dataclasses.dataclass
class ReplayStep:
    """One resolved step of the walk, in trace order (tau included)."""
    activity_name: str
    is_tau: bool
    attributes_applied: Dict[str, str]  # {} for tau or a no-effect-group firing


@dataclasses.dataclass
class ReplayOutcome:
    """Result of replay_full_trace (caso 1).

    final_marking/final_attributes reflect the state after the last
    successfully executed step (i.e. up to error_step - 1 when not
    is_replayable, or the full walk when is_replayable is True).
    """
    is_replayable: bool
    error_step: Optional[int]
    error_reason: Optional[str]
    steps: List[ReplayStep]
    final_marking: Set[str]
    final_attributes: Dict[str, str]
    warnings: List[str]


@dataclasses.dataclass
class PartialTraceSnapshot:
    """Result of replay_partial_trace (caso 3) — raw, unguarded state.

    marking/attributes are derived purely from Phase 1's resolved FiringStep
    sequence and the log events observed along it — no guard/effect-group
    interpretation is applied (see module docstring, case 3).
    """
    marking: Set[str]
    attributes: Dict[str, str]
    warnings: List[str]


@dataclasses.dataclass
class EvaluationReplayOutcome:
    """Result of replay_evaluation_split (caso 2) — 3 independently
    inspectable checks rather than one aggregate boolean:
      (a) reached_end: the tail walk (cut -> end) completed without blocking.
      (b) matches_expected_final_state: its resulting attributes equal the
          real final snapshot exactly.
      (c) within_deadline: accumulated_duration_seconds (counted from 0 at
          the cut, never including time before it) is within deadline_seconds
          — None when check_deadline was False or no deadline_seconds was
          given.
    error_step/error_reason are populated only for a genuine structural block
    (reached_end is False); a target mismatch that survives every backtrack
    alternative (reached_end True, matches_expected_final_state False) has no
    single failing step, so error_step stays None while error_reason still
    describes the mismatch.

    max_modeled_duration_seconds is a separate, independent quantity from
    accumulated_duration_seconds/within_deadline above: it sums each fired
    (non-tau) transition's *domain-declared* effective_max duration bound
    (PreparedTransition.duration.effective_max) over the tail
    (cut -> end), not the real observed inter-event gaps. It depends only
    on WHICH transitions fired — fixed by Phase 1, independent of which
    effect group Phase 2 ends up choosing for any of them — so it is
    always computed directly from firing_steps, with no dependency on
    _walk's backtracking outcome (see trace_sampler.PrefixSample's
    reached_within_time, which compares this sum against the trace's real
    remaining duration as a sufficient condition for "reachable within the
    time the log actually took").
    """
    reached_end: bool
    matches_expected_final_state: bool
    error_step: Optional[int]
    error_reason: Optional[str]
    steps: List[ReplayStep]
    split_marking: Set[str]
    split_attributes: Dict[str, str]
    final_marking: Set[str]
    final_attributes: Dict[str, str]
    expected_final_attributes: Dict[str, str]
    accumulated_duration_seconds: Optional[float]
    within_deadline: Optional[bool]
    max_modeled_duration_seconds: float
    warnings: List[str]


@dataclasses.dataclass
class _EffectChoicePoint:
    """A recorded fork where more than one effect group's guard was satisfied.

    candidates is fixed at creation time (sorted best-first, see
    _rank_candidates) — re-trying an alternative after backtracking always
    restores state_before first, so re-deriving candidates would yield the
    exact same list; storing it avoids relying on that determinism.
    """
    step_index: int
    state_before: Dict[str, str]
    candidates: List[PreparedEffectGroup]
    tried_count: int = 0


@dataclasses.dataclass
class _WalkResult:
    """Internal result of TraceReplayer._walk, before each public method
    reshapes it into its own outcome dataclass."""
    reached_end: bool
    matches_target: bool
    error_step: Optional[int]
    error_reason: Optional[str]
    steps: List[ReplayStep]
    final_marking: Set[str]
    final_attributes: Dict[str, str]
    accumulated_duration_seconds: Optional[float]
    warnings: List[str]


def _normalize_event_value(
    attr: str, raw_value: Any, entry: AttributeCatalogEntry, warnings: List[str],
) -> Optional[str]:
    """Convert a raw event attribute value into the catalog's value space.

    Same three-way dispatch as PartialTraceReplayer._normalize_value, but
    operating on the typed AttributeCatalogEntry (attribute_type/
    possible_values/bin_boundaries) instead of a current.json dict, and
    without the legacy sanitize_value(attr, value) "{attr}_val_{value}"
    wrapping — Guard.value/PreparedEffectGroup assignments carry plain
    catalog values (e.g. "flu"), not PDDL constant names (see
    encoding/guard_encoder.py::guard_to_condition), so the normalized value
    must live in that same plain space to compare equal.
    """
    if entry.attribute_type == "boolean":
        return "true" if str(raw_value).lower() in ("true", "1", "yes") else "false"

    if entry.attribute_type == "numerical":
        if not entry.bin_boundaries:
            warnings.append(f"Attribute '{attr}' ignored: no bin boundaries available.")
            return None
        try:
            return utils.discretize_value(attr, float(raw_value), {attr: entry.bin_boundaries})
        except (TypeError, ValueError):
            warnings.append(f"Value '{raw_value}' for '{attr}' ignored: cannot convert to float.")
            return None

    value = str(raw_value)
    if entry.possible_values and value not in entry.possible_values:
        warnings.append(f"Value '{value}' for '{attr}' ignored: not a known category.")
        return None
    return value


def _extract_event_attributes(
    attributes: Dict[str, Any],
    attribute_catalog: Dict[str, AttributeCatalogEntry],
    warnings: List[str],
) -> Dict[str, str]:
    """Normalize a FiringStep's raw event attributes into catalog values.

    Attributes not present in attribute_catalog are silently ignored (not
    every log column feeds the PDDL domain) — pm4py meta keys and None
    values are skipped before the catalog lookup.
    """
    observed: Dict[str, str] = {}
    for raw_key, raw_value in attributes.items():
        if raw_key in _PM4PY_META_KEYS or raw_value is None:
            continue
        entry = attribute_catalog.get(raw_key)
        if entry is None:
            continue
        value = _normalize_event_value(raw_key, raw_value, entry, warnings)
        if value is not None:
            observed[raw_key] = value
    return observed


def _score_group(group: PreparedEffectGroup, observed: Dict[str, str]) -> Tuple[int, int]:
    """Count how many of group's assignments match the observed event attrs."""
    equal_attr = sum(1 for attr, value in group.assignments if observed.get(attr) == value)
    return equal_attr, len(group.assignments) - equal_attr


def _score_effect_group(group: PreparedEffectGroup, observed: Dict[str, str]) -> Tuple[int, int, float]:
    eq_attr, add_attr = _score_group(group, observed)
    return -eq_attr, add_attr, -group.probability


def _rank_candidates(
    effect_groups: List[PreparedEffectGroup],
    pddl_state: Dict[str, str],
    observed: Dict[str, str],
) -> List[PreparedEffectGroup]:
    """Guard-satisfying groups, best-match first.

    Sort key: matching-attribute count desc, then differing-attribute count
    asc (tie-break added by design: among groups predicting the same number
    of observed attributes, prefer the leaner one — fewer unverified
    assignments), then probability desc. A stable sort preserves original
    order among exact ties, matching PartialTraceReplayer._select_best_variant
    's linear-scan "keep the first strict improvement" tie-break.
    """
    matching = [g for g in effect_groups if sop_holds(g.guard, pddl_state)]
    return sorted(matching, key=lambda g: _score_effect_group(g, observed))


def _collapse_no_impact(
    ranked_candidates: List[PreparedEffectGroup], pddl_state: Dict[str, str],
) -> List[PreparedEffectGroup]:
    """Collapse every "no-impact" candidate into a single representative.

    A candidate is no-impact when every one of its assignments already
    holds in pddl_state — applying it leaves pddl_state byte-for-byte
    unchanged. Any two no-impact candidates are therefore behaviorally
    interchangeable (same resulting state, so identical consequences for
    every later step and for the final target-match check); keeping more
    than one of them as separate choice-point alternatives can never
    change the walk's outcome, only multiply backtracking work — the
    dominant driver of the combinatorial blow-up on traces where an
    ambiguous transition fires repeatedly inside a process loop.

    Impactful candidates (assign at least one value the state doesn't
    already have) are never collapsed with each other — only redundant
    no-impact ones are pruned, keeping the highest-ranked among them.
    """
    result: List[PreparedEffectGroup] = []
    kept_no_impact = False
    for g in ranked_candidates:
        no_impact = all(pddl_state.get(attr) == value for attr, value in g.assignments)
        if no_impact:
            if kept_no_impact:
                continue
            kept_no_impact = True
        result.append(g)
    return result


def _sum_modeled_max_duration(
    firing_steps: List[FiringStep],
    transitions: Dict[str, PreparedTransition],
) -> float:
    """Sum of effective_max (domain-declared upper duration bound) over
    every non-tau firing in firing_steps.

    Tau firings contribute 0 -- no PDDL duration is ever modeled for them
    (domain_builder's tau actions are plain, non-durative PDDLAction).
    Transitions with no mined duration data (transition.duration is None)
    also contribute 0 -- absence of information is treated as
    non-constraining, same convention used elsewhere (e.g.
    query_builder.is_q3_reachable).

    Depends only on WHICH transitions fired (fixed by Phase 1) — not on
    which effect group Phase 2 picks for any of them, since duration is a
    property of the transition itself. Safe to call directly on a resolved
    firing_steps list, with no dependency on _walk/backtracking.
    """
    total = 0.0
    for step in firing_steps:
        if step.is_tau:
            continue
        transition = transitions.get(step.activity_name)
        if transition is not None and transition.duration is not None:
            total += transition.duration.effective_max
    return total


class TraceReplayer:
    """Replays a trace (full, evaluation-split, or partial) against a
    PreparedDomainInput.

    Fase 1 (tau/marking resolution) is delegated to _log_alignment
    .align_single_trace per call — it is cheap enough (single trace) to redo
    for every call rather than caching, and keeps this class stateless across
    calls beyond the domain/model it was built for. Fase 1 always receives
    the complete trace it is given, never a truncated copy — see module
    docstring on why truncating before Fase 1 is unsound.
    """

    def __init__(
        self,
        prepared: PreparedDomainInput,
        petrinet_model: PetriNetModel,
        config: Optional[AnalysisConfig] = None,
    ) -> None:
        self.prepared = prepared
        self.petrinet_model = petrinet_model
        self.config = config or AnalysisConfig()
        self._xor_branch_of = _prepared_xor_branch_of(prepared)

    # ------------------------------------------------------------------
    # Public API — one method per use case
    # ------------------------------------------------------------------

    def replay_full_trace(self, trace: Any) -> ReplayOutcome:
        """Case 1: replay the entire trace, verify it is fully executable."""
        model = self.petrinet_model
        execution, phase1_warnings = align_single_trace(
            trace, model.petrinet, model.initial_marking, model.final_marking,
            model.trans_inputs, model.trans_outputs, model.silent_transitions,
            self.config, is_partial=False,
        )
        initial_marking = {p.name for p in model.initial_marking}
        result = self._walk(
            execution.steps, initial_marking=initial_marking, initial_attributes={},
            target_final_attributes=None, accumulate_duration=False,
        )
        return ReplayOutcome(
            is_replayable=result.reached_end,
            error_step=result.error_step,
            error_reason=result.error_reason,
            steps=result.steps,
            final_marking=result.final_marking,
            final_attributes=result.final_attributes,
            warnings=phase1_warnings + result.warnings,
        )

    def replay_partial_trace(self, trace: Any) -> PartialTraceSnapshot:
        """Caso 3: the trace itself is genuinely incomplete — there is
        nothing to cut, and Phase 2's guard/effect-group walk is not
        required (only Phase 1 + the raw final snapshot)."""
        model = self.petrinet_model
        execution, phase1_warnings = align_single_trace(
            trace, model.petrinet, model.initial_marking, model.final_marking,
            model.trans_inputs, model.trans_outputs, model.silent_transitions,
            self.config, is_partial=True,
        )
        marking_at, raw_attributes_at, _event_counts, snapshot_warnings = (
            self._build_snapshots(execution.steps)
        )
        return PartialTraceSnapshot(
            marking=marking_at[-1],
            attributes=raw_attributes_at[-1],
            warnings=phase1_warnings + snapshot_warnings,
        )

    def replay_evaluation_split(
        self,
        trace: Any,
        n_prefix: int,
        check_deadline: bool = False,
        deadline_seconds: Optional[float] = None,
    ) -> EvaluationReplayOutcome:
        """Caso 2: evaluation split of a COMPLETE trace.

        Phase 1 runs on the whole trace. A raw (unguarded) pass extracts the
        log-observed state at the cut point (n_prefix events consumed) and at
        the true end. Phase 2 then walks only the tail, starting from the cut
        snapshot, verifying it reaches the end matching that real final
        snapshot exactly — retrying alternate effect-group choices at any
        ambiguous split if the first attempt doesn't match — and, when
        check_deadline is True, accumulating duration_seconds along that tail
        only (never time before the cut) against deadline_seconds.
        """
        model = self.petrinet_model
        execution, phase1_warnings = align_single_trace(
            trace, model.petrinet, model.initial_marking, model.final_marking,
            model.trans_inputs, model.trans_outputs, model.silent_transitions,
            self.config, is_partial=False,
        )
        firing_steps = execution.steps
        marking_at, raw_attributes_at, event_counts, snapshot_warnings = (
            self._build_snapshots(firing_steps)
        )
        warnings = phase1_warnings + snapshot_warnings

        total_events = event_counts[-1]
        if n_prefix > total_events:
            warnings.append(
                f"n_prefix ({n_prefix}) exceeds the {total_events} visible "
                f"events available in the trace; using the full trace as the "
                f"split point."
            )
        # event_counts is non-decreasing by construction (_build_snapshots),
        # so this finds the smallest firing_steps index at which at least
        # n_prefix events have been consumed -- it will be an EXACT match
        # (event_counts[k] == n_prefix) whenever n_prefix is achievable,
        # since consecutive entries differ by 0 or 1; using >= instead of ==
        # only matters for the n_prefix-too-large case just warned about
        # above, where it naturally falls back to len(firing_steps).
        cut_idx = next(
            (k for k in range(len(event_counts)) if event_counts[k] >= n_prefix),
            len(firing_steps),
        )

        split_marking = marking_at[cut_idx]
        split_attributes = raw_attributes_at[cut_idx]
        expected_final_attributes = raw_attributes_at[-1]

        # Independent of _walk/backtracking (see _sum_modeled_max_duration's
        # docstring) -- computed directly from the resolved tail, regardless
        # of whether the walk below ends up reaching the end or blocking.
        max_modeled_duration_seconds = _sum_modeled_max_duration(
            firing_steps[cut_idx:], self.prepared.transitions,
        )

        # Only the TAIL is walked -- everything before cut_idx is taken as
        # given (the split snapshot), never re-verified here. This is what
        # makes replay_evaluation_split answer "does the domain correctly
        # explain the trace's real continuation from this point", not
        # "is the whole trace executable from the very start" (that's caso 1).
        result = self._walk(
            firing_steps[cut_idx:],
            initial_marking=split_marking,
            initial_attributes=split_attributes,
            target_final_attributes=expected_final_attributes,
            accumulate_duration=check_deadline,
        )
        warnings.extend(result.warnings)

        # Per user confirmation: duration is counted from 0 at the cut, never
        # including time spent before it -- accumulate_duration=check_deadline
        # above already guarantees _walk only sums durations of steps in the
        # tail, so no further adjustment is needed here, only the deadline
        # comparison itself.
        within_deadline = None
        if check_deadline and deadline_seconds is not None:
            within_deadline = (
                result.accumulated_duration_seconds is not None
                and result.accumulated_duration_seconds <= deadline_seconds
            )

        return EvaluationReplayOutcome(
            reached_end=result.reached_end,
            matches_expected_final_state=result.matches_target,
            error_step=result.error_step,
            error_reason=result.error_reason,
            steps=result.steps,
            split_marking=split_marking,
            split_attributes=split_attributes,
            final_marking=result.final_marking,
            final_attributes=result.final_attributes,
            expected_final_attributes=expected_final_attributes,
            accumulated_duration_seconds=result.accumulated_duration_seconds,
            within_deadline=within_deadline,
            max_modeled_duration_seconds=max_modeled_duration_seconds,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Shared building blocks
    # ------------------------------------------------------------------

    def _xor_guard_for(self, activity_name: str) -> List[List[Guard]]:
        """Combined axis-B SOP for activity_name (empty SOP = no constraint).

        Reuses transition_action_builder._combine_xor_branches so the same
        AND-of-all-independent-XOR-splits semantics used to build the PDDL
        domain applies here (the probability component it also returns is a
        cost concept, irrelevant to plain satisfiability)."""
        branches = self._xor_branch_of.get(activity_name)
        combined_sop, _unguarded_prob = _combine_xor_branches(branches)
        return combined_sop

    def _build_snapshots(
        self, firing_steps: List[FiringStep],
    ) -> Tuple[List[Set[str]], List[Dict[str, str]], List[int], List[str]]:
        """One linear pass, no ambiguity: marking is already deterministic
        from Phase 1, attributes are taken exactly as observed on each log
        event — no guard/effect-group interpretation (that is Phase 2's
        job, in _walk). This is the "log-observed state" PartialTraceReplayer
        used to call attrs_at, kept separate from the PDDL-simulated
        pddl_state that only _walk computes.

        Returns (marking_at, raw_attributes_at, event_counts, warnings), each
        of length len(firing_steps) + 1 — index k is the state after the
        first k steps (k=0 is the net's true initial state).
        """
        warnings: List[str] = []
        # Always the net's TRUE initial marking -- firing_steps here is
        # always Phase 1's resolved sequence for a trace replayed from the
        # start (both callers, replay_partial_trace and
        # replay_evaluation_split, pass the whole execution.steps).
        marking: Set[str] = {p.name for p in self.petrinet_model.initial_marking}
        # acc accumulates raw event attributes over time, later values
        # overwriting earlier ones for the same key -- same "running case
        # state" pattern as PartialTraceReplayer's acc/attrs_at. This is
        # deliberately NOT pddl_state: no guard or effect group is consulted
        # here, only what the log itself says happened.
        acc: Dict[str, str] = {}

        marking_at: List[Set[str]] = [set(marking)]
        raw_attributes_at: List[Dict[str, str]] = [dict(acc)]
        event_counts: List[int] = [0]

        for step in firing_steps:
            # Structural, unconditional -- same helper _walk uses, no
            # ambiguity possible here since it doesn't depend on any choice.
            self._advance_marking(marking, step)
            if not step.is_tau:
                # Tau never carries log attributes (FiringStep.attributes is
                # always {} for it), so skipping the extraction call is just
                # an optimization, not a correctness branch.
                observed = _extract_event_attributes(
                    step.attributes, self.prepared.attribute_catalog, warnings,
                )
                acc.update(observed)
            # Snapshot a COPY at every index, not a reference to the mutating
            # marking/acc -- otherwise every entry in marking_at/
            # raw_attributes_at would silently alias the same final object.
            marking_at.append(set(marking))
            raw_attributes_at.append(dict(acc))
            # +1 exactly on a labeled (non-tau) step: this is what lets
            # replay_evaluation_split map "n_prefix events consumed" back to
            # a firing_steps index, tau firings included in the sequence but
            # invisible to this count.
            event_counts.append(event_counts[-1] + (0 if step.is_tau else 1))

        return marking_at, raw_attributes_at, event_counts, warnings

    def _walk(
        self,
        firing_steps: List[FiringStep],
        initial_marking: Set[str],
        initial_attributes: Dict[str, str],
        target_final_attributes: Optional[Dict[str, str]],
        accumulate_duration: bool,
    ) -> _WalkResult:
        """Phase 2 core engine: walk firing_steps checking axis-B (XOR) and
        axis-C (effect-group) guards, backtracking over which effect group a
        labeled transition applied whenever more than one is
        guard-satisfiable (see module docstring).

        When target_final_attributes is given, reaching the end of
        firing_steps is not enough on its own: the accumulated attribute
        state must equal it exactly, or the walk backtracks to try the next
        effect-group alternative — even if every individual step's guard was
        satisfied along the way (replay_evaluation_split's case: verifying
        the domain actually explains the trace's real continuation, not just
        that *some* continuation exists).

        Axis A (structural attribute preconditions) is not checked here: it
        is empty today for every PreparedTransition
        (docs/trace_replayer_analysis.md §4) — the day
        PreparedTransition.preconditions stops being always [], a
        structural-precondition check belongs here too, alongside axis B.
        """
        marking: Set[str] = set(initial_marking)
        pddl_state: Dict[str, str] = dict(initial_attributes)
        steps: List[ReplayStep] = []
        choice_points: List[_EffectChoicePoint] = []
        warnings: List[str] = []
        duration_total: Optional[float] = 0.0 if accumulate_duration else None
        # Memoized (step_idx, frozen pddl_state) pairs already proven, on an
        # earlier backtracking attempt, to have no successful continuation
        # from here — see _backtrack. firing_steps is fixed and
        # deterministic, so the same (step_idx, state) pair always leads to
        # the same outcome; skipping recomputation here is what keeps a
        # repeatedly-firing ambiguous transition (a process loop) from
        # multiplying backtracking work across every repetition.
        dead_states: Set[Tuple[int, FrozenSet[Tuple[str, str]]]] = set()

        # Two nested loops, on purpose:
        #  - the INNER loop is a plain forward walk over firing_steps, from
        #    step_idx to either the end or the first block.
        #  - the OUTER loop is what makes backtracking possible: every time
        #    the inner loop stops (whether from a real block, or because it
        #    reached the end but the final state doesn't match the target),
        #    we ask _backtrack for the next untried alternative and, if one
        #    exists, jump step_idx back and re-enter the inner loop. If none
        #    is left, we return from inside the outer loop instead of ever
        #    reaching its bottom.
        step_idx = 0
        while True:
            # Reset every outer-loop attempt: blocked_step/blocked_reason are
            # only ever set by a genuine guard failure inside the inner loop
            # below (not by a target mismatch, which is handled separately
            # right after the inner loop exits normally). memo_hit_reason is
            # its own third category (see the dead_states branch below) --
            # deliberately kept OUT of blocked_step so a memoized skip can
            # never itself flip the final classification from "matches_target
            # False" to a fabricated "reached_end False": dead_states makes
            # no distinction between a subtree that was doomed by a genuine
            # guard failure deeper down versus one doomed only by an eventual
            # attribute mismatch, so treating every memo hit as at most a
            # mismatch keeps reached_end's meaning ("blocked_step really was
            # set on this exact attempt") intact and never claims a
            # structural block that this particular attempt didn't itself
            # observe.
            blocked_step: Optional[int] = None
            blocked_reason: Optional[str] = None
            memo_hit_reason: Optional[str] = None

            while step_idx < len(firing_steps):
                step = firing_steps[step_idx]
                # Token movement happens unconditionally, tau or not, guard
                # outcome or not — it is purely structural (Phase 1 already
                # decided which transition fires here; nothing in Phase 2
                # can undo that, only the attribute-level choice is ours).
                self._advance_marking(marking, step)

                if step.is_tau:
                    # No axis-A/B/C check for tau (see _walk's docstring):
                    # nothing to verify, nothing to apply.
                    steps.append(ReplayStep(step.activity_name, is_tau=True, attributes_applied={}))
                    if duration_total is not None:
                        duration_total += step.duration_seconds or 0.0
                    step_idx += 1
                    continue

                # transition may be absent from prepared.transitions (e.g. an
                # activity with no mined effect data) -- treated the same as
                # "no effect groups": fires with no attribute assignment.
                transition = self.prepared.transitions.get(step.activity_name)
                effect_groups = transition.effect_groups if transition else []
                xor_sop = self._xor_guard_for(step.activity_name)
                # observed = this step's own log event, normalized -- used
                # only to break ties between candidates (_rank_candidates),
                # never to decide guard satisfiability.
                observed = _extract_event_attributes(
                    step.attributes, self.prepared.attribute_catalog, warnings,
                )

                # Are we re-entering this exact step because a LATER step
                # (or the final target check) forced a backtrack that landed
                # right back here? If so, don't recompute candidates from
                # scratch -- pddl_state was just restored by _backtrack to
                # the identical state it had the first time we were at this
                # step, so recomputing would yield the same candidate list;
                # we just need to move to the NEXT untried one.
                top = choice_points[-1] if choice_points else None
                resuming = top is not None and top.step_index == step_idx

                if resuming:
                    xor_ok = True  # state_before is unchanged from first visit — already checked
                    candidates = top.candidates
                    chosen = candidates[top.tried_count] if candidates else None
                    is_blocked = bool(effect_groups) and not candidates
                elif (step_idx, frozenset(pddl_state.items())) in dead_states:
                    # A DIFFERENT earlier backtracking branch already reached
                    # this exact (position, attribute state) pair and proved
                    # -- exhaustively, over every candidate -- that no
                    # continuation from here matches the target. Since
                    # firing_steps is fixed, this branch would derive the
                    # identical candidate list and rediscover the identical
                    # failure; skip straight to backtracking instead of
                    # redoing that work (this is what stops an ambiguous
                    # transition firing N times in a process loop from
                    # costing work proportional to (branching factor)^N).
                    # Deliberately NOT routed through is_blocked/blocked_step
                    # -- see this loop's memo_hit_reason comment above.
                    memo_hit_reason = (
                        f"'{step.activity_name}' at this attribute state was already proven "
                        f"unreachable-to-target by an earlier backtrack attempt (memoized)."
                    )
                    break
                else:
                    # First time reaching this step (this attempt): evaluate
                    # the XOR guard (axis B) against the current attribute
                    # state, then rank every effect group (axis C) whose own
                    # guard also holds, best-match first. Candidates whose
                    # assignments are all already true in pddl_state (a
                    # no-op) are collapsed to a single representative --
                    # they are behaviorally interchangeable, so keeping
                    # several as separate alternatives only multiplies
                    # backtracking work without ever changing the outcome.
                    xor_ok = sop_holds(xor_sop, pddl_state)
                    candidates = (
                        _collapse_no_impact(_rank_candidates(effect_groups, pddl_state, observed), pddl_state)
                        if xor_ok else []
                    )
                    # Blocked when the XOR guard itself fails, OR when this
                    # transition does carry effect groups but none of their
                    # guards are satisfiable right now (an empty
                    # effect_groups list is NOT a block -- it just means "no
                    # attribute effect to apply", handled by chosen is None
                    # below).
                    is_blocked = (not xor_ok) or (bool(effect_groups) and not candidates)
                    chosen = candidates[0] if candidates else None
                    # Only a REAL ambiguity (2+ guard-satisfying candidates)
                    # is worth remembering as a choice point -- a single
                    # candidate has nothing to backtrack to later.
                    if xor_ok and len(candidates) > 1:
                        choice_points.append(_EffectChoicePoint(
                            step_index=step_idx,
                            state_before=dict(pddl_state),
                            candidates=candidates,
                        ))

                if is_blocked:
                    # Stop the inner loop right here (without advancing
                    # step_idx or appending to steps) and let the outer loop
                    # decide whether a backtrack can rescue this attempt.
                    blocked_step = step_idx
                    blocked_reason = self._block_reason(step.activity_name, xor_ok, effect_groups)
                    break

                # chosen is None exactly when effect_groups was empty (no
                # attribute effect for this transition) -- apply nothing in
                # that case rather than an empty assignments dict from a
                # real group, so attributes_applied stays {} either way.
                if chosen is not None:
                    pddl_state.update(dict(chosen.assignments))
                steps.append(ReplayStep(
                    step.activity_name, is_tau=False,
                    attributes_applied=dict(chosen.assignments) if chosen is not None else {},
                ))
                if duration_total is not None:
                    duration_total += step.duration_seconds or 0.0
                step_idx += 1

            # The inner loop above stopped for exactly one of three reasons:
            #  (1) blocked_step is not None -- a genuine guard failure.
            #  (2) memo_hit_reason is not None -- landed on a (step_idx,
            #      state) pair already proven dead; treated the same as (3)
            #      below (something to backtrack away from) but never as (1),
            #      since dead_states does not track whether the original
            #      exhaustion was itself a genuine block or "just" a
            #      mismatch further down -- see this loop's memo_hit_reason
            #      comment above.
            #  (3) blocked_step is None and memo_hit_reason is None --
            #      step_idx reached len(firing_steps), i.e. the walk is
            #      structurally complete. That is a real SUCCESS only if
            #      there is no target to match, or the final attributes
            #      happen to equal it exactly; otherwise it is treated below
            #      like (1)/(2) -- something to try backtracking away from
            #      (mismatch_reason is only ever used if we end up returning
            #      without finding a further alternative, see the
            #      "restored_idx is None" branch below).
            mismatch_reason: Optional[str] = None
            if memo_hit_reason is not None:
                mismatch_reason = memo_hit_reason
            elif blocked_step is None:
                if target_final_attributes is None or all(
                        pddl_state.get(attr) == val for attr, val in target_final_attributes.items()
                ):
                    return _WalkResult(
                        reached_end=True, matches_target=True,
                        error_step=None, error_reason=None,
                        steps=steps, final_marking=marking, final_attributes=pddl_state,
                        accumulated_duration_seconds=duration_total, warnings=warnings,
                    )
                mismatch_reason = (
                    "Final attribute state does not match the expected snapshot "
                    "after exhausting every effect-group alternative."
                )

            # Whether we got here via a structural block or a target
            # mismatch, the recovery mechanism is identical: pop back to the
            # most recent choice point with an untried candidate left.
            restored_idx = self._backtrack(choice_points, pddl_state, dead_states)
            if restored_idx is None:
                # No choice point left to try -- this is a final, terminal
                # outcome. Which one depends on why we got here:
                if blocked_step is not None:
                    # Real structural failure: never reached the end at all.
                    return _WalkResult(
                        reached_end=False, matches_target=False,
                        error_step=blocked_step, error_reason=blocked_reason,
                        steps=steps, final_marking=marking, final_attributes=pddl_state,
                        accumulated_duration_seconds=duration_total, warnings=warnings,
                    )
                # Reached the end on every attempt, just never with the
                # right attribute values -- structurally fine, but the
                # domain never explains the trace's real continuation (see
                # open question #3 in the implementation plan: this is
                # reported distinctly from a structural block, with no
                # single failing step).
                return _WalkResult(
                    reached_end=True, matches_target=False,
                    error_step=None, error_reason=mismatch_reason,
                    steps=steps, final_marking=marking, final_attributes=pddl_state,
                    accumulated_duration_seconds=duration_total, warnings=warnings,
                )

            # _backtrack already restored pddl_state in place (to
            # cp.state_before) and returned cp.step_index as restored_idx —
            # everything below just brings the OTHER walk state (steps,
            # marking, duration_total) back in sync with that same point,
            # then the outer `while True` loops back into the inner walk
            # from step_idx = restored_idx (no explicit `continue` needed:
            # falling off the end of the outer loop body re-enters it).
            #
            # steps[i] was appended exactly when step_idx advanced past i, so
            # len(steps) == step_idx invariably held right before a block
            # (blocking is detected before appending) — truncating to
            # restored_idx drops only the steps beyond the point being retried.
            steps = steps[:restored_idx]
            step_idx = restored_idx
            # Re-simulate the marking up to restored_idx: cheap re-derivation
            # from the walk's own starting marking keeps _advance_marking as
            # the single source of truth instead of snapshotting marking per
            # choice point too. Marking never depends on which effect group
            # was chosen (it only depends on the fixed Phase-1-resolved
            # transition sequence), so replaying it from scratch is always
            # correct, not just a cheap shortcut.
            marking = set(initial_marking)
            for prior in firing_steps[:restored_idx]:
                self._advance_marking(marking, prior)
            # duration_seconds is a property of the FiringStep itself
            # (lifecycle timestamps), independent of which effect group was
            # chosen — safe to re-derive the same way as the marking.
            if duration_total is not None:
                duration_total = sum(
                    (s.duration_seconds or 0.0) for s in firing_steps[:restored_idx]
                )

    def _advance_marking(self, marking: Set[str], step: FiringStep) -> None:
        """Move tokens per step: consume from_places, produce the transition's
        static output places (FiringStep only records from_places; output
        places are looked up via PetriNetModel.trans_outputs, keyed by the
        same Transition object FiringStep.transition holds — see
        models.FiringStep's docstring)."""
        marking.difference_update({p.name for p in step.from_places})
        marking.update({
            p.name for p in self.petrinet_model.trans_outputs.get(step.transition, set())
        })

    @staticmethod
    def _backtrack(
        choice_points: List[_EffectChoicePoint],
        pddl_state: Dict[str, str],
        dead_states: Set[Tuple[int, FrozenSet[Tuple[str, str]]]],
    ) -> Optional[int]:
        """Try the next untried alternative at the most recent choice point.

        Mutates pddl_state in place (restored to the choice point's
        state_before) and truncates choice_points to drop any points pushed
        after the one being retried (they belonged to the abandoned forward
        attempt and will be freshly recreated if reached again).

        Returns the step index to resume from, or None if every choice point
        is exhausted (no backtracking possible — matches
        PartialTraceReplayer._backtrack raising once split_stack is empty).
        """
        while choice_points:
            cp = choice_points[-1]
            cp.tried_count += 1
            if cp.tried_count < len(cp.candidates):
                # Found an untried alternative at the top of the stack --
                # restore the exact attribute state this choice point had
                # when it was first created (before any of its candidates
                # were applied) and hand back where to resume from.
                pddl_state.clear()
                pddl_state.update(cp.state_before)
                return cp.step_index
            # This choice point's every candidate has now been tried without
            # success -- discard it and look at the next one down the stack
            # (an earlier choice further back in the walk). Memoize
            # (step_index, state_before) as dead first: any other
            # backtracking branch that later lands on this exact same
            # (position, attribute state) pair -- e.g. a repeated ambiguous
            # transition inside a process loop -- is guaranteed, by
            # firing_steps being fixed and deterministic, to rediscover the
            # identical exhaustive failure; recording it here lets _walk
            # skip straight past it instead of redoing that work.
            dead_states.add((cp.step_index, frozenset(cp.state_before.items())))
            choice_points.pop()
        # Stack exhausted: no earlier decision, anywhere in the walk, could
        # possibly change the outcome.
        return None

    @staticmethod
    def _block_reason(activity_name: str, xor_ok: bool, effect_groups: List[PreparedEffectGroup]) -> str:
        if not xor_ok:
            return f"No XOR branch guard satisfied for '{activity_name}'."
        return (
            f"No effect group guard satisfied for '{activity_name}' "
            f"(and no alternative left to backtrack to)."
        )
