"""Tests for replay.trace_replayer.TraceReplayer (casi 1/2/3 of
docs/trace_replayer_analysis.md §2).

tau_seq_net (tests/replay/conftest.py) is a real, fully-wired pm4py net —
Phase 1 (replay._log_alignment) runs actual pm4py conformance checking, so
these are closer to integration tests than the mocked-pm4py style used in
tests/parsing/test_petri_net_log_builder.py.
"""
from tests.helpers import make_event, make_trace

from encoding.prepared_input import PreparedDomainInput, PreparedEffectGroup, PreparedTransition
from models import ActionDurationStats, AnalysisConfig, AttributeCatalogEntry, FiringStep, Guard
from replay.trace_replayer import TraceReplayer, _sum_modeled_max_duration

_FLAG_RISK_CATALOG = {
    "flag": AttributeCatalogEntry(attribute_type="categorical", possible_values={"x", "y"}, bin_boundaries=None),
    "risk": AttributeCatalogEntry(attribute_type="categorical", possible_values={"high", "low"}, bin_boundaries=None),
}


def _replayer(tau_seq_net, transitions, catalog=None):
    prepared = PreparedDomainInput(
        transitions=transitions, xor_branches={},
        attribute_catalog=catalog or _FLAG_RISK_CATALOG, nodes=[], edges=[],
    )
    return TraceReplayer(prepared, tau_seq_net, AnalysisConfig())


class TestReplayFullTraceNoAmbiguity:
    def test_single_effect_group_applies_directly(self, tau_seq_net):
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=1.0)],
        )
        trans_b = PreparedTransition(
            activity_name="B", input_places=["p_mid2"], preconditions=[], effect_groups=[],
        )
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(make_event("A"), make_event("B"))

        outcome = replayer.replay_full_trace(trace)

        assert outcome.is_replayable
        assert outcome.error_step is None
        assert outcome.final_attributes == {"flag": "x"}
        assert outcome.final_marking == {"p_out"}
        assert [s.activity_name for s in outcome.steps] == ["A", "tau_1", "B"]
        assert outcome.steps[1].is_tau is True

    def test_no_effect_groups_fires_with_no_attribute_effects(self, tau_seq_net):
        trans_a = PreparedTransition(activity_name="A", input_places=["p_in"], preconditions=[], effect_groups=[])
        replayer = _replayer(tau_seq_net, {"A": trans_a})
        trace = make_trace(make_event("A"), make_event("B"))

        outcome = replayer.replay_full_trace(trace)

        assert outcome.is_replayable
        assert outcome.final_attributes == {}
        assert outcome.steps[0].attributes_applied == {}


class TestEffectGroupBacktracking:
    """The only real choice point: which effect group a labeled transition
    applied. Ambiguous when >1 group's guard is satisfiable — resolved by
    scoring against the observed event, backtracking if the greedy pick
    leads to a dead end later (see trace_replayer.py's module docstring)."""

    def _ambiguous_prepared(self):
        # A's two effect groups are both unguarded (always "satisfiable"),
        # so choosing between them is pure ambiguity, tie-broken by score
        # against the observed event (which carries no matching attribute
        # here, so both score 0 and the greedy pick keeps list order: group_a1).
        group_a1 = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_a2 = PreparedEffectGroup(assignments=[("flag", "y")], guard=[], probability=0.5)
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[group_a1, group_a2],
        )
        # B can only fire if flag=y — forces backtracking away from the
        # greedy first pick (group_a1, flag=x).
        group_b = PreparedEffectGroup(
            assignments=[("risk", "high")], guard=[[Guard("flag", "y", False)]], probability=1.0,
        )
        trans_b = PreparedTransition(
            activity_name="B", input_places=["p_mid2"], preconditions=[], effect_groups=[group_b],
        )
        return {"A": trans_a, "B": trans_b}

    def test_backtracks_to_the_alternative_that_unblocks_later_step(self, tau_seq_net):
        replayer = _replayer(tau_seq_net, self._ambiguous_prepared())
        trace = make_trace(make_event("A"), make_event("B"))

        outcome = replayer.replay_full_trace(trace)

        assert outcome.is_replayable
        assert outcome.final_attributes == {"flag": "y", "risk": "high"}
        assert outcome.steps[0].attributes_applied == {"flag": "y"}
        assert outcome.steps[2].attributes_applied == {"risk": "high"}

    def test_greedy_pick_preferred_when_observed_event_matches_it(self, tau_seq_net):
        """When the observed event's own attribute matches one candidate,
        that candidate is scored higher and chosen without any backtrack
        being needed — B's guard here accepts either flag value."""
        group_a1 = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_a2 = PreparedEffectGroup(assignments=[("flag", "y")], guard=[], probability=0.5)
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[group_a1, group_a2],
        )
        trans_b = PreparedTransition(activity_name="B", input_places=["p_mid2"], preconditions=[], effect_groups=[])
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(make_event("A", flag="y"), make_event("B"))

        outcome = replayer.replay_full_trace(trace)

        assert outcome.is_replayable
        assert outcome.final_attributes == {"flag": "y"}

    def test_stops_when_no_alternative_can_unblock(self, tau_seq_net):
        group_a1 = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_a2 = PreparedEffectGroup(assignments=[("flag", "y")], guard=[], probability=0.5)
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[group_a1, group_a2],
        )
        # No value of flag satisfies B's guard -> unrecoverable.
        group_b = PreparedEffectGroup(assignments=[], guard=[[Guard("flag", "z", False)]], probability=1.0)
        trans_b = PreparedTransition(
            activity_name="B", input_places=["p_mid2"], preconditions=[], effect_groups=[group_b],
        )
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(make_event("A"), make_event("B"))

        outcome = replayer.replay_full_trace(trace)

        assert not outcome.is_replayable
        assert outcome.error_step == 2
        assert "B" in outcome.error_reason


class TestLoopSafety:
    """A process loop repeats the same ambiguous transition many times in
    one trace -- _walk's DFS-with-backtracking is otherwise a
    product-of-candidates-per-choice-point in the worst case (see the
    module's loop-blowup analysis). _collapse_no_impact and dead_states
    mitigate this without changing correctness; these tests exercise both
    on a real repeating transition (loop_net: p_loop -> A -> p_loop, exits
    via B)."""

    def test_all_candidates_no_impact_on_repeat_firing(self, loop_net):
        # 3 candidates for "A": one sets both attributes at once, the other
        # two set one each. Seed flag=x/risk=high via the log-observed
        # prefix (replay_evaluation_split's cut point derives split_attributes
        # from the raw event, no effect-group interpretation -- see
        # _build_snapshots) so the FIRST tail "A" firing already sees the
        # state all three candidates would produce -- i.e. ALL THREE are
        # simultaneously no-impact right from the start, deliberately
        # sidestepping _rank_candidates' own tie-break preference for
        # "leaner" candidates (which would otherwise favor a singleton group
        # over group_both on an ambiguous FIRST firing and never exercise
        # the all-no-impact collapse at all).
        group_both = PreparedEffectGroup(assignments=[("flag", "x"), ("risk", "high")], guard=[], probability=1.0)
        group_flag_only = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_risk_only = PreparedEffectGroup(assignments=[("risk", "high")], guard=[], probability=0.5)
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_loop"], preconditions=[],
            effect_groups=[group_both, group_flag_only, group_risk_only],
        )
        trans_b = PreparedTransition(activity_name="B", input_places=["p_loop"], preconditions=[], effect_groups=[])
        replayer = _replayer(loop_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(
            make_event("A", flag="x", risk="high"),  # seed event, stays before the cut
            make_event("A"),                          # first TAIL firing -- all 3 candidates no-impact
            make_event("B", flag="x", risk="high"),
        )

        outcome = replayer.replay_evaluation_split(trace, n_prefix=1)

        assert outcome.reached_end
        assert outcome.matches_expected_final_state
        assert outcome.split_attributes == {"flag": "x", "risk": "high"}
        assert outcome.final_attributes == {"flag": "x", "risk": "high"}

    def test_repeated_no_impact_group_collapses_to_a_single_candidate(self, loop_net):
        # Two candidates that, once flag is already "x", are BOTH no-ops --
        # group_reassert re-asserts the same value flag=x already holds,
        # group_noop_risk re-asserts risk=high which was set by the first
        # "A" firing too. On the second "A" firing both are no-impact
        # simultaneously -> must collapse to a single representative, so no
        # _EffectChoicePoint (and therefore no ambiguity/backtracking) is
        # needed for that firing at all.
        group_flag = PreparedEffectGroup(assignments=[("flag", "x"), ("risk", "high")], guard=[], probability=1.0)
        group_reassert = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_noop_risk = PreparedEffectGroup(assignments=[("risk", "high")], guard=[], probability=0.5)
        trans_a_first = PreparedTransition(
            activity_name="A", input_places=["p_loop"], preconditions=[], effect_groups=[group_flag],
        )
        replayer = _replayer(loop_net, {"A": trans_a_first, "B": PreparedTransition(
            activity_name="B", input_places=["p_loop"], preconditions=[], effect_groups=[],
        )})
        # Swap in the ambiguous-but-all-no-impact effect groups only for the
        # SECOND firing by mutating the shared PreparedTransition's
        # effect_groups after the fact is awkward with one dict entry per
        # activity name (the replayer looks up "A" once per step, same
        # PreparedTransition every time) -- instead, verify the collapse
        # directly via the module-level helper, which is what _walk calls
        # at every firing.
        from replay.trace_replayer import _collapse_no_impact
        pddl_state = {"flag": "x", "risk": "high"}
        candidates = [group_reassert, group_noop_risk]
        collapsed = _collapse_no_impact(candidates, pddl_state)
        assert len(collapsed) == 1
        assert collapsed[0] in candidates

    def test_mixed_impact_candidates_are_not_collapsed(self, loop_net):
        # Only genuinely-redundant (no-impact) candidates get merged --
        # group_real still changes state and must survive collapsing
        # alongside the one kept no-impact representative.
        from replay.trace_replayer import _collapse_no_impact
        group_noop = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_real = PreparedEffectGroup(assignments=[("flag", "y")], guard=[], probability=0.5)
        pddl_state = {"flag": "x"}
        collapsed = _collapse_no_impact([group_noop, group_real], pddl_state)
        assert len(collapsed) == 2

    def test_evaluation_split_terminates_when_repeated_ambiguity_cannot_match_target(self, loop_net):
        # Two REAL (non-collapsible) candidates on every "A" firing, none of
        # which ever touch "risk" -- the target snapshot below requires
        # risk=high, which no combination of A's choices can ever produce,
        # so every branch of the exhaustive backtrack is doomed. Must still
        # terminate (not hang) and correctly report a mismatch rather than
        # a false positive.
        group_x = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_y = PreparedEffectGroup(assignments=[("flag", "y")], guard=[], probability=0.5)
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_loop"], preconditions=[],
            effect_groups=[group_x, group_y],
        )
        trans_b = PreparedTransition(activity_name="B", input_places=["p_loop"], preconditions=[], effect_groups=[])
        replayer = _replayer(loop_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(
            make_event("A"), make_event("A"), make_event("A"), make_event("A"),
            make_event("B", risk="high"),
        )

        outcome = replayer.replay_evaluation_split(trace, n_prefix=0)

        assert outcome.reached_end is True
        assert outcome.matches_expected_final_state is False
        assert outcome.error_step is None


class TestReplayPartialTrace:
    """Caso 3: only Phase 1 + the raw final snapshot — no guard/effect-group
    walk at all (docs/trace_replayer_analysis.md: "il replay vero e proprio
    con i firing step non è richiesto in questo caso")."""

    def test_snapshot_is_raw_log_observed_state_not_pddl_simulated(self, tau_seq_net):
        # A's effect group would assign flag=x if Phase 2 ran, but it never
        # does here — the raw snapshot only reflects what the log event
        # itself carries.
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=1.0)],
        )
        replayer = _replayer(tau_seq_net, {"A": trans_a})
        trace = make_trace(make_event("A"))  # trace stops after A — genuinely partial

        snapshot = replayer.replay_partial_trace(trace)

        assert snapshot.marking == {"p_mid1"}
        assert snapshot.attributes == {}  # nothing observed on A's own event

    def test_snapshot_accumulates_observed_event_attributes(self, tau_seq_net):
        trans_a = PreparedTransition(activity_name="A", input_places=["p_in"], preconditions=[], effect_groups=[])
        replayer = _replayer(tau_seq_net, {"A": trans_a})
        trace = make_trace(make_event("A", flag="y"))

        snapshot = replayer.replay_partial_trace(trace)

        assert snapshot.attributes == {"flag": "y"}


class TestReplayEvaluationSplit:
    """Caso 2: Phase 1 always runs on the whole trace (never truncated —
    unlike the removed replay_with_split, which used to truncate before
    Phase 1 and could desync an "alignments"-engine replay from the true
    optimal alignment of the full trace). The tail (cut -> end) is then
    walked verifying it reaches the end matching the real final snapshot,
    retrying alternate effect-group choices if it doesn't."""

    def test_split_and_final_snapshots_and_tail_walk_are_consistent(self, tau_seq_net):
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=1.0)],
        )
        trans_b = PreparedTransition(
            activity_name="B", input_places=["p_mid2"], preconditions=[],
            effect_groups=[PreparedEffectGroup(assignments=[("risk", "high")], guard=[], probability=1.0)],
        )
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(make_event("A", flag="x"), make_event("B", risk="high"))

        outcome = replayer.replay_evaluation_split(trace, n_prefix=1)

        assert outcome.split_marking == {"p_mid1"}
        assert outcome.split_attributes == {"flag": "x"}  # observed directly on A's own event
        assert outcome.expected_final_attributes == {"flag": "x", "risk": "high"}
        assert outcome.reached_end
        assert outcome.matches_expected_final_state
        assert outcome.final_marking == {"p_out"}
        assert outcome.final_attributes == {"flag": "x", "risk": "high"}
        assert [s.activity_name for s in outcome.steps] == ["tau_1", "B"]

    def test_extra_attributes_beyond_the_target_still_match(self, tau_seq_net):
        """The match check is a subset test (expected_final_attributes ⊆
        final_attributes), not exact equality: B's effect group sets an
        extra attribute ("risk") the real log never observed at all (so it
        is absent from expected_final_attributes) -- that alone must not
        cause a mismatch, since we don't care about having *more* than
        expected, only about not having less/wrong values."""
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=1.0)],
        )
        trans_b = PreparedTransition(
            activity_name="B", input_places=["p_mid2"], preconditions=[],
            effect_groups=[PreparedEffectGroup(assignments=[("risk", "high")], guard=[], probability=1.0)],
        )
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(make_event("A", flag="x"), make_event("B"))  # B's own event carries no "risk"

        outcome = replayer.replay_evaluation_split(trace, n_prefix=1)

        assert outcome.expected_final_attributes == {"flag": "x"}  # "risk" never observed
        assert outcome.final_attributes == {"flag": "x", "risk": "high"}  # but B's effect group sets it anyway
        assert outcome.reached_end
        assert outcome.matches_expected_final_state

    def test_mismatch_forces_backtrack_to_alternative_effect_group(self, tau_seq_net):
        """A's greedy pick (flag=x, no hint on A's own event -> tie, keeps
        list order) reaches the end structurally fine (B's effect group is
        unguarded, fires regardless of flag), but the resulting flag value
        doesn't match the real observed one (flag=y, carried on B's own
        event) -- this is a pure end-of-walk mismatch, not a guard failure,
        and must still trigger a backtrack to A's other candidate."""
        group_a1 = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_a2 = PreparedEffectGroup(assignments=[("flag", "y")], guard=[], probability=0.5)
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[group_a1, group_a2],
        )
        trans_b = PreparedTransition(
            activity_name="B", input_places=["p_mid2"], preconditions=[],
            effect_groups=[PreparedEffectGroup(assignments=[("risk", "high")], guard=[], probability=1.0)],
        )
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(make_event("A"), make_event("B", flag="y", risk="high"))

        outcome = replayer.replay_evaluation_split(trace, n_prefix=0)

        assert outcome.reached_end
        assert outcome.matches_expected_final_state
        assert outcome.final_attributes == {"flag": "y", "risk": "high"}
        assert outcome.steps[0].attributes_applied == {"flag": "y"}

    def test_mismatch_exhausting_every_alternative_is_reported_distinctly(self, tau_seq_net):
        """Neither candidate ever assigns "risk" at all, but the real log
        observes risk=high somewhere -- since a match only requires the
        expected attributes to be a SUBSET of the final state (extra
        attributes are fine, but missing/wrong expected ones are not), this
        can never be satisfied regardless of which flag candidate is chosen.
        reached_end stays True (structurally nothing ever blocks), but
        matches_expected_final_state is False, with no single failing step
        (error_step is None, error_reason still explains why)."""
        group_a1 = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_a2 = PreparedEffectGroup(assignments=[("flag", "y")], guard=[], probability=0.5)
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[group_a1, group_a2],
        )
        replayer = _replayer(tau_seq_net, {"A": trans_a})
        trace = make_trace(make_event("A", risk="high"))

        outcome = replayer.replay_evaluation_split(trace, n_prefix=0)

        assert outcome.reached_end
        assert not outcome.matches_expected_final_state
        assert outcome.error_step is None
        assert outcome.error_reason is not None

    def test_structural_block_is_distinct_from_a_mismatch(self, tau_seq_net):
        group_a1 = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_a2 = PreparedEffectGroup(assignments=[("flag", "y")], guard=[], probability=0.5)
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[group_a1, group_a2],
        )
        # No flag value ever satisfies B's guard -> a genuine structural block.
        trans_b = PreparedTransition(
            activity_name="B", input_places=["p_mid2"], preconditions=[],
            effect_groups=[PreparedEffectGroup(assignments=[], guard=[[Guard("flag", "z", False)]], probability=1.0)],
        )
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(make_event("A"), make_event("B"))

        outcome = replayer.replay_evaluation_split(trace, n_prefix=0)

        assert not outcome.reached_end
        assert not outcome.matches_expected_final_state
        assert outcome.error_step == 2
        assert "B" in outcome.error_reason

    def test_deadline_accumulates_duration_only_from_the_cut_onward(self, tau_seq_net):
        trans_a = PreparedTransition(activity_name="A", input_places=["p_in"], preconditions=[], effect_groups=[])
        trans_b = PreparedTransition(activity_name="B", input_places=["p_mid2"], preconditions=[], effect_groups=[])
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        # A takes 100s (before the cut -> must not count), B takes 5s (after).
        trace = make_trace(
            make_event("A", lifecycle="start", offset=0),
            make_event("A", lifecycle="complete", offset=100),
            make_event("B", lifecycle="start", offset=100),
            make_event("B", lifecycle="complete", offset=105),
        )

        within = replayer.replay_evaluation_split(
            trace, n_prefix=1, check_deadline=True, deadline_seconds=10.0,
        )
        exceeded = replayer.replay_evaluation_split(
            trace, n_prefix=1, check_deadline=True, deadline_seconds=1.0,
        )
        not_checked = replayer.replay_evaluation_split(trace, n_prefix=1)

        assert within.accumulated_duration_seconds == 5.0
        assert within.within_deadline is True
        assert exceeded.within_deadline is False
        assert not_checked.accumulated_duration_seconds is None
        assert not_checked.within_deadline is None

    def test_n_prefix_beyond_available_events_warns_and_uses_full_trace(self, tau_seq_net):
        trans_a = PreparedTransition(activity_name="A", input_places=["p_in"], preconditions=[], effect_groups=[])
        replayer = _replayer(tau_seq_net, {"A": trans_a})
        trace = make_trace(make_event("A"))

        outcome = replayer.replay_evaluation_split(trace, n_prefix=99)

        assert outcome.split_marking == outcome.final_marking
        assert any("exceeds" in w for w in outcome.warnings)


def _duration(effective_min: float, effective_max: float) -> ActionDurationStats:
    return ActionDurationStats(effective_min=effective_min, effective_max=effective_max, source="external")


def _step(activity_name: str, is_tau: bool = False) -> FiringStep:
    return FiringStep(
        transition=None, activity_name=activity_name, is_tau=is_tau,
        from_places=set(), attributes={},
    )


class TestSumModeledMaxDuration:
    """Unit tests for the module-level helper _sum_modeled_max_duration --
    called directly on synthetic FiringStep/PreparedTransition objects, no
    real Petri net or Phase 1 alignment needed."""

    def test_sums_effective_max_across_fired_transitions(self):
        transitions = {
            "A": PreparedTransition(activity_name="A", input_places=[], preconditions=[], effect_groups=[], duration=_duration(1.0, 10.0)),
            "B": PreparedTransition(activity_name="B", input_places=[], preconditions=[], effect_groups=[], duration=_duration(2.0, 20.0)),
        }
        firing_steps = [_step("A"), _step("B")]
        assert _sum_modeled_max_duration(firing_steps, transitions) == 30.0

    def test_tau_firings_excluded(self):
        transitions = {"A": PreparedTransition(activity_name="A", input_places=[], preconditions=[], effect_groups=[], duration=_duration(1.0, 10.0))}
        firing_steps = [_step("tau_1", is_tau=True), _step("A")]
        assert _sum_modeled_max_duration(firing_steps, transitions) == 10.0

    def test_transition_without_duration_contributes_zero(self):
        transitions = {"A": PreparedTransition(activity_name="A", input_places=[], preconditions=[], effect_groups=[], duration=None)}
        firing_steps = [_step("A")]
        assert _sum_modeled_max_duration(firing_steps, transitions) == 0.0

    def test_transition_absent_from_map_contributes_zero(self):
        firing_steps = [_step("unknown")]
        assert _sum_modeled_max_duration(firing_steps, {}) == 0.0

    def test_repeated_firing_counted_each_time(self):
        transitions = {"A": PreparedTransition(activity_name="A", input_places=[], preconditions=[], effect_groups=[], duration=_duration(1.0, 5.0))}
        firing_steps = [_step("A"), _step("A"), _step("A")]
        assert _sum_modeled_max_duration(firing_steps, transitions) == 15.0

    def test_empty_firing_steps_is_zero(self):
        assert _sum_modeled_max_duration([], {}) == 0.0


class TestMaxModeledDurationInEvaluationSplit:
    """Integration-level: max_modeled_duration_seconds as wired through
    replay_evaluation_split, on a real (tau_seq_net) Phase 1 resolution."""

    def test_matches_sum_of_effective_max_for_tail_transitions(self, tau_seq_net):
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[], effect_groups=[],
            duration=_duration(1.0, 10.0),
        )
        trans_b = PreparedTransition(
            activity_name="B", input_places=["p_mid2"], preconditions=[], effect_groups=[],
            duration=_duration(2.0, 20.0),
        )
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(make_event("A"), make_event("B"))

        outcome = replayer.replay_evaluation_split(trace, n_prefix=0)

        # A (10.0) + tau_1 (silent, no PreparedTransition, contributes 0) + B (20.0)
        assert outcome.max_modeled_duration_seconds == 30.0

    def test_independent_of_which_effect_group_backtracking_chooses(self, tau_seq_net):
        # Two ambiguous, unguarded candidates for A -- whichever one Phase 2
        # ends up picking, the duration sum must be identical: duration is
        # a property of the transition (fixed by Phase 1), not of which
        # effect group is chosen.
        group_x = PreparedEffectGroup(assignments=[("flag", "x")], guard=[], probability=0.5)
        group_y = PreparedEffectGroup(assignments=[("flag", "y")], guard=[], probability=0.5)
        trans_a = PreparedTransition(
            activity_name="A", input_places=["p_in"], preconditions=[],
            effect_groups=[group_x, group_y], duration=_duration(1.0, 10.0),
        )
        trans_b = PreparedTransition(
            activity_name="B", input_places=["p_mid2"], preconditions=[], effect_groups=[],
            duration=_duration(2.0, 20.0),
        )
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(make_event("A"), make_event("B"))

        outcome = replayer.replay_evaluation_split(trace, n_prefix=0)

        assert outcome.reached_end
        assert outcome.max_modeled_duration_seconds == 30.0

    def test_zero_when_no_transition_has_duration_data(self, tau_seq_net):
        trans_a = PreparedTransition(activity_name="A", input_places=["p_in"], preconditions=[], effect_groups=[])
        trans_b = PreparedTransition(activity_name="B", input_places=["p_mid2"], preconditions=[], effect_groups=[])
        replayer = _replayer(tau_seq_net, {"A": trans_a, "B": trans_b})
        trace = make_trace(make_event("A"), make_event("B"))

        outcome = replayer.replay_evaluation_split(trace, n_prefix=0)

        assert outcome.max_modeled_duration_seconds == 0.0
