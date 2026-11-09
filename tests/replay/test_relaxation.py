"""replay.relaxation: the per-attribute necessary condition used to prune the replayer's search.

Steps and groups are plain stand-ins (the module only reads activity_name / is_tau and
assignments / guard), so the cases below are small enough to check by hand."""
import random
from types import SimpleNamespace

import pytest

from models import Guard, sop_holds
from replay.relaxation import AttributeRelaxation, project, projected_holds


def step(name, tau=False):
    return SimpleNamespace(activity_name=name, is_tau=tau)


def group(assignments=(), guard=()):
    return SimpleNamespace(assignments=list(assignments), guard=[list(c) for c in guard])


def relaxation(groups=None, xor=None, declared=None):
    groups, xor = groups or {}, xor or {}
    names = sorted(set(groups) | set(xor))
    return AttributeRelaxation(names, lambda a: groups.get(a, []), lambda a: xor.get(a, []), declared or {})


def G(attr, value, negated=False):
    return Guard(attribute=attr, value=value, negated=negated)


class TestProjection:
    def test_keeps_only_the_literals_on_the_attribute(self):
        sop = [[G("f", "x"), G("g", "y")], [G("f", "z")]]
        assert project(sop, "f") == frozenset({frozenset({G("f", "x")}), frozenset({G("f", "z")})})
        assert project([[G("f", "x"), G("g", "y")]], "g") == frozenset({frozenset({G("g", "y")})})

    def test_a_clause_without_literals_on_the_attribute_leaves_it_unconstrained(self):
        assert project([[G("g", "y")], [G("f", "x")]], "f") is None

    def test_no_guard_means_no_constraint(self):
        assert project([], "f") is None and project(None, "f") is None
        assert projected_holds(None, "f", "anything")

    def test_evaluation_uses_the_replays_own_semantics_for_missing_attributes(self):
        negated = project([[G("f", "x", negated=True)]], "f")
        assert projected_holds(negated, "f", "y")
        assert not projected_holds(negated, "f", "x")
        assert not projected_holds(negated, "f", None)  # a negated literal fails on a missing attribute
        boolean = project([[G("b", None)]], "b")  # value None reads as "true"
        assert projected_holds(boolean, "b", "true") and not projected_holds(boolean, "b", "false")


class TestViableSets:
    def test_a_later_step_can_repair_a_value_an_earlier_step_got_wrong(self):
        r = relaxation({"A": [group([("f", "x")]), group([("f", "y")])], "B": [group([("f", "z")])]})
        v = r.build([step("A"), step("B")], {"f": "z"})
        assert v.state_ok(0, {"f": "x"}) and v.state_ok(0, {})
        assert v.viable[2]["f"] == frozenset({"z"})
        # a value the table never saw cannot be refuted, so it is not pruned (soundness first)
        assert v.state_ok(1, {"f": "never-seen"})

    def test_a_value_nobody_can_write_is_not_viable(self):
        r = relaxation({"A": [group([("f", "x")]), group([("f", "y")])]})
        v = r.build([step("A")], {"f": "z"})
        assert not v.state_ok(0, {"f": "x"}) and not v.state_ok(0, {})

    def test_a_group_that_does_not_assign_the_attribute_lets_the_value_stay(self):
        r = relaxation({"A": [group([("f", "x")]), group([("g", "y")])]}, declared={"f": ["w"]})  # 2nd group skips f
        v = r.build([step("A")], {"f": "z"})
        assert v.state_ok(0, {"f": "z"}) and not v.state_ok(0, {"f": "w"})

    def test_a_forced_overwrite_with_the_wrong_value_kills_even_a_matching_state(self):
        r = relaxation({"A": [group([("f", "x")])]})  # every group assigns f=x
        v = r.build([step("A")], {"f": "z"})
        assert not v.state_ok(0, {"f": "z"})

    def test_an_xor_guard_that_contradicts_the_target_is_detected_without_any_search(self):
        # the silent step needs f == y; nothing writes f after it; the observed final value is x
        r = relaxation({"A": [group([("f", "x")]), group([("f", "y")])]}, xor={"tau": [[G("f", "y")]]})
        v = r.build([step("A"), step("tau", tau=True)], {"f": "x"})
        assert not v.state_ok(0, {"f": "x"}) and not v.state_ok(0, {})

    def test_with_a_later_writer_the_same_guard_is_fine(self):
        r = relaxation({"A": [group([("f", "x")]), group([("f", "y")])], "C": [group([("f", "x")])]},
                       xor={"tau": [[G("f", "y")]]})
        v = r.build([step("A"), step("tau", tau=True), step("C")], {"f": "x"})
        assert v.state_ok(0, {"f": "x"}) and v.state_ok(0, {})  # A may pick y, tau passes, C restores x

    def test_group_guards_count_too(self):
        # the only group of B needs f == y, but A can only write x
        r = relaxation({"A": [group([("f", "x")])], "B": [group([], guard=[[G("f", "y")]])]})
        v = r.build([step("A"), step("B")], None)
        assert not v.state_ok(0, {})
        r2 = relaxation({"A": [group([("f", "x")]), group([("f", "y")])], "B": [group([], guard=[[G("f", "y")]])]})
        assert r2.build([step("A"), step("B")], None).state_ok(0, {})

    def test_without_a_target_only_the_guards_matter(self):
        r = relaxation({"A": [group([("f", "x")])]}, xor={"A": [[G("f", "q")]]}, declared={"f": ["other"]})
        v = r.build([step("A")], None)
        assert v.state_ok(0, {"f": "q"}) and not v.state_ok(0, {"f": "other"})

    def test_a_value_the_model_never_mentions_is_still_tracked_when_it_was_observed(self):
        r = relaxation({"A": [group([("g", "y")])]})  # nobody touches f
        v = r.build([step("A")], {"f": "unseen"}, observed_values={"f": ["unseen", "other"]})
        assert v.state_ok(0, {"f": "unseen"}) and not v.state_ok(0, {"f": "other"})

    def test_attributes_nobody_mentions_are_not_tracked(self):
        r = relaxation({"A": [group([("f", "x")])]})
        assert r.build([step("A")], {"f": "x"}).tracked == ["f"]


class TestCandidateFilter:
    def test_keeps_the_groups_that_leave_a_viable_state(self):
        good, bad_value, skips = group([("f", "z")]), group([("f", "x")]), group([("g", "y")])
        r = relaxation({"A": [good, bad_value, skips]})
        v = r.build([step("A")], {"f": "z"})
        # f is already z: assigning z or leaving it alone is fine, assigning x is not
        assert v.filter_candidates(0, {"f": "z"}, [good, bad_value, skips]) == [good, skips]
        # f is x: only the group that writes z repairs it
        assert v.filter_candidates(0, {"f": "x"}, [good, bad_value, skips]) == [good]

    def test_everything_is_pruned_when_no_group_can_repair_the_state(self):
        a, b = group([("f", "x")]), group([("g", "y")])
        r = relaxation({"A": [a, b]}, declared={"f": ["w"]})
        v = r.build([step("A")], {"f": "z"})
        assert v.filter_candidates(0, {"f": "w"}, [a, b]) == []


class TestFirstBlockedStep:
    def test_names_the_step_and_the_kind_of_guard(self):
        r = relaxation({"A": [group([("f", "x")])], "B": [group([], guard=[[G("f", "z")]])]},
                       xor={"tau": [[G("f", "x")]]})
        steps = [step("A"), step("tau", tau=True), step("B")]
        v = r.build(steps, None)
        assert v.first_blocked_step({}) == (2, "effect")  # B's only group needs f == z
        r2 = relaxation({"A": [group([("f", "x")])]}, xor={"tau": [[G("f", "q")]]})
        assert r2.build([step("A"), step("tau", tau=True)], None).first_blocked_step({}) == (1, "xor")

    def test_none_when_nothing_blocks(self):
        r = relaxation({"A": [group([("f", "x")])]})
        assert r.build([step("A")], None).first_blocked_step({}) is None


# ----------------------------------------------------------------------------------------------
# Soundness: the relaxation must never call a state non-viable when a real continuation exists.
# ----------------------------------------------------------------------------------------------

VALUES = ["a", "b", "c"]
ATTRS = ["p", "q", "r"]


def _random_sop(rng, max_clauses=2, max_lits=2):
    clauses = []
    for _ in range(rng.randint(0, max_clauses)):
        clauses.append([G(rng.choice(ATTRS), rng.choice(VALUES), negated=rng.random() < 0.3)
                        for _ in range(rng.randint(0, max_lits))])
    return clauses


def _random_case(rng):
    n = rng.randint(2, 5)
    steps, groups, xor = [], {}, {}
    for i in range(n):
        tau = rng.random() < 0.25
        name = f"s{i}"
        steps.append(step(name, tau=tau))
        if rng.random() < 0.5:
            xor[name] = _random_sop(rng)
        if not tau and rng.random() < 0.8:
            groups[name] = [
                group([(a, rng.choice(VALUES)) for a in rng.sample(ATTRS, rng.randint(0, 3))],
                      guard=_random_sop(rng))
                for _ in range(rng.randint(1, 4))
            ]
    initial = {a: rng.choice(VALUES) for a in ATTRS if rng.random() < 0.7}
    target = {a: rng.choice(VALUES) for a in ATTRS if rng.random() < 0.5} if rng.random() < 0.8 else None
    return steps, groups, xor, initial, target


def _brute_force(steps, groups, xor, initial, target):
    """Exhaustive search with the replayer's real semantics (joint groups, full guards)."""
    def go(i, state):
        if i == len(steps):
            return target is None or all(state.get(a) == v for a, v in target.items())
        s = steps[i]
        if not sop_holds(xor.get(s.activity_name, []), state):
            return False
        gs = [] if s.is_tau else groups.get(s.activity_name, [])
        if not gs:
            return go(i + 1, state)
        return any(go(i + 1, {**state, **dict(g.assignments)}) for g in gs if sop_holds(g.guard, state))
    return go(0, dict(initial))


@pytest.mark.parametrize("seed", range(400))
def test_the_relaxation_never_prunes_a_feasible_tail(seed):
    rng = random.Random(seed)
    steps, groups, xor, initial, target = _random_case(rng)
    r = relaxation(groups, xor, declared={a: VALUES for a in ATTRS})
    viability = r.build(steps, target, observed_values={a: [v] for a, v in initial.items()})
    if _brute_force(steps, groups, xor, initial, target):
        assert viability.state_ok(0, initial), (seed, "pruned a feasible tail")
        # and the candidate filter keeps at least one group of the first labeled step it meets
        first = steps[0]
        gs = [] if first.is_tau else groups.get(first.activity_name, [])
        real = [g for g in gs if sop_holds(g.guard, initial) and sop_holds(xor.get(first.activity_name, []), initial)]
        if real and _brute_force(steps[1:], groups, xor, {**initial, **dict(real[0].assignments)}, target):
            assert viability.filter_candidates(0, initial, real), (seed, "filtered out every good group")


def test_the_relaxation_is_not_vacuous_it_does_reject_some_infeasible_tails():
    rejected = infeasible = 0
    for seed in range(400):
        steps, groups, xor, initial, target = _random_case(random.Random(seed))
        r = relaxation(groups, xor, declared={a: VALUES for a in ATTRS})
        v = r.build(steps, target, observed_values={a: [x] for a, x in initial.items()})
        if not _brute_force(steps, groups, xor, initial, target):
            infeasible += 1
            rejected += 0 if v.state_ok(0, initial) else 1
    assert infeasible > 50 and rejected > infeasible * 0.4  # a useful necessary condition, not just a sound one


# ----------------------------------------------------------------------------------------------
# The walk itself: with and without the relaxation, against an exhaustive search.
# ----------------------------------------------------------------------------------------------

def _walk_case(seed, relaxation_on):
    from encoding.prepared_input import PreparedDomainInput, PreparedEffectGroup, PreparedTransition
    from models import AnalysisConfig, AttributeCatalogEntry
    from replay.trace_replayer import TraceReplayer

    rng = random.Random(seed)
    steps, groups, xor, initial, target = _random_case(rng)
    transitions = {
        name: PreparedTransition(
            activity_name=name, input_places=[], preconditions=[],
            effect_groups=[PreparedEffectGroup(assignments=list(g.assignments), guard=g.guard, probability=1.0)
                           for g in gs])
        for name, gs in groups.items()
    }
    catalog = {a: AttributeCatalogEntry("categorical", set(VALUES), None) for a in ATTRS}
    prepared = PreparedDomainInput(transitions=transitions, xor_branches={}, attribute_catalog=catalog)
    config = AnalysisConfig(replay_relaxation=relaxation_on)
    replayer = TraceReplayer(prepared, SimpleNamespace(trans_outputs={}, initial_marking=[]), config)
    replayer._xor_guard_cache.update({name: sop for name, sop in xor.items()})
    firing = [SimpleNamespace(activity_name=s.activity_name, is_tau=s.is_tau, attributes={},
                              duration_seconds=0.0, from_places=set(), transition=None) for s in steps]
    relax = replayer._relax_context(firing, target, [initial]) if relaxation_on else None
    result = replayer._walk(firing, set(), dict(initial), target, False, relax=relax)
    return steps, groups, xor, initial, target, result


@pytest.mark.parametrize("seed", range(300))
def test_the_walk_gives_the_same_verdicts_with_the_relaxation_as_by_exhaustive_search(seed):
    steps, groups, xor, initial, target, pruned = _walk_case(seed, True)
    _, _, _, _, _, plain = _walk_case(seed, False)
    matches = _brute_force(steps, groups, xor, initial, target)
    reaches = _brute_force(steps, groups, xor, initial, None)  # a guard-respecting path, target aside
    assert pruned.matches_target == matches == plain.matches_target, (seed, "matches")
    assert pruned.reached_end == reaches == plain.reached_end, (seed, "reached_end")
    assert not (matches and not reaches)  # sanity of the oracle


# ----------------------------------------------------------------------------------------------
# Bulk guard evaluation (_GuardIndex) = group-by-group evaluation.
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(60))
def test_the_guard_index_selects_exactly_what_sop_holds_selects_in_the_same_order(seed):
    from encoding.prepared_input import PreparedEffectGroup
    from replay.trace_replayer import _GuardIndex

    rng = random.Random(seed)
    groups = [
        PreparedEffectGroup(assignments=[(rng.choice(ATTRS), rng.choice(VALUES))], guard=_random_sop(rng, 3, 3),
                            probability=rng.random())
        for _ in range(rng.randint(64, 200))
    ]
    index = _GuardIndex.build(groups)
    assert index is not None
    for _ in range(25):
        state = {a: rng.choice(VALUES) for a in ATTRS if rng.random() < 0.7}
        expected = [g for g in groups if sop_holds(g.guard, state)]
        got = index.matching(state)
        assert [id(g) for g in got] == [id(g) for g in expected]


def test_the_guard_index_is_only_built_for_activities_with_many_groups():
    from encoding.prepared_input import PreparedEffectGroup
    from replay.trace_replayer import _GuardIndex
    few = [PreparedEffectGroup(assignments=[], guard=[], probability=1.0)] * 10
    assert _GuardIndex.build(few) is None
    many = [PreparedEffectGroup(assignments=[], guard=[], probability=1.0)] * 100
    assert _GuardIndex.build(many) is not None  # no guard at all: every group passes
    assert len(_GuardIndex.build(many).matching({})) == 100
