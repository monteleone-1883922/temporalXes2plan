"""Tests for encoding.domain_builder.DomainBuilder.build_prepared_input
and the effect-group derivation helper (_prepared_effect_groups).

Builds TransitionInfo/EffectInfo/XorBranchInfo/ParseResult by hand, using the
CURRENT dataclass signatures directly — deliberately not reusing the shared
fixtures in tests/encoding/conftest.py, which are stale WIP (built against an
older TransitionInfo/XorBranchInfo shape) and fail to construct today.
"""
import logging

from pm4py.objects.petri_net.obj import Marking, PetriNet

import pytest

from encoding.domain_builder import DomainBuilder, _prepared_effect_groups
from encoding.effect_group_builder import EffectGroupBudgetExceeded
from models import (
    ActionDurationStats,
    AttributeCatalogEntry,
    EffectGuards,
    EffectInfo,
    Guard,
    ParseResult,
    PetriNetModel,
    TransitionInfo,
)


def _sequence_petri_net_model() -> PetriNetModel:
    """p_start -> t_a (activity_a) -> p_mid -> t_b (activity_b) -> p_end."""
    net = PetriNet("seq")
    p_start = PetriNet.Place("p_start")
    p_mid = PetriNet.Place("p_mid")
    p_end = PetriNet.Place("p_end")
    net.places.update([p_start, p_mid, p_end])

    t_a = PetriNet.Transition("t_a", label="activity_a")
    t_b = PetriNet.Transition("t_b", label="activity_b")
    net.transitions.update([t_a, t_b])

    for src, tgt in [(p_start, t_a), (t_a, p_mid), (p_mid, t_b), (t_b, p_end)]:
        net.arcs.add(PetriNet.Arc(src, tgt))

    return PetriNetModel(
        petrinet=net,
        initial_marking=Marking({p_start: 1}),
        final_marking=Marking({p_end: 1}),
        activities={"activity_a", "activity_b"},
        silent_transitions={},
        trans_inputs={t_a: {p_start}, t_b: {p_mid}},
        trans_outputs={t_a: {p_mid}, t_b: {p_end}},
        xor_splits={},
        place_inputs={p_mid: [t_a], p_end: [t_b]},
    )


def _effect(
    attribute: str,
    presence_probability: float = 1.0,
    appearance_level: int = 1,
    appearance_guards=None,
    value_probabilities=None,
    value_level: int = 1,
    value_guards=None,
) -> EffectInfo:
    return EffectInfo(
        attribute=attribute,
        presence_probability=presence_probability,
        appearance_level=appearance_level,
        appearance_guards=appearance_guards,
        value_probabilities=value_probabilities or {},
        value_level=value_level,
        value_guards=value_guards,
    )


def _transition_info(effects, **kwargs) -> TransitionInfo:
    defaults = dict(
        activity_name="activity_a",
        input_places=["p_start"],
        total_firings=10,
        xor_branches=None,
        effects=effects,
    )
    defaults.update(kwargs)
    return TransitionInfo(**defaults)


class TestFullyDeterministicEffect:
    def test_always_added_single_group(self):
        info = _transition_info({
            "priority": _effect(
                "priority", value_probabilities={"high": 1.0},
            ),
        })
        groups = _prepared_effect_groups(info)
        assert len(groups) == 1
        assert groups[0].assignments == [("priority", "high")]
        assert groups[0].guard == []
        assert groups[0].probability == 1.0


class TestDtGuardedAppearance:
    def test_with_and_without_groups(self):
        appearance_guards = EffectGuards(
            subtype="appearance",
            guards={"appears": [[Guard("risk", "high")]]},
            total_samples=10,
            dt_accuracy=0.9,
        )
        info = _transition_info({
            "manager_approval": _effect(
                "manager_approval",
                appearance_level=1,
                appearance_guards=appearance_guards,
                value_probabilities={"approved": 1.0},
            ),
        })
        groups = _prepared_effect_groups(info)
        by_assignments = {tuple(g.assignments): g for g in groups}

        with_key = (("manager_approval", "approved"),)
        without_key: tuple = ()
        assert with_key in by_assignments
        assert without_key in by_assignments
        assert by_assignments[with_key].guard == [[Guard("risk", "high")]]
        assert by_assignments[without_key].guard == []


class TestDtGuardedValue:
    def test_per_value_groups(self):
        value_guards = EffectGuards(
            subtype="value",
            guards={
                "approved": [[Guard("score", "high")]],
                "rejected": [[Guard("score", "low")]],
            },
            total_samples=10,
            dt_accuracy=0.9,
        )
        info = _transition_info({
            "outcome": _effect(
                "outcome",
                appearance_level=1,
                appearance_guards=None,
                value_level=1,
                value_guards=value_guards,
                value_probabilities={"approved": 0.5, "rejected": 0.5},
            ),
        })
        groups = _prepared_effect_groups(info)
        by_assignments = {tuple(g.assignments): g for g in groups}

        assert (("outcome", "approved"),) in by_assignments
        assert (("outcome", "rejected"),) in by_assignments
        assert by_assignments[(("outcome", "approved"),)].guard == [[Guard("score", "high")]]
        assert by_assignments[(("outcome", "rejected"),)].guard == [[Guard("score", "low")]]


class TestStatisticalAppearanceAndValue:
    def test_probability_multiplication(self):
        info = _transition_info({
            "notes_added": _effect(
                "notes_added",
                appearance_level=2,
                presence_probability=0.3,
                value_level=2,
                value_probabilities={"short": 0.7, "long": 0.3},
            ),
        })
        groups = _prepared_effect_groups(info)
        by_assignments = {tuple(g.assignments): g for g in groups}

        assert by_assignments[()].probability == 0.7
        assert by_assignments[(("notes_added", "long"),)].probability == 0.09
        assert by_assignments[(("notes_added", "short"),)].probability == 0.21
        for g in groups:
            assert g.guard == []


class TestIncompatibleEffects:
    def test_forces_attribute_off(self):
        info = _transition_info(
            {
                "attribute_a": _effect("attribute_a", value_probabilities={"x": 1.0}),
                "attribute_b": _effect(
                    "attribute_b", appearance_level=1, appearance_guards=None,
                    value_level=2, value_probabilities={"y": 1.0},
                ),
            },
            incompatible_effects={frozenset({"attribute_a", "attribute_b"})},
        )
        groups = _prepared_effect_groups(info)
        assert len(groups) == 1
        assert groups[0].assignments == [("attribute_a", "x")]


class TestRelatedEffects:
    def test_forces_attribute_on_in_dt_phase(self):
        appearance_guards = EffectGuards(
            subtype="appearance",
            guards={"appears": [[Guard("g", "1")]]},
            total_samples=10,
            dt_accuracy=0.9,
        )
        info = _transition_info(
            {
                "attribute_a": _effect("attribute_a", value_probabilities={"x": 1.0}),
                "attribute_b": _effect(
                    "attribute_b",
                    appearance_level=1,
                    appearance_guards=appearance_guards,
                    value_probabilities={"y": 1.0},
                ),
            },
            related_effects={frozenset({"attribute_a", "attribute_b"})},
        )
        groups = _prepared_effect_groups(info)
        for g in groups:
            attrs = {a for a, _ in g.assignments}
            assert "attribute_b" in attrs


class TestContradictoryDtGuards:
    def test_conflicting_branches_are_dropped(self, caplog):
        c_guards = EffectGuards(
            subtype="appearance",
            guards={"appears": [[Guard("risk", "high")]]},
            total_samples=10,
            dt_accuracy=0.9,
        )
        d_guards = EffectGuards(
            subtype="appearance",
            guards={"appears": [[Guard("risk", "low")]]},
            total_samples=10,
            dt_accuracy=0.9,
        )
        info = _transition_info({
            "attribute_c": _effect(
                "attribute_c", appearance_level=1, appearance_guards=c_guards,
                value_probabilities={"x": 1.0},
            ),
            "attribute_d": _effect(
                "attribute_d", appearance_level=1, appearance_guards=d_guards,
                value_probabilities={"y": 1.0},
            ),
        })
        with caplog.at_level(logging.DEBUG, logger="domain_builder"):
            groups = _prepared_effect_groups(info)

        for g in groups:
            attrs = {a for a, _ in g.assignments}
            assert not ({"attribute_c", "attribute_d"} <= attrs)


class TestBuildPreparedDomainInputEndToEnd:
    def test_transitions_dict_populated(self):
        pnm = _sequence_petri_net_model()
        parse_result = ParseResult(
            petri_net_model=pnm,
            place_predecessors={},
            transition_predecessors={},
            transitions={
                "activity_a": _transition_info(
                    {"priority": _effect("priority", value_probabilities={"high": 1.0})},
                    activity_name="activity_a",
                    input_places=["p_start"],
                    duration=ActionDurationStats(
                        effective_min=1.0, effective_max=2.0, source="observed",
                    ),
                ),
                "activity_b": _transition_info(
                    {},
                    activity_name="activity_b",
                    input_places=["p_mid"],
                ),
            },
            start_place="p_start",
            end_place="p_end",
            attribute_catalog={
                "priority": AttributeCatalogEntry(
                    attribute_type="categorical", possible_values={"high", "low"},
                ),
            },
        )

        builder = DomainBuilder()
        prepared = builder.build_prepared_input(parse_result)

        assert set(prepared.transitions.keys()) == {"activity_a", "activity_b"}

        activity_a = prepared.transitions["activity_a"]
        assert activity_a.activity_name == "activity_a"
        assert activity_a.input_places == ["p_start"]
        assert activity_a.preconditions == []
        assert activity_a.duration.effective_min == 1.0
        assert len(activity_a.effect_groups) == 1
        assert activity_a.effect_groups[0].assignments == [("priority", "high")]

        activity_b = prepared.transitions["activity_b"]
        assert all(g.assignments == [] for g in activity_b.effect_groups)
        assert activity_b.preconditions == []


def _binary_attributes(n: int) -> TransitionInfo:
    """n attributes that each may or may not be set, with a fixed value when
    set: they multiply the groups by 2 each, so the transition ends with 2**n."""
    return _transition_info({
        f"attr_{i}": _effect(
            f"attr_{i}", appearance_level=2, presence_probability=0.5,
            value_level=1, value_probabilities={"x": 1.0},
        )
        for i in range(n)
    })


class TestEffectGroupBudget:
    def test_limit_exceeded_raises_with_details(self):
        with pytest.raises(EffectGroupBudgetExceeded) as exc:
            _prepared_effect_groups(_binary_attributes(6), max_groups=63)
        assert exc.value.activity_name == "activity_a"
        assert exc.value.limit == 63
        assert exc.value.n_groups > 63

    def test_exactly_at_the_limit_is_allowed(self):
        assert len(_prepared_effect_groups(_binary_attributes(6), max_groups=64)) == 64

    def test_no_limit_gives_the_same_groups_as_a_generous_one(self):
        info = _binary_attributes(5)
        unlimited = _prepared_effect_groups(info)
        generous = _prepared_effect_groups(info, max_groups=10_000)
        assert [(g.assignments, g.probability) for g in unlimited] == \
               [(g.assignments, g.probability) for g in generous]

    def test_abort_happens_before_the_product_is_built(self):
        # 2**30 groups if it were built; the check inside the phase must stop
        # it after about 2 * limit partial groups.
        with pytest.raises(EffectGroupBudgetExceeded) as exc:
            _prepared_effect_groups(_binary_attributes(30), max_groups=100)
        assert exc.value.n_groups <= 2 * 100 + 2

    def test_value_phase_is_checked_too(self):
        info = _transition_info({
            f"attr_{i}": _effect(
                f"attr_{i}", appearance_level=1, value_level=2,
                value_probabilities={"a": 0.5, "b": 0.3, "c": 0.2},
            )
            for i in range(10)
        })
        with pytest.raises(EffectGroupBudgetExceeded):
            _prepared_effect_groups(info, max_groups=50)

    def test_domain_builder_uses_the_config_limit(self):
        from models import AnalysisConfig
        parse_result = ParseResult(
            petri_net_model=_sequence_petri_net_model(),
            place_predecessors={},
            transition_predecessors={},
            transitions={"activity_a": _binary_attributes(6)},
            start_place="p_start",
            end_place="p_end",
            attribute_catalog={},
        )
        with pytest.raises(EffectGroupBudgetExceeded):
            DomainBuilder().build_prepared_input(
                parse_result, config=AnalysisConfig(max_effect_groups_per_transition=10),
            )
        # no limit by default: same input builds fine
        prepared = DomainBuilder().build_prepared_input(parse_result)
        assert len(prepared.transitions["activity_a"].effect_groups) == 64


# ===========================================================================
# Filter 2: unused attributes (AnalysisConfig.unused_attribute_filter)
# ===========================================================================

from models import AnalysisConfig, XorBranchInfo  # noqa: E402


def _stat_attribute(name: str) -> EffectInfo:
    """Written about half the time, always with the value "x": doubles the groups."""
    return _effect(name, appearance_level=2, presence_probability=0.5, value_level=1,
                   value_probabilities={"x": 1.0})


def _catalog(*names: str):
    return {n: AttributeCatalogEntry(attribute_type="categorical", possible_values={"x"}, bin_boundaries=None)
            for n in names}


def _filter_parse_result(guard_reader_effects=None, xor_guard_attribute=None) -> ParseResult:
    """activity_a writes a, b, c (each doubles the groups). activity_b writes r.

    Which attributes are READ depends on the arguments:
    - xor_guard_attribute: activity_b is an XOR branch guarded by that attribute;
    - guard_reader_effects: effect(s) of activity_b whose value guard reads an attribute.
    """
    xor_branches = None
    if xor_guard_attribute:
        xor_branches = [XorBranchInfo(place_name="p_mid", probability=1.0, total_samples=10,
                                      cascade_level=1, guards=[[Guard(xor_guard_attribute, "x")]])]
    return ParseResult(
        petri_net_model=_sequence_petri_net_model(),
        place_predecessors={}, transition_predecessors={},
        transitions={
            "activity_a": _transition_info(
                {"a": _stat_attribute("a"), "b": _stat_attribute("b"), "c": _stat_attribute("c")},
                activity_name="activity_a", input_places=["p_start"],
            ),
            "activity_b": _transition_info(
                guard_reader_effects or {}, activity_name="activity_b", input_places=["p_mid"],
                xor_branches=xor_branches,
            ),
        },
        start_place="p_start", end_place="p_end",
        attribute_catalog=_catalog("a", "b", "c", "r"),
    )


def _reader_of(attribute: str) -> dict:
    """An effect whose value guard reads `attribute` (so it counts as read)."""
    guards = EffectGuards(subtype="value", guards={"x": [[Guard(attribute, "x")]]},
                          total_samples=10, dt_accuracy=0.9)
    return {"r": _effect("r", appearance_level=1, value_level=1, value_guards=guards,
                         value_probabilities={"x": 1.0})}


def _groups(prepared, activity: str):
    return prepared.transitions[activity].effect_groups


class TestUnusedAttributeFilter:
    ON = AnalysisConfig(unused_attribute_filter=True)

    def test_off_by_default_keeps_every_effect(self):
        prepared = DomainBuilder().build_prepared_input(_filter_parse_result(xor_guard_attribute="a"))
        assert len(_groups(prepared, "activity_a")) == 8  # a x b x c

    def test_nothing_is_reported_unused_when_the_filter_is_off(self):
        pr = _filter_parse_result(xor_guard_attribute="a")
        assert DomainBuilder().unused_attributes(pr, AnalysisConfig()) == set()

    def test_written_but_never_read_attributes_are_unused(self):
        pr = _filter_parse_result(xor_guard_attribute="a")
        assert DomainBuilder().unused_attributes(pr, self.ON) == {"b", "c"}

    def test_unused_effects_are_dropped_and_the_groups_that_coincide_merge(self):
        pr = _filter_parse_result(xor_guard_attribute="a")
        prepared = DomainBuilder().build_prepared_input(pr, self.ON)
        assigned = sorted(tuple(g.assignments) for g in _groups(prepared, "activity_a"))
        assert assigned == [(), (("a", "x"),)]  # only `a` is left to branch on

    def test_merged_group_probability_is_the_sum_of_the_groups_it_replaces(self):
        pr = _filter_parse_result(xor_guard_attribute="a")
        before = _groups(DomainBuilder().build_prepared_input(pr), "activity_a")
        after = _groups(DomainBuilder().build_prepared_input(pr, self.ON), "activity_a")
        assert sum(g.probability for g in before) == pytest.approx(1.0)
        for kept in after:
            fused = [g for g in before if (("a", "x") in g.assignments) == (("a", "x") in kept.assignments)]
            assert kept.probability == pytest.approx(sum(g.probability for g in fused))

    def test_attribute_read_by_an_xor_branch_guard_is_kept(self):
        pr = _filter_parse_result(xor_guard_attribute="b")
        prepared = DomainBuilder().build_prepared_input(pr, self.ON)
        assert DomainBuilder().unused_attributes(pr, self.ON) == {"a", "c"}
        assert sorted(tuple(g.assignments) for g in _groups(prepared, "activity_a")) == [(), (("b", "x"),)]

    def test_attribute_read_by_an_effect_guard_is_kept(self):
        # activity_b's effect r reads c; r itself is unused. The reader's guard is
        # still counted (conservative), so c stays.
        pr = _filter_parse_result(guard_reader_effects=_reader_of("c"))
        assert DomainBuilder().unused_attributes(pr, self.ON) == {"a", "b", "r"}

    def test_keep_attributes_protect_from_the_filter(self):
        pr = _filter_parse_result(xor_guard_attribute="a")
        config = AnalysisConfig(unused_attribute_filter=True, keep_attributes=("b",))
        prepared = DomainBuilder().build_prepared_input(pr, config)
        assert DomainBuilder().unused_attributes(pr, config) == {"c"}
        assert len(_groups(prepared, "activity_a")) == 4  # a x b

    def test_catalog_keeps_only_read_or_kept_attributes(self):
        pr = _filter_parse_result(xor_guard_attribute="a")
        assert set(DomainBuilder().build_prepared_input(pr, self.ON).attribute_catalog) == {"a"}
        keep_b = AnalysisConfig(unused_attribute_filter=True, keep_attributes=("b",))
        assert set(DomainBuilder().build_prepared_input(pr, keep_b).attribute_catalog) == {"a", "b"}

    def test_catalog_is_untouched_when_the_filter_is_off(self):
        pr = _filter_parse_result(xor_guard_attribute="a")
        assert set(DomainBuilder().build_prepared_input(pr).attribute_catalog) == {"a", "b", "c", "r"}

    def test_the_parse_result_is_not_modified(self):
        pr = _filter_parse_result(xor_guard_attribute="a")
        DomainBuilder().build_prepared_input(pr, self.ON)
        assert set(pr.transitions["activity_a"].effects) == {"a", "b", "c"}

    def test_a_transition_whose_attributes_are_all_unused_keeps_one_empty_group(self):
        pr = _filter_parse_result()  # nothing is read anywhere
        prepared = DomainBuilder().build_prepared_input(pr, self.ON)
        assert [g.assignments for g in _groups(prepared, "activity_a")] == [[]]

    def test_the_groups_budget_sees_the_filtered_product(self):
        # 8 groups unfiltered would exceed a limit of 4; the filtered build has 2
        pr = _filter_parse_result(xor_guard_attribute="a")
        config = AnalysisConfig(unused_attribute_filter=True, max_effect_groups_per_transition=4)
        assert len(_groups(DomainBuilder().build_prepared_input(pr, config), "activity_a")) == 2
        with pytest.raises(EffectGroupBudgetExceeded):
            DomainBuilder().build_prepared_input(pr, AnalysisConfig(max_effect_groups_per_transition=4))


class TestStartOfTraceAttributes:
    """The model remembers which attributes are set by the first event of a case and never change
    (ParseResult.static_attributes), as many as a guard reads (or keep_attributes protects) and
    its catalog keeps."""

    @staticmethod
    def _read_by_a_and_c() -> ParseResult:
        """a is read by an XOR guard, c by an effect's value guard; b and r are not read by any guard."""
        return _filter_parse_result(guard_reader_effects=_reader_of("c"), xor_guard_attribute="a")

    def test_they_are_copied_to_the_prepared_input_sorted(self):
        parse_result = self._read_by_a_and_c()
        parse_result.static_attributes = {"c", "a"}
        prepared = DomainBuilder().build_prepared_input(parse_result)
        assert prepared.start_of_trace_attributes == ["a", "c"]

    def test_one_no_guard_reads_is_left_out_even_with_the_unused_filter_off(self):
        parse_result = self._read_by_a_and_c()
        parse_result.static_attributes = {"a", "b"}
        prepared = DomainBuilder().build_prepared_input(parse_result)
        assert "b" in prepared.attribute_catalog  # the filter is off: still in the catalog
        assert prepared.start_of_trace_attributes == ["a"]

    def test_keep_attributes_protects_an_unread_one(self):
        parse_result = self._read_by_a_and_c()
        parse_result.static_attributes = {"a", "b"}
        prepared = DomainBuilder().build_prepared_input(parse_result, AnalysisConfig(keep_attributes=("b",)))
        assert prepared.start_of_trace_attributes == ["a", "b"]

    def test_only_those_in_the_final_catalog_are_kept(self):
        parse_result = self._read_by_a_and_c()
        parse_result.static_attributes = {"a", "not_in_the_catalog"}
        prepared = DomainBuilder().build_prepared_input(parse_result)
        assert prepared.start_of_trace_attributes == ["a"]

    def test_none_by_default(self):
        prepared = DomainBuilder().build_prepared_input(self._read_by_a_and_c())
        assert prepared.start_of_trace_attributes == []
