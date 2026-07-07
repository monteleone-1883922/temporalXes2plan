"""Tests for encoding.domain_builder.DomainBuilder._build_prepared_domain_input
and the effect-group derivation helper (_prepared_effect_groups).

Builds TransitionInfo/EffectInfo/XorBranchInfo/ParseResult by hand, using the
CURRENT dataclass signatures directly — deliberately not reusing the shared
fixtures in tests/encoding/conftest.py, which are stale WIP (built against an
older TransitionInfo/XorBranchInfo shape) and fail to construct today.
"""
import logging

from pm4py.objects.petri_net.obj import Marking, PetriNet

from encoding.domain_builder import DomainBuilder, _prepared_effect_groups
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
        prepared = builder._build_prepared_domain_input(parse_result)

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
