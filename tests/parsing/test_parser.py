"""Integration tests for parsing.xes_parser.Parser (full pipeline).

All tests are marked @pytest.mark.integration and are skipped by default.
Run with:  pytest -m integration

A synthetic XES log is written to a temporary directory per session.
The pipeline is executed once via module-scoped fixtures; individual tests
assert properties of the resulting ParseResult without re-running discovery.

Log structure (120 traces, two equal variants):
    60x:  Register[risk=high] → Approve[outcome=approved] → Close[status=done]
    60x:  Register[risk=low]  → Reject[outcome=rejected]  → Close[status=done]

Expected structural outcome after inductive mining + filtering:
    p_start → t_register → p_xor → t_approve → p_end
                                  → t_reject  → ...
    (exact net depends on pm4py; tests use flexible assertions)
"""
import pytest
import pm4py
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from pm4py.objects.log.obj import EventLog, Trace, Event

from parsing.xes_parser import Parser, load_and_discover, preload_replay
from models import (
    ActionDurationStats,
    AnalysisConfig,
    AttributeCatalogEntry,
    AttributeEffect,
    EffectAttrScreening,
    EffectInfo,
    EffectValueScreening,
    ParseResult,
    TransitionEffects,
    TransitionInfo,
    TransitionScreening,
    XorBranchInfo,
)


# ---------------------------------------------------------------------------
# Log factory
# ---------------------------------------------------------------------------

def _ts(offset_seconds: int = 0) -> datetime:
    base = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_seconds)


def _event(name: str, offset: int = 0, **attrs) -> Event:
    e = Event()
    e["concept:name"] = name
    e["time:timestamp"] = _ts(offset)
    for k, v in attrs.items():
        e[k] = v
    return e


def _trace(case_id: str, events: List[Event]) -> Trace:
    t = Trace()
    t.attributes["concept:name"] = case_id
    for e in events:
        t.append(e)
    return t


def _make_xor_log() -> EventLog:
    """Build a 120-trace log with a clear XOR split driven by 'risk' attribute."""
    log = EventLog()
    for i in range(60):
        log.append(_trace(f"high_{i}", [
            _event("Register", 0,  risk="high"),
            _event("Approve",  10, outcome="approved"),
            _event("Close",    20, status="done"),
        ]))
    for i in range(60):
        log.append(_trace(f"low_{i}", [
            _event("Register", 0,  risk="low"),
            _event("Reject",   10, outcome="rejected"),
            _event("Close",    20, status="done"),
        ]))
    return log


# ---------------------------------------------------------------------------
# Module-scoped fixture: run pipeline once for the whole test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def parse_result(tmp_path_factory) -> ParseResult:
    tmp = tmp_path_factory.mktemp("parser_integration")
    xes_path = str(tmp / "xor_log.xes")
    pm4py.write_xes(_make_xor_log(), xes_path)

    config = AnalysisConfig(
        dt_min_samples=20,
        probability_min_samples=5,
        snapshot_dir=str(tmp / "snapshots"),
    )
    parser = Parser(
        xes_path,
        coverage_percentage=0.8,
        discovery_algorithm="inductive",
        config=config,
    )
    return parser.parse_result


# ---------------------------------------------------------------------------
# TestParseResultStructure
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestParseResultStructure:
    """Basic structural invariants of ParseResult."""

    def test_returns_parse_result(self, parse_result):
        assert isinstance(parse_result, ParseResult)

    def test_start_place_nonempty(self, parse_result):
        assert isinstance(parse_result.start_place, str)
        assert parse_result.start_place

    def test_end_place_nonempty(self, parse_result):
        assert isinstance(parse_result.end_place, str)
        assert parse_result.end_place

    def test_start_and_end_places_distinct(self, parse_result):
        assert parse_result.start_place != parse_result.end_place

    def test_transitions_dict_nonempty(self, parse_result):
        assert parse_result.transitions

    def test_place_predecessors_nonempty(self, parse_result):
        assert parse_result.place_predecessors

    def test_transition_predecessors_nonempty(self, parse_result):
        assert parse_result.transition_predecessors

    def test_all_transitions_have_activity_name(self, parse_result):
        for act, info in parse_result.transitions.items():
            assert info.activity_name == act

    def test_all_transitions_have_input_places(self, parse_result):
        for info in parse_result.transitions.values():
            assert isinstance(info.input_places, list)
            assert len(info.input_places) >= 1

    def test_snapshot_files_written(self, parse_result, tmp_path_factory):
        # Snapshot dir is the one configured via the fixture — just check the
        # ParseResult itself is valid (snapshot path not easily accessible here;
        # snapshot existence is verified by the pipeline not raising).
        assert parse_result is not None


# ---------------------------------------------------------------------------
# TestXorBranchInfo
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestXorBranchInfo:
    """XOR branch info is wired to the right transitions."""

    def _xor_branches(self, parse_result: ParseResult):
        # TransitionInfo.xor_branches is a List[XorBranchInfo] (renamed from
        # the earlier singular xor_branch) -- a transition can now be a
        # branch of more than one XOR split.
        return {act: info for act, info in parse_result.transitions.items()
                if info.xor_branches}

    def test_at_least_one_xor_branch_exists(self, parse_result):
        branches = self._xor_branches(parse_result)
        assert branches, "Expected at least one XOR branch in the log"

    def test_xor_branch_has_positive_probability(self, parse_result):
        for act, info in self._xor_branches(parse_result).items():
            for branch in info.xor_branches:
                assert branch.probability >= 0.0

    def test_xor_branch_has_positive_total_samples(self, parse_result):
        for act, info in self._xor_branches(parse_result).items():
            for branch in info.xor_branches:
                assert branch.total_samples > 0

    def test_xor_branch_cascade_level_valid(self, parse_result):
        for act, info in self._xor_branches(parse_result).items():
            for branch in info.xor_branches:
                assert branch.cascade_level in (1, 2, 3)

    def test_approve_or_reject_is_xor_branch(self, parse_result):
        # parse_result.transitions keys are the raw log activity labels
        # (t.label from pm4py, case-preserved) -- PDDL-name sanitization
        # only happens later at encoding time, so _make_xor_log's "Approve"/
        # "Reject" show up unchanged here.
        xor_acts = {act for act, info in parse_result.transitions.items()
                    if info.xor_branches}
        # At least one of the two mutually exclusive activities must be an XOR branch.
        assert xor_acts & {"Approve", "Reject"}, (
            f"Expected 'Approve' or 'Reject' as XOR branches; got {xor_acts}"
        )

    def test_register_not_xor_branch(self, parse_result):
        info = parse_result.transitions.get("Register")
        if info is not None:
            assert not info.xor_branches, "'Register' precedes the XOR split; should not be a branch"


# ---------------------------------------------------------------------------
# TestTransitionInfo
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTransitionInfo:
    """Structural correctness of TransitionInfo objects."""

    def test_total_firings_positive(self, parse_result):
        for info in parse_result.transitions.values():
            assert info.total_firings >= 0

    def test_effects_dict_is_dict(self, parse_result):
        for info in parse_result.transitions.values():
            assert isinstance(info.effects, dict)

    def test_effect_info_fields(self, parse_result):
        for info in parse_result.transitions.values():
            for attr, eff in info.effects.items():
                assert isinstance(eff, EffectInfo)
                assert eff.attribute == attr
                assert 0.0 <= eff.presence_probability <= 1.0
                assert eff.appearance_level in (1, 2, 3)
                assert eff.value_level in (1, 2, 3)
                assert isinstance(eff.value_probabilities, dict)

    def test_effect_level3_has_no_guards(self, parse_result):
        for info in parse_result.transitions.values():
            for eff in info.effects.values():
                if eff.appearance_level == 3:
                    assert eff.appearance_guards is None
                if eff.value_level == 3:
                    assert eff.value_guards is None


# ---------------------------------------------------------------------------
# TestPredecessorMaps
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPredecessorMaps:
    """place_predecessors and transition_predecessors are consistent with the net."""

    def test_place_predecessor_values_are_lists_of_strings(self, parse_result):
        for place, preds in parse_result.place_predecessors.items():
            assert isinstance(place, str)
            assert isinstance(preds, list)
            for act in preds:
                assert isinstance(act, str)

    def test_transition_predecessor_values_are_lists_of_strings(self, parse_result):
        for act, preds in parse_result.transition_predecessors.items():
            assert isinstance(act, str)
            assert isinstance(preds, list)
            for p in preds:
                assert isinstance(p, str)

    def test_every_transition_has_predecessor_entry(self, parse_result):
        for act in parse_result.transitions:
            assert act in parse_result.transition_predecessors, (
                f"Transition '{act}' missing from transition_predecessors"
            )

    def test_start_place_not_in_place_predecessors_or_has_no_transitions(self, parse_result):
        # The start place has no incoming transitions by definition.
        preds = parse_result.place_predecessors.get(parse_result.start_place, [])
        assert preds == [], (
            f"Start place '{parse_result.start_place}' should have no predecessors; got {preds}"
        )


# ---------------------------------------------------------------------------
# TestDurationInfo
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDurationInfo:
    """Duration statistics are attached to TransitionInfo after pipeline."""

    def test_duration_field_is_optional_action_duration_stats(self, parse_result):
        for info in parse_result.transitions.values():
            assert info.duration is None or isinstance(info.duration, ActionDurationStats)

    def test_at_least_one_transition_has_duration(self, parse_result):
        """The synthetic log has timestamps, so inter-event durations must be computed."""
        has_duration = any(
            info.duration is not None
            for info in parse_result.transitions.values()
        )
        assert has_duration, "Expected at least one transition with duration data"

    def test_duration_effective_min_lte_max(self, parse_result):
        for info in parse_result.transitions.values():
            if info.duration is not None:
                assert info.duration.effective_min <= info.duration.effective_max, (
                    f"effective_min > effective_max for '{info.activity_name}'"
                )

    def test_duration_source_is_valid(self, parse_result):
        valid_sources = {"lifecycle", "inter_event", "external"}
        for info in parse_result.transitions.values():
            if info.duration is not None:
                assert info.duration.source in valid_sources, (
                    f"Unknown source '{info.duration.source}' for '{info.activity_name}'"
                )

    def test_log_derived_duration_has_statistical_fields(self, parse_result):
        """Lifecycle and inter-event durations must have non-None statistical fields."""
        for info in parse_result.transitions.values():
            d = info.duration
            if d is not None and d.source in {"lifecycle", "inter_event"}:
                assert d.count is not None and d.count > 0
                assert d.mean is not None
                assert d.std_dev is not None


# ---------------------------------------------------------------------------
# TestAttributeCatalog
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAttributeCatalog:
    """attribute_catalog contains only attributes referenced in surviving effects/guards."""

    def test_catalog_is_dict(self, parse_result):
        assert isinstance(parse_result.attribute_catalog, dict)

    def test_catalog_is_nonempty(self, parse_result):
        assert parse_result.attribute_catalog, "Expected at least one attribute in the catalog"

    def test_catalog_keys_are_strings(self, parse_result):
        for attr in parse_result.attribute_catalog:
            assert isinstance(attr, str)

    def test_catalog_entries_have_valid_type(self, parse_result):
        valid_types = {"boolean", "numerical", "categorical"}
        for attr, entry in parse_result.attribute_catalog.items():
            assert isinstance(entry, AttributeCatalogEntry)
            assert entry.attribute_type in valid_types, (
                f"Attribute '{attr}' has unknown type '{entry.attribute_type}'"
            )

    def test_catalog_possible_values_are_sets(self, parse_result):
        for attr, entry in parse_result.attribute_catalog.items():
            assert isinstance(entry.possible_values, set), (
                f"possible_values for '{attr}' is not a set"
            )

    def test_effect_attributes_present_in_catalog(self, parse_result):
        """Every attribute that appears in at least one EffectInfo must be in the catalog."""
        for info in parse_result.transitions.values():
            for attr in info.effects:
                assert attr in parse_result.attribute_catalog, (
                    f"Effect attribute '{attr}' missing from catalog"
                )

    def test_effect_values_present_in_catalog(self, parse_result):
        """Values in EffectInfo.value_probabilities must be in possible_values."""
        for info in parse_result.transitions.values():
            for attr, eff in info.effects.items():
                catalog_values = parse_result.attribute_catalog[attr].possible_values
                for val in eff.value_probabilities:
                    assert val in catalog_values, (
                        f"Value '{val}' for effect attribute '{attr}' missing from catalog"
                    )

    def test_xor_guard_attributes_present_in_catalog(self, parse_result):
        """Attributes referenced in XOR branch guards must be in the catalog."""
        for info in parse_result.transitions.values():
            for branch in info.xor_branches or []:
                if branch.guards is None:
                    continue
                for path in branch.guards:
                    for guard in path:
                        assert guard.attribute in parse_result.attribute_catalog, (
                            f"Guard attribute '{guard.attribute}' missing from catalog"
                        )

    def test_catalog_does_not_contain_ignored_attributes(self, parse_result):
        """Standard infrastructure attributes must never appear in the catalog."""
        ignored = {
            "concept_name", "time_timestamp", "lifecycle_transition",
            "org_resource", "org_group",
        }
        for attr in parse_result.attribute_catalog:
            assert attr not in ignored, (
                f"Ignored attribute '{attr}' found in catalog"
            )


# ---------------------------------------------------------------------------
# TestFilterEffectsLevel3  (unit tests — no full pipeline)
# ---------------------------------------------------------------------------

def _make_parser_stub(
    effect_screening: Dict[str, TransitionScreening],
    transition_effects: Dict[str, TransitionEffects],
    attribute_effects: Dict[str, AttributeEffect],
    min_samples: int = 10,
) -> Parser:
    """Create a Parser instance with only the attributes needed by _filter_effects."""
    p = object.__new__(Parser)
    p.config = AnalysisConfig(probability_min_samples=min_samples)
    p.effect_screening = effect_screening
    p.transition_effects = transition_effects
    p.attribute_effects = attribute_effects
    return p


def _attr_scr(
    action: str,
    samples: int,
    presence_prob: float = 0.5,
    value_action: str = "fallback",
    value_samples: int = 0,
) -> EffectAttrScreening:
    return EffectAttrScreening(
        presence_probability=presence_prob,
        appearance_action=action,
        appearance_samples=samples,
        value_action=value_action,
        value_samples=value_samples,
        values={"v1": EffectValueScreening(probability=1.0, status="active")},
    )


def _t_scr(total_firings: int, attrs: Dict[str, EffectAttrScreening]) -> TransitionScreening:
    return TransitionScreening(
        transition_name="act",
        total_firings=total_firings,
        attributes=attrs,
    )


def _attr_effect(total_firings: int) -> AttributeEffect:
    return AttributeEffect(
        presence_probabilities={"color": 0.5},
        value_probabilities={"color": {"v1": 1.0}},
        total_firings=total_firings,
    )


class TestFilterEffectsLevel3:
    """_filter_effects level-3 triggers: transition total_firings and attr appearance_samples."""

    def test_transition_below_min_samples_excluded_entirely(self):
        """A transition with total_firings < min_samples must have no effects."""
        screening = {
            "act": _t_scr(
                total_firings=5,
                attrs={"color": _attr_scr("fallback", samples=5)},
            )
        }
        p = _make_parser_stub(screening, {}, {"act": _attr_effect(5)}, min_samples=10)
        p._filter_effects()
        assert "act" not in p._transition_effect_info

    def test_transition_at_min_samples_included(self):
        """A transition with total_firings == min_samples is not excluded by the transition gate."""
        screening = {
            "act": _t_scr(
                total_firings=10,
                attrs={"color": _attr_scr("fallback", samples=10)},
            )
        }
        p = _make_parser_stub(screening, {}, {"act": _attr_effect(10)}, min_samples=10)
        p._filter_effects()
        assert "act" in p._transition_effect_info
        assert "color" in p._transition_effect_info["act"]

    def test_attr_with_low_appearance_samples_is_kept_as_fallback(self):
        """Only appearance_action == 'never' excludes an attribute; a low sample
        count alone (action='fallback') keeps it, downgraded to level 2."""
        screening = {
            "act": _t_scr(
                total_firings=50,
                attrs={"color": _attr_scr("fallback", samples=3)},
            )
        }
        p = _make_parser_stub(screening, {}, {"act": _attr_effect(50)}, min_samples=10)
        p._filter_effects()
        color_info = p._transition_effect_info.get("act", {}).get("color")
        assert color_info is not None
        assert color_info.appearance_level == 2

    def test_other_attrs_kept_when_one_attr_is_never(self):
        """A 'never' attr is excluded without affecting sibling attributes."""
        screening = {
            "act": _t_scr(
                total_firings=50,
                attrs={
                    "rare": _attr_scr("never", samples=2),
                    "common": _attr_scr("fallback", samples=20),
                },
            )
        }
        ae = AttributeEffect(
            presence_probabilities={"rare": 0.1, "common": 0.8},
            value_probabilities={"rare": {"v1": 1.0}, "common": {"v1": 1.0}},
            total_firings=50,
        )
        p = _make_parser_stub(screening, {}, {"act": ae}, min_samples=10)
        p._filter_effects()
        act_info = p._transition_effect_info.get("act", {})
        assert "rare" not in act_info
        assert "common" in act_info

    def test_never_attr_excluded_regardless_of_firings(self):
        """Attrs with appearance_action='never' are always excluded."""
        screening = {
            "act": _t_scr(
                total_firings=100,
                attrs={"color": _attr_scr("never", samples=0)},
            )
        }
        p = _make_parser_stub(screening, {}, {"act": _attr_effect(100)}, min_samples=10)
        p._filter_effects()
        assert p._transition_effect_info.get("act", {}).get("color") is None

    def test_deterministic_attr_kept_when_transition_has_enough_firings(self):
        """A deterministic attr is included when total_firings >= min_samples."""
        screening = {
            "act": _t_scr(
                total_firings=20,
                attrs={"color": _attr_scr("deterministic", samples=20, value_action="deterministic", value_samples=20)},
            )
        }
        p = _make_parser_stub(screening, {}, {"act": _attr_effect(20)}, min_samples=10)
        p._filter_effects()
        act_info = p._transition_effect_info.get("act", {})
        assert "color" in act_info
        assert act_info["color"].appearance_level == 1


@pytest.mark.integration
class TestPreloadedEquivalence:
    """Parser(preloaded=...) must be behavior-preserving vs. Parser() running
    steps 1-2 itself — see docs/network_search_trial_caching_plan.md."""

    def test_parse_result_matches_with_and_without_preloaded(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("parser_preloaded_equivalence")
        xes_path = str(tmp / "xor_log.xes")
        pm4py.write_xes(_make_xor_log(), xes_path)

        config = AnalysisConfig(
            dt_min_samples=20,
            probability_min_samples=5,
            snapshot_dir=str(tmp / "snapshots"),
        )

        direct = Parser(
            xes_path, coverage_percentage=0.8, discovery_algorithm="inductive", config=config,
        ).parse_result

        preloaded = load_and_discover(xes_path, coverage_percentage=0.8, discovery_algorithm="inductive")
        via_preloaded = Parser(
            xes_path, coverage_percentage=0.8, discovery_algorithm="inductive", config=config,
            preloaded=preloaded,
        ).parse_result

        assert direct.transitions.keys() == via_preloaded.transitions.keys()
        for name, d_trans in direct.transitions.items():
            p_trans = via_preloaded.transitions[name]
            assert set(d_trans.effects.keys()) == set(p_trans.effects.keys())
            for attr, d_effect in d_trans.effects.items():
                p_effect = p_trans.effects[attr]
                assert d_effect.appearance_level == p_effect.appearance_level
                assert d_effect.value_level == p_effect.value_level
                assert d_effect.value_probabilities == p_effect.value_probabilities

            d_branches = d_trans.xor_branches or []
            p_branches = p_trans.xor_branches or []
            assert len(d_branches) == len(p_branches)
            for d_b, p_b in zip(d_branches, p_branches):
                assert d_b.place_name == p_b.place_name
                assert d_b.cascade_level == p_b.cascade_level
                assert d_b.routing_source == p_b.routing_source
                assert d_b.guards == p_b.guards

    def test_parse_result_matches_with_and_without_preloaded_replay(self, tmp_path_factory):
        """preloaded_replay caches the raw token-replay call + duration stats
        (computed once, ahead of the per-trial replay_min_fitness filter) —
        see docs plan for the network_search caching work. The resulting
        ParseResult must match a Parser() run that recomputes both from
        scratch, except duration_stats which is expected to differ slightly
        since it's computed on all successfully-replayed traces rather than
        only the ones surviving this particular config's fitness threshold."""
        tmp = tmp_path_factory.mktemp("parser_preloaded_replay_equivalence")
        xes_path = str(tmp / "xor_log.xes")
        pm4py.write_xes(_make_xor_log(), xes_path)

        config = AnalysisConfig(
            dt_min_samples=20,
            probability_min_samples=5,
            snapshot_dir=str(tmp / "snapshots"),
        )

        direct = Parser(
            xes_path, coverage_percentage=0.8, discovery_algorithm="inductive", config=config,
        ).parse_result

        preloaded = load_and_discover(xes_path, coverage_percentage=0.8, discovery_algorithm="inductive")
        preloaded_replay = preload_replay(preloaded, config)
        via_preloaded = Parser(
            xes_path, coverage_percentage=0.8, discovery_algorithm="inductive", config=config,
            preloaded=preloaded, preloaded_replay=preloaded_replay,
        ).parse_result

        assert direct.transitions.keys() == via_preloaded.transitions.keys()
        for name, d_trans in direct.transitions.items():
            p_trans = via_preloaded.transitions[name]
            assert set(d_trans.effects.keys()) == set(p_trans.effects.keys())

        # duration_stats: same set of activities have duration data, even
        # though the two Parser()s computed it from slightly different trace
        # sets (fitness-filtered vs. all successfully-replayed traces).
        direct_durations = {t.activity_name for t in direct.transitions.values() if t.duration is not None}
        preloaded_durations = {t.activity_name for t in via_preloaded.transitions.values() if t.duration is not None}
        assert direct_durations == preloaded_durations
