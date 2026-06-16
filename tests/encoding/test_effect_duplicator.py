"""Tests for encoding.effect_duplicator.EffectDuplicator."""
import math

import pytest

from models import (
    AnalysisConfig, AttributeCatalogEntry, EffectGuards, EffectInfo,
    Guard, ParseResult, TransitionInfo,
)
from encoding.action_builder import ActionBuilder
from encoding.action_registry import ActionRegistry
from encoding.effect_duplicator import (
    EffectDuplicator, _conditions_conflict, _extract_effect_attributes,
)
from encoding.pddl_model import PDDLAction, PDDLCondition, PDDLDurativeAction, PDDLEffect


def _make_registry(parse_result, use_durative=False):
    actions = ActionBuilder(parse_result, use_durative=use_durative).build_all()
    return ActionRegistry.from_base_actions(actions)


def _make_pr(sequence_net, transitions, catalog, negated=None):
    """Build a ParseResult for the sequence net with custom transitions."""
    return ParseResult(
        petri_net_model=sequence_net,
        place_predecessors={"p_mid": ["activity_a"], "p_end": ["activity_b"]},
        transition_predecessors={"activity_a": ["p_start"], "activity_b": ["p_mid"]},
        transitions=transitions,
        start_place="p_start",
        end_place="p_end",
        attribute_catalog=catalog,
        negated_attributes=negated or set(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return AnalysisConfig()


@pytest.fixture
def cfg_no_cost():
    return AnalysisConfig(
        effect_appearance_mode="duplicate_no_cost",
        effect_value_mode="duplicate_no_cost",
    )


@pytest.fixture
def catalog_diag():
    return {
        "diagnosis": AttributeCatalogEntry("categorical", {"flu", "cold"}),
    }


@pytest.fixture
def catalog_diag_outcome():
    return {
        "diagnosis": AttributeCatalogEntry("categorical", {"flu", "cold"}),
        "outcome": AttributeCatalogEntry("categorical", {"recovered", "deceased"}),
    }


@pytest.fixture
def catalog_bool():
    return {"urgent": AttributeCatalogEntry("boolean", set())}


# ---------------------------------------------------------------------------
# Appearance: deterministic
# ---------------------------------------------------------------------------

class TestAppearanceDeterministic:

    def test_effect_added_to_all_variants(self, sequence_net, cfg, catalog_diag):
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"flu": 0.6, "cold": 0.4},
            value_level=2, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        assert len(variants) == 2
        for v in variants:
            assert any(
                e.attribute == "diagnosis" and not e.clear for e in v.effects
            )


# ---------------------------------------------------------------------------
# Appearance: DT guards
# ---------------------------------------------------------------------------

class TestAppearanceDT:

    def _make_effect_with_appearance_guards(self):
        return EffectInfo(
            attribute="diagnosis", presence_probability=0.8,
            appearance_level=1,
            appearance_guards=EffectGuards(
                subtype="appearance",
                guards={
                    "appears": [
                        [Guard("urgent", None, negated=False)],
                    ],
                },
                total_samples=100, dt_accuracy=0.9,
            ),
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )

    def test_creates_copy_with_preconditions(self, sequence_net, cfg, catalog_diag):
        effect = self._make_effect_with_appearance_guards()
        catalog = {
            **catalog_diag,
            "urgent": AttributeCatalogEntry("boolean", set()),
        }
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_effect = [
            v for v in variants
            if any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]
        without_effect = [
            v for v in variants
            if not any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]

        assert len(with_effect) >= 1
        assert len(without_effect) >= 1
        for v in with_effect:
            assert PDDLCondition.attr_true("urgent") in v.preconditions

    def test_or_clauses_create_multiple_copies(self, sequence_net, cfg, catalog_diag):
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.8,
            appearance_level=1,
            appearance_guards=EffectGuards(
                subtype="appearance",
                guards={
                    "appears": [
                        [Guard("urgent", None, negated=False)],
                        [Guard("risk", "high", negated=False)],
                    ],
                },
                total_samples=100, dt_accuracy=0.9,
            ),
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        catalog = {
            **catalog_diag,
            "urgent": AttributeCatalogEntry("boolean", set()),
            "risk": AttributeCatalogEntry("categorical", {"high", "low"}),
        }
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_effect = [
            v for v in variants
            if any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]
        assert len(with_effect) == 2


# ---------------------------------------------------------------------------
# Appearance: probabilistic
# ---------------------------------------------------------------------------

class TestAppearanceProbabilistic:

    def _make_prob_effect(self):
        return EffectInfo(
            attribute="diagnosis", presence_probability=0.7,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )

    def test_creates_copy_and_original(self, sequence_net, cfg, catalog_diag):
        effect = self._make_prob_effect()
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_eff = [
            v for v in variants
            if any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]
        without_eff = [
            v for v in variants
            if not any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]
        assert len(with_eff) >= 1
        assert len(without_eff) >= 1

    def test_probability_tracked_on_copy(self, sequence_net, cfg, catalog_diag):
        effect = self._make_prob_effect()
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_eff = [
            v for v in variants
            if any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]
        without_eff = [
            v for v in variants
            if not any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]

        assert with_eff[0].effect_probability == pytest.approx(0.7)
        assert without_eff[0].effect_probability == pytest.approx(0.3)

    def test_no_cost_mode_skips_probability(self, sequence_net, cfg_no_cost, catalog_diag):
        effect = self._make_prob_effect()
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg_no_cost).apply(registry)

        variants = registry.get("execute_activity_a")
        for v in variants:
            assert v.effect_probability == 1.0


# ---------------------------------------------------------------------------
# Value: deterministic
# ---------------------------------------------------------------------------

class TestValueDeterministic:

    def test_single_value_inserted(self, sequence_net, cfg, catalog_diag):
        effect_with_guards = EffectInfo(
            attribute="diagnosis", presence_probability=0.8,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect_with_guards},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_eff = [v for v in variants if PDDLEffect.set_attr_is("diagnosis", "flu") in v.effects]
        assert len(with_eff) >= 1


# ---------------------------------------------------------------------------
# Value: DT guards
# ---------------------------------------------------------------------------

class TestValueDT:

    def test_creates_variants_per_value(self, sequence_net, cfg, catalog_diag):
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"flu": 0.6, "cold": 0.4},
            value_level=1,
            value_guards=EffectGuards(
                subtype="value",
                guards={
                    "flu": [[Guard("urgent", None, negated=False)]],
                    "cold": [[Guard("urgent", None, negated=True)]],
                },
                total_samples=100, dt_accuracy=0.85,
            ),
        )
        catalog = {
            **catalog_diag,
            "urgent": AttributeCatalogEntry("boolean", set()),
        }
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        flu_variants = [v for v in variants if PDDLEffect.set_attr_is("diagnosis", "flu") in v.effects]
        cold_variants = [v for v in variants if PDDLEffect.set_attr_is("diagnosis", "cold") in v.effects]

        assert len(flu_variants) >= 1
        assert len(cold_variants) >= 1

        for v in flu_variants:
            assert PDDLCondition.attr_true("urgent") in v.preconditions
        for v in cold_variants:
            assert PDDLCondition.attr_false("urgent") in v.preconditions


# ---------------------------------------------------------------------------
# Value: probabilistic
# ---------------------------------------------------------------------------

class TestValueProbabilistic:

    def test_creates_variants_with_probability(self, sequence_net, cfg, catalog_diag):
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"flu": 0.6, "cold": 0.4},
            value_level=2, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        flu_variants = [v for v in variants if PDDLEffect.set_attr_is("diagnosis", "flu") in v.effects]
        cold_variants = [v for v in variants if PDDLEffect.set_attr_is("diagnosis", "cold") in v.effects]

        assert len(flu_variants) >= 1
        assert len(cold_variants) >= 1
        assert flu_variants[0].effect_probability == pytest.approx(0.6)
        assert cold_variants[0].effect_probability == pytest.approx(0.4)

    def test_no_cost_mode(self, sequence_net, cfg_no_cost, catalog_diag):
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"flu": 0.6, "cold": 0.4},
            value_level=2, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg_no_cost).apply(registry)

        variants = registry.get("execute_activity_a")
        for v in variants:
            assert v.effect_probability == 1.0


# ---------------------------------------------------------------------------
# Cartesian product: appearance DT × value DT
# ---------------------------------------------------------------------------

class TestCartesianProduct:

    def test_appearance_x_value(self, sequence_net, cfg):
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.8,
            appearance_level=1,
            appearance_guards=EffectGuards(
                subtype="appearance",
                guards={
                    "appears": [
                        [Guard("admitted", None, negated=False)],
                        [Guard("risk", "high", negated=False)],
                    ],
                },
                total_samples=100, dt_accuracy=0.9,
            ),
            value_probabilities={"flu": 0.6, "cold": 0.4},
            value_level=1,
            value_guards=EffectGuards(
                subtype="value",
                guards={
                    "flu": [[Guard("urgent", None, negated=False)]],
                    "cold": [[Guard("urgent", None, negated=True)]],
                },
                total_samples=100, dt_accuracy=0.85,
            ),
        )
        catalog = {
            "diagnosis": AttributeCatalogEntry("categorical", {"flu", "cold"}),
            "admitted": AttributeCatalogEntry("boolean", set()),
            "risk": AttributeCatalogEntry("categorical", {"high", "low"}),
            "urgent": AttributeCatalogEntry("boolean", set()),
        }
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_eff = [
            v for v in variants
            if any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]
        # 2 appearance clauses × 2 value outcomes = 4 variants with effect
        assert len(with_eff) == 4


# ---------------------------------------------------------------------------
# Incompatible effects
# ---------------------------------------------------------------------------

class TestIncompatibleEffects:

    def test_skips_variants_with_incompatible_attrs(self, sequence_net, cfg, catalog_diag_outcome):
        effect_diag = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        effect_outcome = EffectInfo(
            attribute="outcome", presence_probability=0.7,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"recovered": 1.0},
            value_level=1, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect_diag, "outcome": effect_outcome},
                incompatible_effects={frozenset({"diagnosis", "outcome"})},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag_outcome)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        for v in variants:
            has_diag = any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
            has_outcome = any(e.attribute == "outcome" and not e.clear for e in v.effects)
            assert not (has_diag and has_outcome), \
                "Incompatible effects should not appear on the same action"


# ---------------------------------------------------------------------------
# Related effects
# ---------------------------------------------------------------------------

class TestRelatedEffects:

    def test_modifies_in_place_for_related(self, sequence_net, cfg, catalog_diag_outcome):
        effect_diag = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        effect_outcome = EffectInfo(
            attribute="outcome", presence_probability=0.8,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"recovered": 1.0},
            value_level=1, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect_diag, "outcome": effect_outcome},
                related_effects={frozenset({"diagnosis", "outcome"})},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag_outcome)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        both = [
            v for v in variants
            if any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
            and any(e.attribute == "outcome" and not e.clear for e in v.effects)
        ]
        assert len(both) >= 1


# ---------------------------------------------------------------------------
# Merge compatible variants
# ---------------------------------------------------------------------------

class TestMergeVariants:

    def test_merges_same_preconditions(self, sequence_net, cfg, catalog_diag_outcome):
        effect_diag = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        effect_outcome = EffectInfo(
            attribute="outcome", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"recovered": 1.0},
            value_level=1, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect_diag, "outcome": effect_outcome},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag_outcome)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        # Both effects are fully deterministic — handled by ActionBuilder.
        # EffectDuplicator should not touch them.
        variants = registry.get("execute_activity_a")
        assert len(variants) == 1

    def test_merge_probability_capped_at_1(self, sequence_net, cfg, catalog_diag_outcome):
        effect_diag = EffectInfo(
            attribute="diagnosis", presence_probability=0.7,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        effect_outcome = EffectInfo(
            attribute="outcome", presence_probability=0.8,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"recovered": 1.0},
            value_level=1, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect_diag, "outcome": effect_outcome},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag_outcome)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        for v in variants:
            assert v.effect_probability <= 1.0


# ---------------------------------------------------------------------------
# Cost finalisation
# ---------------------------------------------------------------------------

class TestCostFinalisation:

    def test_additional_cost_set_from_probability(self, sequence_net, cfg, catalog_diag):
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.7,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        for v in variants:
            if v.effect_probability < 1.0:
                assert v.additional_cost is not None
                assert v.additional_cost == pytest.approx(-math.log(v.effect_probability))

    def test_cost_added_to_existing_xor_cost(self, sequence_net, cfg, catalog_diag):
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.7,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)

        # Simulate XOR cost already applied
        import dataclasses
        xor_cost = 0.5
        old = registry.get("execute_activity_a")
        registry.replace(
            "execute_activity_a",
            [dataclasses.replace(v, additional_cost=xor_cost) for v in old],
        )

        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        for v in variants:
            if v.effect_probability < 1.0:
                expected = xor_cost + (-math.log(v.effect_probability))
                assert v.additional_cost == pytest.approx(expected)

    def test_no_cost_when_probability_is_1(self, sequence_net, cfg, catalog_diag):
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"flu": 0.6, "cold": 0.4},
            value_level=2, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        for v in variants:
            if v.effect_probability < 1.0:
                assert v.additional_cost is not None


# ---------------------------------------------------------------------------
# Durative actions
# ---------------------------------------------------------------------------

class TestDurativeActions:

    def test_effects_go_to_at_end(self, sequence_net, cfg, catalog_diag):
        from models import ActionDurationStats
        dur = ActionDurationStats(10.0, 30.0, "lifecycle", 20.0, 5.0, 8.0, 35.0, 100)
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.7,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
                duration=dur,
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr, use_durative=True)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_eff = [
            v for v in variants
            if isinstance(v, PDDLDurativeAction)
            and any(e.attribute == "diagnosis" and not e.clear for e in v.effects_at_end)
        ]
        assert len(with_eff) >= 1

    def test_preconditions_go_to_at_start(self, sequence_net, cfg, catalog_diag):
        from models import ActionDurationStats
        dur = ActionDurationStats(10.0, 30.0, "lifecycle", 20.0, 5.0, 8.0, 35.0, 100)
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.8,
            appearance_level=1,
            appearance_guards=EffectGuards(
                subtype="appearance",
                guards={"appears": [[Guard("urgent", None, negated=False)]]},
                total_samples=100, dt_accuracy=0.9,
            ),
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        catalog = {
            **catalog_diag,
            "urgent": AttributeCatalogEntry("boolean", set()),
        }
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
                duration=dur,
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog)
        registry = _make_registry(pr, use_durative=True)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_guard = [
            v for v in variants
            if isinstance(v, PDDLDurativeAction)
            and PDDLCondition.attr_true("urgent") in v.conditions_at_start
        ]
        assert len(with_guard) >= 1


# ---------------------------------------------------------------------------
# effect_attributes on actions
# ---------------------------------------------------------------------------

class TestEffectAttributes:

    def test_deterministic_effect_sets_attribute(self, sequence_net, catalog_diag):
        """ActionBuilder populates effect_attributes for deterministic effects."""
        det_effect = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": det_effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        actions = ActionBuilder(pr).build_all()
        action_a = next(a for a in actions if a.name == "execute_activity_a")
        assert "diagnosis" in action_a.effect_attributes

    def test_no_effects_empty_set(self, sequence_net, catalog_diag):
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None, effects={},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        actions = ActionBuilder(pr).build_all()
        action_a = next(a for a in actions if a.name == "execute_activity_a")
        assert action_a.effect_attributes == set()

    def test_probabilistic_effect_updates_attribute(self, sequence_net, cfg, catalog_diag):
        """EffectDuplicator sets effect_attributes when adding a conditional effect."""
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.7,
            appearance_level=2, appearance_guards=None,
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None,
                effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo(
                "activity_b", ["p_mid"], 80, None, effects={},
            ),
        }
        pr = _make_pr(sequence_net, transitions, catalog_diag)
        registry = _make_registry(pr)
        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_eff = [
            v for v in variants if PDDLEffect.set_attr_is("diagnosis", "flu") in v.effects
        ]
        without_eff = [
            v for v in variants if PDDLEffect.set_attr_is("diagnosis", "flu") not in v.effects
        ]

        for v in with_eff:
            assert "diagnosis" in v.effect_attributes
        for v in without_eff:
            assert "diagnosis" not in v.effect_attributes


# ---------------------------------------------------------------------------
# Helper: _extract_effect_attributes
# ---------------------------------------------------------------------------

class TestExtractEffectAttributes:

    def test_extracts_categorical(self):
        assert _extract_effect_attributes([PDDLEffect.set_attr_is("diagnosis", "flu")]) == {"diagnosis"}

    def test_extracts_boolean(self):
        assert _extract_effect_attributes([PDDLEffect.set_attr_true("urgent")]) == {"urgent"}

    def test_extracts_negated_clear(self):
        assert _extract_effect_attributes([PDDLEffect.clear_attr_is("diagnosis", "flu")]) == {"diagnosis"}

    def test_ignores_marked(self):
        assert _extract_effect_attributes([PDDLEffect.marking("activity_a")]) == set()

    def test_extracts_is_not(self):
        assert _extract_effect_attributes([PDDLEffect.set_attr_is_not("diagnosis", "flu")]) == {"diagnosis"}

    def test_extracts_false(self):
        assert _extract_effect_attributes([PDDLEffect.set_attr_false("urgent")]) == {"urgent"}


# ---------------------------------------------------------------------------
# PDDLCondition.logical_negation
# ---------------------------------------------------------------------------

class TestLogicalNegation:

    def test_attr_is_to_attr_is_not(self):
        c = PDDLCondition.attr_is("risk", "high")
        assert c.logical_negation() == PDDLCondition.attr_is_not("risk", "high")

    def test_attr_is_not_to_attr_is(self):
        c = PDDLCondition.attr_is_not("risk", "high")
        assert c.logical_negation() == PDDLCondition.attr_is("risk", "high")

    def test_attr_true_to_attr_false(self):
        assert PDDLCondition.attr_true("urgent").logical_negation() == PDDLCondition.attr_false("urgent")

    def test_attr_false_to_attr_true(self):
        assert PDDLCondition.attr_false("urgent").logical_negation() == PDDLCondition.attr_true("urgent")

    def test_marked_raises(self):
        with pytest.raises(ValueError):
            PDDLCondition.marked("p_start").logical_negation()


# ---------------------------------------------------------------------------
# _conditions_conflict
# ---------------------------------------------------------------------------

class TestConditionsConflict:

    def test_is_vs_is_not(self):
        assert _conditions_conflict(
            {PDDLCondition.attr_is_not("risk", "high")},
            [PDDLCondition.attr_is("risk", "high")],
        )

    def test_is_not_vs_is(self):
        assert _conditions_conflict(
            {PDDLCondition.attr_is("risk", "high")},
            [PDDLCondition.attr_is_not("risk", "high")],
        )

    def test_true_vs_false(self):
        assert _conditions_conflict(
            {PDDLCondition.attr_false("urgent")},
            [PDDLCondition.attr_true("urgent")],
        )

    def test_false_vs_true(self):
        assert _conditions_conflict(
            {PDDLCondition.attr_true("urgent")},
            [PDDLCondition.attr_false("urgent")],
        )

    def test_same_attr_different_value(self):
        assert _conditions_conflict(
            {PDDLCondition.attr_is("risk", "high")},
            [PDDLCondition.attr_is("risk", "low")],
        )

    def test_same_attr_same_value_no_conflict(self):
        assert not _conditions_conflict(
            {PDDLCondition.attr_is("risk", "high")},
            [PDDLCondition.attr_is("risk", "high")],
        )

    def test_different_attributes_no_conflict(self):
        assert not _conditions_conflict(
            {PDDLCondition.attr_is("risk", "high")},
            [PDDLCondition.attr_true("urgent")],
        )

    def test_empty_existing_no_conflict(self):
        assert not _conditions_conflict(set(), [PDDLCondition.attr_is("risk", "high")])

    def test_marked_predicate_no_conflict(self):
        assert not _conditions_conflict(
            {PDDLCondition.marked("p_start")},
            [PDDLCondition.marked("p_xor")],
        )


# ---------------------------------------------------------------------------
# Integration: conflict skips clause during duplication
# ---------------------------------------------------------------------------

class TestPreconditionConflict:

    def _inject_precondition(self, registry, action_name: str, cond: PDDLCondition):
        """Add a precondition to all variants of an action (simulates prior XOR guard)."""
        import dataclasses
        old = registry.get(action_name)
        registry.replace(
            action_name,
            [dataclasses.replace(v, preconditions=v.preconditions | {cond}) for v in old],
        )

    def test_appearance_conflicting_clause_skipped(self, sequence_net, cfg):
        """OR-clause whose AND conditions conflict with existing preconditions is skipped."""
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.8,
            appearance_level=1,
            appearance_guards=EffectGuards(
                subtype="appearance",
                guards={
                    "appears": [
                        [Guard("risk", "low", negated=False)],   # conflicts with (risk_is high)
                        [Guard("urgent", None, negated=False)],  # compatible
                    ],
                },
                total_samples=100, dt_accuracy=0.9,
            ),
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        catalog = {
            "diagnosis": AttributeCatalogEntry("categorical", {"flu", "cold"}),
            "risk": AttributeCatalogEntry("categorical", {"high", "low"}),
            "urgent": AttributeCatalogEntry("boolean", set()),
        }
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None, effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo("activity_b", ["p_mid"], 80, None, effects={}),
        }
        pr = _make_pr(sequence_net, transitions, catalog)
        registry = _make_registry(pr)
        self._inject_precondition(
            registry, "execute_activity_a", PDDLCondition.attr_is("risk", "high")
        )

        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_effect = [
            v for v in variants
            if any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]
        assert all(PDDLCondition.attr_is("risk", "low") not in v.preconditions for v in with_effect)
        assert len(with_effect) == 1
        assert PDDLCondition.attr_true("urgent") in with_effect[0].preconditions

    def test_value_conflicting_clause_skipped(self, sequence_net, cfg):
        """Value DT guard conflicting with existing preconditions is dropped."""
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=1.0,
            appearance_level=1, appearance_guards=None,
            value_probabilities={"flu": 0.6, "cold": 0.4},
            value_level=1,
            value_guards=EffectGuards(
                subtype="value",
                guards={
                    "flu":  [[Guard("risk", "high", negated=False)]],
                    "cold": [[Guard("risk", "low",  negated=False)]],
                },
                total_samples=100, dt_accuracy=0.85,
            ),
        )
        catalog = {
            "diagnosis": AttributeCatalogEntry("categorical", {"flu", "cold"}),
            "risk": AttributeCatalogEntry("categorical", {"high", "low"}),
        }
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None, effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo("activity_b", ["p_mid"], 80, None, effects={}),
        }
        pr = _make_pr(sequence_net, transitions, catalog)
        registry = _make_registry(pr)
        self._inject_precondition(
            registry, "execute_activity_a", PDDLCondition.attr_is("risk", "high")
        )

        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        cold_variants = [v for v in variants if PDDLEffect.set_attr_is("diagnosis", "cold") in v.effects]
        flu_variants = [v for v in variants if PDDLEffect.set_attr_is("diagnosis", "flu") in v.effects]

        assert len(cold_variants) == 0
        assert len(flu_variants) >= 1

    def test_boolean_conflict_skipped(self, sequence_net, cfg):
        """(urgent_true) is skipped when existing precondition is (urgent_false)."""
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.8,
            appearance_level=1,
            appearance_guards=EffectGuards(
                subtype="appearance",
                guards={"appears": [[Guard("urgent", None, negated=False)]]},
                total_samples=100, dt_accuracy=0.9,
            ),
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        catalog = {
            "diagnosis": AttributeCatalogEntry("categorical", {"flu", "cold"}),
            "urgent": AttributeCatalogEntry("boolean", set()),
        }
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None, effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo("activity_b", ["p_mid"], 80, None, effects={}),
        }
        pr = _make_pr(sequence_net, transitions, catalog)
        registry = _make_registry(pr)
        self._inject_precondition(
            registry, "execute_activity_a", PDDLCondition.attr_false("urgent")
        )

        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_effect = [
            v for v in variants
            if any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]
        assert len(with_effect) == 0

    def test_different_attribute_no_false_positive(self, sequence_net, cfg):
        """Conditions on different attributes never cause a conflict."""
        effect = EffectInfo(
            attribute="diagnosis", presence_probability=0.8,
            appearance_level=1,
            appearance_guards=EffectGuards(
                subtype="appearance",
                guards={"appears": [[Guard("urgent", None, negated=False)]]},
                total_samples=100, dt_accuracy=0.9,
            ),
            value_probabilities={"flu": 1.0},
            value_level=1, value_guards=None,
        )
        catalog = {
            "diagnosis": AttributeCatalogEntry("categorical", {"flu", "cold"}),
            "urgent": AttributeCatalogEntry("boolean", set()),
        }
        transitions = {
            "activity_a": TransitionInfo(
                "activity_a", ["p_start"], 100, None, effects={"diagnosis": effect},
            ),
            "activity_b": TransitionInfo("activity_b", ["p_mid"], 80, None, effects={}),
        }
        pr = _make_pr(sequence_net, transitions, catalog)
        registry = _make_registry(pr)
        self._inject_precondition(
            registry, "execute_activity_a", PDDLCondition.attr_is("risk", "high")
        )

        EffectDuplicator(pr, cfg).apply(registry)

        variants = registry.get("execute_activity_a")
        with_effect = [
            v for v in variants
            if any(e.attribute == "diagnosis" and not e.clear for e in v.effects)
        ]
        assert len(with_effect) >= 1
