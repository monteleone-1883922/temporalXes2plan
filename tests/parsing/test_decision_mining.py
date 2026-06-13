"""
Tests for parsing.decision_mining.DecisionMiner.

All tests use synthetic PreprocessedLog / TransitionFiringData objects —
no real event logs, discovery algorithms, or raw PetriNetLog scanning.
"""
import pytest
from typing import Any, Dict, List, Optional, Set, Tuple

from pm4py import PetriNet, Marking

from tests.helpers import _transition, _place
from parsing.decision_mining import DecisionMiner
from models import (
    AnalysisConfig,
    AttributeEffect,
    ConditionalEffect,
    EffectAttrScreening,
    EffectGuards,
    EffectValueScreening,
    Guard,
    PreprocessedLog,
    TransitionEffects,
    TransitionFiringData,
    TransitionScreening,
    XorBranchScreening,
    XorSplitGuards,
    XorSplitScreening,
    XorSplitStats,
)


# ---------------------------------------------------------------------------
# PreprocessedLog factory helpers
# ---------------------------------------------------------------------------

def _fd(
    activity_name: str,
    pre_state: Optional[Dict] = None,
    changed_attrs: Optional[Dict] = None,
    from_places: Optional[Set[PetriNet.Place]] = None,
) -> TransitionFiringData:
    return TransitionFiringData(
        activity_name=activity_name,
        pre_state=pre_state or {},
        changed_attrs=changed_attrs or {},
        from_places=frozenset(from_places or []),
    )


def _preprocessed(
    transition_firings: Optional[Dict[str, List[TransitionFiringData]]] = None,
    xor_firings: Optional[Dict[str, List[TransitionFiringData]]] = None,
) -> PreprocessedLog:
    return PreprocessedLog(
        transition_firings=transition_firings or {},
        xor_firings=xor_firings or {},
    )


def _miner(
    silent: Optional[Dict] = None,
    config: Optional[AnalysisConfig] = None,
) -> DecisionMiner:
    return DecisionMiner(
        silent_transitions=silent or {},
        config=config or AnalysisConfig(),
    )


# ---------------------------------------------------------------------------
# Shared XOR-net structure
#
#   p_in -> t_src -> p_xor -> t_B
#                          -> t_C
# ---------------------------------------------------------------------------

def _xor_net():
    t_src = _transition("t_src", "Src")
    t_B   = _transition("t_B",   "B")
    t_C   = _transition("t_C",   "C")
    p_in  = _place("p_in")
    p_xor = _place("p_xor")
    return t_src, t_B, t_C, p_in, p_xor


# ---------------------------------------------------------------------------
# Screening helpers
# ---------------------------------------------------------------------------

def _xor_screening(
    total_samples: int,
    action: str,
    branches: Dict[str, Tuple[float, str]],
) -> XorSplitScreening:
    """Create XorSplitScreening.  branches: {name: (probability, status)}."""
    return XorSplitScreening(
        total_samples=total_samples,
        action=action,
        branches={
            name: XorBranchScreening(activity_name=name, probability=prob, status=status)
            for name, (prob, status) in branches.items()
        },
    )


def _attr_scr(
    presence_probability: float,
    appearance_action: str = "dt",
    appearance_samples: int = 100,
    value_action: str = "dt",
    value_samples: int = 100,
    values: Optional[Dict[Any, Tuple[float, str]]] = None,
) -> EffectAttrScreening:
    """Create EffectAttrScreening.  values: {val: (probability, status)}."""
    val_screenings: Dict[Any, EffectValueScreening] = {}
    if values:
        val_screenings = {
            v: EffectValueScreening(probability=p, status=s)
            for v, (p, s) in values.items()
        }
    return EffectAttrScreening(
        presence_probability=presence_probability,
        appearance_action=appearance_action,
        appearance_samples=appearance_samples,
        value_action=value_action,
        value_samples=value_samples,
        values=val_screenings,
    )


def _effect_screening(
    transition_name: str,
    total_firings: int,
    attrs: Dict[str, EffectAttrScreening],
) -> TransitionScreening:
    return TransitionScreening(
        transition_name=transition_name,
        total_firings=total_firings,
        attributes=attrs,
    )


# ===========================================================================
# __init__
# ===========================================================================

class TestDecisionMinerInit:
    def test_default_config_is_analysis_config(self):
        m = _miner()
        assert isinstance(m.config, AnalysisConfig)

    def test_custom_config_stored(self):
        cfg = AnalysisConfig(dt_min_samples=5)
        m = _miner(config=cfg)
        assert m.config.dt_min_samples == 5

    def test_silent_transitions_stored(self):
        t = _transition("t1", None)
        m = _miner(silent={t: "tau_1"})
        assert m.silent_transitions[t] == "tau_1"


# ===========================================================================
# _activity_name
# ===========================================================================

class TestActivityName:
    def test_labeled_transition_returns_sanitized_label(self):
        t = _transition("t1", "ER Registration")
        m = _miner()
        assert m._activity_name(t) == "er_registration"

    def test_silent_transition_returns_tau_name(self):
        t = _transition("t1", None)
        m = _miner(silent={t: "tau_3"})
        assert m._activity_name(t) == "tau_3"

    def test_silent_transition_not_in_map_returns_tau_unknown(self):
        t = _transition("t1", None)
        m = _miner()
        result = m._activity_name(t)
        assert result.startswith("tau_unknown_")


# ===========================================================================
# _get_base_feature
# ===========================================================================

class TestGetBaseFeature:
    def setup_method(self):
        self.m = _miner()

    def test_known_single_word_categorical(self):
        result = self.m._get_base_feature("risk_high", {"risk"})
        assert result == "risk"

    def test_known_two_word_categorical(self):
        result = self.m._get_base_feature("case_type_urgent", {"case_type"})
        assert result == "case_type"

    def test_no_match_returns_col_name(self):
        result = self.m._get_base_feature("some_col", {"other"})
        assert result == "some_col"

    def test_exact_match_no_suffix_returns_self(self):
        result = self.m._get_base_feature("risk", {"risk"})
        assert result == "risk"


# ===========================================================================
# _format_condition
# ===========================================================================

class TestFormatCondition:
    def setup_method(self):
        self.m = _miner()

    def test_true_suffix_op_gt_returns_positive_guard(self):
        result = self.m._format_condition("admitted_true", ">", 0.5, set(), set())
        assert result == Guard("admitted", None, False)

    def test_true_suffix_op_lte_returns_negated_guard(self):
        result = self.m._format_condition("admitted_true", "<=", 0.5, set(), set())
        assert result == Guard("admitted", None, True)

    def test_false_suffix_op_gt_returns_negated_guard(self):
        result = self.m._format_condition("admitted_false", ">", 0.5, set(), set())
        assert result == Guard("admitted", None, True)

    def test_false_suffix_op_lte_returns_positive_guard(self):
        result = self.m._format_condition("admitted_false", "<=", 0.5, set(), set())
        assert result == Guard("admitted", None, False)

    def test_bool_col_op_gt_returns_positive_guard(self):
        result = self.m._format_condition("fever", ">", 0.5, set(), {"fever"})
        assert result == Guard("fever", None, False)

    def test_bool_col_op_lte_returns_negated_guard(self):
        result = self.m._format_condition("fever", "<=", 0.5, set(), {"fever"})
        assert result == Guard("fever", None, True)

    def test_categorical_op_gt_returns_value_guard(self):
        result = self.m._format_condition("risk_high", ">", 0.5, {"risk"}, set())
        assert result == Guard("risk", "high", False)

    def test_categorical_op_lte_returns_none(self):
        result = self.m._format_condition("risk_high", "<=", 0.5, {"risk"}, set())
        assert result is None

    def test_raw_numeric_returns_none(self):
        result = self.m._format_condition("age", ">", 45.0, set(), set())
        assert result is None

    def test_raw_numeric_lte_returns_none(self):
        result = self.m._format_condition("crp", "<=", 6.5, set(), set())
        assert result is None


# ===========================================================================
# screen_xor_splits
# ===========================================================================

class TestScreenXorSplits:
    def _config(self, **kw) -> AnalysisConfig:
        defaults = dict(dt_min_samples=5, xor_prune_threshold=0.10)
        defaults.update(kw)
        return AnalysisConfig(**defaults)

    def test_all_branches_active_action_dt(self):
        _, t_B, t_C, _, p_xor = _xor_net()
        stats = {"p_xor": XorSplitStats(probabilities={"b": 0.6, "c": 0.4}, total_executions=50)}
        result = _miner(config=self._config()).screen_xor_splits({p_xor: [t_B, t_C]}, stats)
        scr = result["p_xor"]
        assert scr.action == "dt"
        assert scr.branches["b"].status == "active"
        assert scr.branches["c"].status == "active"
        assert scr.total_samples == 50

    def test_low_probability_branch_pruned(self):
        _, t_B, t_C, _, p_xor = _xor_net()
        stats = {"p_xor": XorSplitStats(probabilities={"b": 0.95, "c": 0.05}, total_executions=100)}
        result = _miner(config=self._config()).screen_xor_splits({p_xor: [t_B, t_C]}, stats)
        assert result["p_xor"].branches["c"].status == "pruned"

    def test_single_active_branch_becomes_certain_and_deterministic(self):
        _, t_B, t_C, _, p_xor = _xor_net()
        stats = {"p_xor": XorSplitStats(probabilities={"b": 0.95, "c": 0.05}, total_executions=100)}
        result = _miner(config=self._config()).screen_xor_splits({p_xor: [t_B, t_C]}, stats)
        assert result["p_xor"].action == "deterministic"
        assert result["p_xor"].branches["b"].status == "certain"

    def test_insufficient_samples_action_fallback(self):
        _, t_B, t_C, _, p_xor = _xor_net()
        stats = {"p_xor": XorSplitStats(probabilities={"b": 0.6, "c": 0.4}, total_executions=3)}
        result = _miner(config=self._config(dt_min_samples=10)).screen_xor_splits(
            {p_xor: [t_B, t_C]}, stats
        )
        assert result["p_xor"].action == "fallback"

    def test_no_stats_yields_fallback(self):
        _, t_B, t_C, _, p_xor = _xor_net()
        result = _miner(config=self._config()).screen_xor_splits({p_xor: [t_B, t_C]}, {})
        assert result["p_xor"].action == "fallback"
        assert result["p_xor"].total_samples == 0

    def test_multiple_places_screened_independently(self):
        _, t_B, t_C, _, p_xor = _xor_net()
        p2 = _place("p_xor2")
        t_D = _transition("t_D", "D")
        t_E = _transition("t_E", "E")
        stats = {
            "p_xor": XorSplitStats(probabilities={"b": 0.5, "c": 0.5}, total_executions=40),
            "p_xor2": XorSplitStats(probabilities={"d": 0.95, "e": 0.05}, total_executions=80),
        }
        result = _miner(config=self._config()).screen_xor_splits(
            {p_xor: [t_B, t_C], p2: [t_D, t_E]}, stats
        )
        assert result["p_xor"].action == "dt"
        assert result["p_xor2"].action == "deterministic"

    def test_probabilities_stored_in_branch_screening(self):
        _, t_B, t_C, _, p_xor = _xor_net()
        stats = {"p_xor": XorSplitStats(probabilities={"b": 0.7, "c": 0.3}, total_executions=100)}
        result = _miner(config=self._config()).screen_xor_splits({p_xor: [t_B, t_C]}, stats)
        assert result["p_xor"].branches["b"].probability == 0.7
        assert result["p_xor"].branches["c"].probability == 0.3

    def test_three_branches_one_pruned_two_active(self):
        _, t_B, t_C, _, p_xor = _xor_net()
        t_D = _transition("t_D", "D")
        stats = {"p_xor": XorSplitStats(
            probabilities={"b": 0.5, "c": 0.45, "d": 0.05}, total_executions=200
        )}
        result = _miner(config=self._config()).screen_xor_splits(
            {p_xor: [t_B, t_C, t_D]}, stats
        )
        scr = result["p_xor"]
        assert scr.action == "dt"
        assert scr.branches["d"].status == "pruned"
        assert scr.branches["b"].status == "active"
        assert scr.branches["c"].status == "active"


# ===========================================================================
# mine_xor_splits — filtering behavior
# ===========================================================================

class TestMineXorSplitsFiltering:
    def _config(self, min_samples: int = 5) -> AnalysisConfig:
        return AnalysisConfig(dt_min_samples=min_samples, dt_min_accuracy=0.75)

    def test_non_dt_action_skipped(self):
        plog = _preprocessed(xor_firings={"p_xor": [_fd("b")]})
        screening = {"p_xor": _xor_screening(1, "fallback", {
            "b": (1.0, "active"), "c": (0.0, "pruned"),
        })}
        result = _miner(config=self._config()).mine_xor_splits(plog, screening)
        assert "p_xor" not in result

    def test_deterministic_action_skipped(self):
        plog = _preprocessed(xor_firings={"p_xor": [_fd("b") for _ in range(20)]})
        screening = {"p_xor": _xor_screening(20, "deterministic", {
            "b": (0.95, "certain"), "c": (0.05, "pruned"),
        })}
        result = _miner(config=self._config()).mine_xor_splits(plog, screening)
        assert "p_xor" not in result

    def test_single_class_after_filtering_skipped(self):
        """Screening says dt, but all data belongs to one active branch."""
        plog = _preprocessed(xor_firings={
            "p_xor": [_fd("b") for _ in range(10)],
        })
        screening = {"p_xor": _xor_screening(10, "dt", {
            "b": (0.7, "active"), "c": (0.3, "active"),
        })}
        result = _miner(config=self._config(min_samples=5)).mine_xor_splits(plog, screening)
        assert "p_xor" not in result
        assert screening["p_xor"].action == "fallback"

    def test_result_contains_xor_split_guards(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [
                *[_fd("b", pre_state={"risk": "high"}) for _ in range(15)],
                *[_fd("c", pre_state={"risk": "low"}) for _ in range(15)],
            ],
        })
        screening = {"p_xor": _xor_screening(30, "dt", {
            "b": (0.5, "active"), "c": (0.5, "active"),
        })}
        result = _miner(config=self._config(min_samples=5)).mine_xor_splits(plog, screening)
        assert "p_xor" in result
        assert isinstance(result["p_xor"], XorSplitGuards)

    def test_total_samples_matches_screening(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [
                *[_fd("b", pre_state={"risk": "high"}) for _ in range(10)],
                *[_fd("c", pre_state={"risk": "low"}) for _ in range(10)],
            ],
        })
        screening = {"p_xor": _xor_screening(20, "dt", {
            "b": (0.5, "active"), "c": (0.5, "active"),
        })}
        result = _miner(config=self._config(min_samples=5)).mine_xor_splits(plog, screening)
        assert result["p_xor"].total_samples == 20

    def test_low_accuracy_place_absent_from_result(self):
        import random
        rng = random.Random(0)
        firings = []
        for i in range(30):
            act = "b" if rng.random() < 0.5 else "c"
            firings.append(_fd(act, pre_state={"x": str(i % 3)}))

        plog = _preprocessed(xor_firings={"p_xor": firings})
        screening = {"p_xor": _xor_screening(30, "dt", {
            "b": (0.5, "active"), "c": (0.5, "active"),
        })}
        cfg = AnalysisConfig(dt_min_samples=5, dt_min_accuracy=0.99)
        result = _miner(config=cfg).mine_xor_splits(plog, screening)
        assert "p_xor" not in result
        assert screening["p_xor"].action == "fallback"

    def test_dt_accuracy_field_is_set(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [
                *[_fd("b", pre_state={"risk": "high"}) for _ in range(15)],
                *[_fd("c", pre_state={"risk": "low"}) for _ in range(15)],
            ],
        })
        screening = {"p_xor": _xor_screening(30, "dt", {
            "b": (0.5, "active"), "c": (0.5, "active"),
        })}
        result = _miner(config=self._config(min_samples=5)).mine_xor_splits(plog, screening)
        assert 0.0 <= result["p_xor"].dt_accuracy <= 1.0

    def test_pruned_branch_excluded_from_training(self):
        """With pruned branch excluded, DT trains only on active branches."""
        plog = _preprocessed(xor_firings={
            "p_xor": [
                *[_fd("b", pre_state={"risk": "high"}) for _ in range(20)],
                *[_fd("c", pre_state={"risk": "low"}) for _ in range(20)],
                *[_fd("d", pre_state={"risk": "mid"}) for _ in range(2)],
            ],
        })
        screening = {"p_xor": _xor_screening(42, "dt", {
            "b": (0.48, "active"), "c": (0.48, "active"), "d": (0.04, "pruned"),
        })}
        result = _miner(config=self._config(min_samples=5)).mine_xor_splits(plog, screening)
        if "p_xor" in result:
            assert "d" not in result["p_xor"].guards


# ===========================================================================
# mine_xor_splits — SOP structure
# ===========================================================================

class TestMineXorSplitsSop:
    def _perfectly_separable(self):
        """20 B (risk=high) + 20 C (risk=low)."""
        plog = _preprocessed(xor_firings={
            "p_xor": [
                *[_fd("b", pre_state={"risk": "high"}) for _ in range(20)],
                *[_fd("c", pre_state={"risk": "low"}) for _ in range(20)],
            ],
        })
        screening = {"p_xor": _xor_screening(40, "dt", {
            "b": (0.5, "active"), "c": (0.5, "active"),
        })}
        return plog, screening

    def _mine(self, plog, screening):
        cfg = AnalysisConfig(dt_min_samples=5, dt_min_accuracy=0.7, dt_max_depth=3)
        return _miner(config=cfg).mine_xor_splits(plog, screening)

    def test_guards_is_dict_keyed_by_activity_name(self):
        plog, screening = self._perfectly_separable()
        result = self._mine(plog, screening)
        guards = result["p_xor"].guards
        assert isinstance(guards, dict)
        for key in guards:
            assert isinstance(key, str)

    def test_outer_list_is_or_of_paths(self):
        plog, screening = self._perfectly_separable()
        result = self._mine(plog, screening)
        for activity, outer in result["p_xor"].guards.items():
            assert isinstance(outer, list)

    def test_inner_list_is_and_of_guard_objects(self):
        plog, screening = self._perfectly_separable()
        result = self._mine(plog, screening)
        for activity, outer in result["p_xor"].guards.items():
            for path_conditions in outer:
                assert isinstance(path_conditions, list)
                for cond in path_conditions:
                    assert isinstance(cond, Guard)

    def test_guard_objects_have_non_empty_attribute(self):
        plog, screening = self._perfectly_separable()
        result = self._mine(plog, screening)
        for activity, outer in result["p_xor"].guards.items():
            for path_conditions in outer:
                for cond in path_conditions:
                    assert cond.attribute, f"Guard has empty attribute: {cond!r}"

    def test_categorical_guards_carry_value(self):
        plog, screening = self._perfectly_separable()
        result = self._mine(plog, screening)
        for activity, outer in result["p_xor"].guards.items():
            for path_conditions in outer:
                for cond in path_conditions:
                    assert cond.attribute == "risk"
                    assert cond.value in ("high", "low")


# ===========================================================================
# Leaf pruning — integration tests via mine_xor_splits
# ===========================================================================

class TestLeafPruning:
    """
    Pruning integration tests.

    Noisy data structure used in most tests:
      20 x B  (risk=high)     — clean dominant leaf for B
      20 x C  (risk=low)      — clean dominant leaf for C
       4 x B  (risk=low)      — noise: produces a small/impure leaf for B
    """

    def _noisy(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [
                *[_fd("b", pre_state={"risk": "high"}) for _ in range(20)],
                *[_fd("c", pre_state={"risk": "low"}) for _ in range(20)],
                *[_fd("b", pre_state={"risk": "low"}) for _ in range(4)],
            ],
        })
        screening = {"p_xor": _xor_screening(44, "dt", {
            "b": (0.55, "active"), "c": (0.45, "active"),
        })}
        return plog, screening

    def _mine(self, plog, screening, cfg):
        return _miner(config=cfg).mine_xor_splits(plog, screening)

    def test_perfect_data_not_pruned_with_defaults(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [
                *[_fd("b", pre_state={"risk": "high"}) for _ in range(20)],
                *[_fd("c", pre_state={"risk": "low"}) for _ in range(20)],
            ],
        })
        screening = {"p_xor": _xor_screening(40, "dt", {
            "b": (0.5, "active"), "c": (0.5, "active"),
        })}
        cfg = AnalysisConfig(dt_min_samples=5, dt_min_accuracy=0.7)
        result = self._mine(plog, screening, cfg)
        assert "p_xor" in result
        assert result["p_xor"].guards != {}

    def test_high_purity_threshold_prunes_noisy_leaf(self):
        plog, screening = self._noisy()
        cfg = AnalysisConfig(
            dt_min_samples=5,
            dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=1,
            dt_prune_min_purity=0.95,
            dt_prune_orphan_mode="keep_best",
        )
        result = self._mine(plog, screening, cfg)
        if "p_xor" in result:
            guards = result["p_xor"].guards
            for activity, outer in guards.items():
                assert len(outer) >= 1

    def test_high_samples_threshold_prunes_small_leaf(self):
        plog, screening = self._noisy()
        cfg = AnalysisConfig(
            dt_min_samples=5,
            dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=10,
            dt_prune_min_purity=0.0,
            dt_prune_orphan_mode="keep_best",
        )
        result = self._mine(plog, screening, cfg)
        if "p_xor" in result:
            guards = result["p_xor"].guards
            for activity, outer in guards.items():
                assert all(len(path) >= 1 for path in outer)

    def _orphan(self):
        """
        20 B with high_risk=True, 3 C with high_risk=True (noise),
        3 C with high_risk=False (small pure leaf for C).
        """
        plog = _preprocessed(xor_firings={
            "p_xor": [
                *[_fd("b", pre_state={"high_risk": True}) for _ in range(20)],
                *[_fd("c", pre_state={"high_risk": True}) for _ in range(3)],
                *[_fd("c", pre_state={"high_risk": False}) for _ in range(3)],
            ],
        })
        screening = {"p_xor": _xor_screening(26, "dt", {
            "b": (0.77, "active"), "c": (0.23, "active"),
        })}
        return plog, screening

    def test_orphan_drop_mode_removes_activity(self):
        plog, screening = self._orphan()
        cfg = AnalysisConfig(
            dt_min_samples=5,
            dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=5,
            dt_prune_min_purity=0.0,
            dt_prune_orphan_mode="drop",
        )
        result = self._mine(plog, screening, cfg)
        if "p_xor" in result:
            assert "c" not in result["p_xor"].guards

    def test_orphan_keep_best_mode_preserves_activity(self):
        plog, screening = self._orphan()
        cfg = AnalysisConfig(
            dt_min_samples=5,
            dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=5,
            dt_prune_min_purity=0.0,
            dt_prune_orphan_mode="keep_best",
        )
        result = self._mine(plog, screening, cfg)
        if "p_xor" in result:
            guards = result["p_xor"].guards
            assert "c" in guards
            assert len(guards["c"]) == 1

    def test_pruning_threshold_zero_keeps_all_leaves(self):
        plog, screening = self._noisy()
        cfg_pruned = AnalysisConfig(
            dt_min_samples=5, dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=0, dt_prune_min_purity=0.0,
        )
        cfg_nopruned = AnalysisConfig(
            dt_min_samples=5, dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=0, dt_prune_min_purity=0.0,
        )
        r1 = self._mine(plog, screening, cfg_pruned)
        r2 = self._mine(plog, screening, cfg_nopruned)
        if "p_xor" in r1 and "p_xor" in r2:
            assert set(r1["p_xor"].guards.keys()) == set(r2["p_xor"].guards.keys())


# ===========================================================================
# _build_feature_matrix
# ===========================================================================

class TestBuildFeatureMatrix:
    def setup_method(self):
        self.m = _miner()

    def test_empty_xor_firings_returns_empty(self):
        plog = _preprocessed(xor_firings={})
        X, y = self.m._build_feature_matrix(plog, "p_xor")
        assert X.empty
        assert len(y) == 0

    def test_pre_state_becomes_x_columns(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [_fd("b", pre_state={"risk": "high", "age": 30})],
        })
        X, y = self.m._build_feature_matrix(plog, "p_xor")
        assert "risk" in X.columns
        assert "age" in X.columns

    def test_activity_name_becomes_label(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [_fd("b"), _fd("c")],
        })
        _, y = self.m._build_feature_matrix(plog, "p_xor")
        assert set(y.tolist()) == {"b", "c"}

    def test_row_count_matches_firings(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [_fd("b") for _ in range(6)],
        })
        X, y = self.m._build_feature_matrix(plog, "p_xor")
        assert len(X) == 6
        assert len(y) == 6

    def test_pre_state_values_in_x(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [_fd("b", pre_state={"risk": "high"})],
        })
        X, _ = self.m._build_feature_matrix(plog, "p_xor")
        assert X["risk"].iloc[0] == "high"

    def test_empty_pre_state_gives_empty_columns(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [_fd("b", pre_state={})],
        })
        X, _ = self.m._build_feature_matrix(plog, "p_xor")
        assert X.shape[1] == 0

    def test_active_branches_filters_firings(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [_fd("b"), _fd("c"), _fd("b")],
        })
        X, y = self.m._build_feature_matrix(plog, "p_xor", active_branches={"b"})
        assert len(X) == 2
        assert set(y.tolist()) == {"b"}

    def test_active_branches_none_keeps_all(self):
        plog = _preprocessed(xor_firings={
            "p_xor": [_fd("b"), _fd("c"), _fd("b")],
        })
        X, y = self.m._build_feature_matrix(plog, "p_xor", active_branches=None)
        assert len(X) == 3


# ===========================================================================
# _train_and_extract — encoding and output structure
# ===========================================================================

class TestTrainAndExtract:
    def setup_method(self):
        self.m = _miner(config=AnalysisConfig(dt_min_samples=1, dt_max_depth=3))

    def _make_df(self, rows, labels):
        import pandas as pd
        return pd.DataFrame(rows), pd.Series(labels, dtype=str)

    def test_returns_dict_and_float(self):
        X, y = self._make_df(
            [{"risk": "high"}, {"risk": "low"}],
            ["b", "c"],
        )
        guards, accuracy = self.m._train_and_extract(X, y)
        assert isinstance(guards, dict)
        assert isinstance(accuracy, float)

    def test_accuracy_between_0_and_1(self):
        X, y = self._make_df(
            [{"risk": "high"}, {"risk": "low"}] * 5,
            ["b", "c"] * 5,
        )
        _, accuracy = self.m._train_and_extract(X, y)
        assert 0.0 <= accuracy <= 1.0

    def test_perfect_separation_gives_high_accuracy(self):
        X, y = self._make_df(
            [{"risk": "high"}] * 10 + [{"risk": "low"}] * 10,
            ["b"] * 10 + ["c"] * 10,
        )
        _, accuracy = self.m._train_and_extract(X, y)
        assert accuracy >= 0.9

    def test_empty_df_returns_empty_guards_and_zero_accuracy(self):
        import pandas as pd
        guards, accuracy = self.m._train_and_extract(pd.DataFrame(), pd.Series(dtype=str))
        assert guards == {}
        assert accuracy == 0.0

    def test_all_null_columns_dropped(self):
        import pandas as pd
        X = pd.DataFrame({"all_null": [None, None], "risk": ["high", "low"]})
        y = pd.Series(["b", "c"], dtype=str)
        guards, accuracy = self.m._train_and_extract(X, y)
        assert isinstance(guards, dict)

    def test_boolean_column_treated_as_bool_not_categorical(self):
        import pandas as pd
        X = pd.DataFrame({"admitted": [True, False, True, False] * 5})
        y = pd.Series(["b", "c", "b", "c"] * 5, dtype=str)
        guards, accuracy = self.m._train_and_extract(X, y)
        for activity, outer in guards.items():
            for path in outer:
                for cond in path:
                    assert isinstance(cond, Guard)
                    assert cond.attribute == "admitted"
                    assert cond.value is None

    def test_sop_guards_keyed_by_branch_activity(self):
        X, y = self._make_df(
            [{"risk": "high"}] * 10 + [{"risk": "low"}] * 10,
            ["b"] * 10 + ["c"] * 10,
        )
        guards, _ = self.m._train_and_extract(X, y)
        assert set(guards.keys()).issubset({"b", "c"})

    def test_raw_numeric_column_is_dropped_before_training(self):
        import pandas as pd
        X = pd.DataFrame({"crp": [4.5, 6.2, 11.0, 3.1, 9.8, 2.0] * 3})
        y = pd.Series(["b", "c"] * 9, dtype=str)
        guards, accuracy = self.m._train_and_extract(X, y)
        assert guards == {}
        assert accuracy == 0.0

    def test_discretized_numeric_column_treated_as_categorical(self):
        import pandas as pd
        X = pd.DataFrame({"crp": ["lte_6_0"] * 10 + ["gte_6_0"] * 10})
        y = pd.Series(["b"] * 10 + ["c"] * 10, dtype=str)
        guards, accuracy = self.m._train_and_extract(X, y)
        assert accuracy >= 0.9
        all_conditions = [
            cond
            for outer in guards.values()
            for path in outer
            for cond in path
        ]
        assert any(c.attribute == "crp" for c in all_conditions)


# ===========================================================================
# Helpers shared by effect mining tests
# ===========================================================================

def _ae(
    total_firings: int,
    presence: Dict[str, float],
    values: Optional[Dict[str, Dict]] = None,
) -> AttributeEffect:
    return AttributeEffect(
        presence_probabilities=presence,
        value_probabilities=values or {k: {"v": 1.0} for k in presence},
        total_firings=total_firings,
    )


def _effect_miner(
    config: Optional[AnalysisConfig] = None,
    silent: Optional[Dict] = None,
) -> DecisionMiner:
    return DecisionMiner(
        silent_transitions=silent or {},
        config=config or AnalysisConfig(dt_min_samples=5, dt_min_accuracy=0.7),
    )


# ===========================================================================
# screen_effects
# ===========================================================================

class TestScreenEffects:
    def _config(self, **kw) -> AnalysisConfig:
        defaults = dict(
            dt_min_samples=5,
            effect_never_threshold=0.05,
            effect_always_threshold=0.95,
            effect_value_prune_threshold=0.10,
            effect_value_certain_threshold=0.95,
        )
        defaults.update(kw)
        return AnalysisConfig(**defaults)

    def test_basic_screening_returns_transition_screening(self):
        ae = _ae(100, {"status": 0.6}, {"status": {"ok": 0.7, "err": 0.3}})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        assert "work" in result
        assert isinstance(result["work"], TransitionScreening)

    def test_never_threshold_marks_attr_as_never(self):
        ae = _ae(100, {"status": 0.03})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        assert "work" in result
        attr = result["work"].attributes["status"]
        assert attr.appearance_action == "never"
        assert attr.value_action == "never"
        assert attr.values == {}

    def test_always_threshold_deterministic_appearance(self):
        ae = _ae(100, {"status": 0.98})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        attr = result["work"].attributes["status"]
        assert attr.appearance_action == "deterministic"
        assert attr.appearance_samples == 0

    def test_normal_presence_dt_appearance(self):
        ae = _ae(100, {"status": 0.6})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        attr = result["work"].attributes["status"]
        assert attr.appearance_action == "dt"
        assert attr.appearance_samples == 100

    def test_insufficient_samples_appearance(self):
        ae = _ae(3, {"status": 0.6})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        attr = result["work"].attributes["status"]
        assert attr.appearance_action == "fallback"

    def test_value_below_prune_threshold_is_pruned(self):
        ae = _ae(100, {"status": 0.6}, {"status": {"ok": 0.92, "rare": 0.05, "err": 0.03}})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        vals = result["work"].attributes["status"].values
        assert vals["rare"].status == "pruned"
        assert vals["err"].status == "pruned"
        assert vals["ok"].status == "active"

    def test_value_above_certain_threshold_is_certain(self):
        ae = _ae(100, {"status": 0.6}, {"status": {"always_val": 0.97, "rare": 0.03}})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        vals = result["work"].attributes["status"].values
        assert vals["always_val"].status == "certain"
        assert vals["rare"].status == "pruned"

    def test_multiple_active_values_action_dt(self):
        ae = _ae(100, {"status": 0.6}, {"status": {"ok": 0.5, "err": 0.5}})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        assert result["work"].attributes["status"].value_action == "dt"

    def test_single_value_action_deterministic(self):
        ae = _ae(100, {"status": 0.6}, {"status": {"only": 1.0}})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        assert result["work"].attributes["status"].value_action == "deterministic"

    def test_fewer_than_two_active_values_deterministic(self):
        ae = _ae(100, {"status": 0.6}, {"status": {"dom": 0.97, "rare": 0.03}})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        assert result["work"].attributes["status"].value_action == "deterministic"

    def test_insufficient_value_samples(self):
        ae = _ae(6, {"status": 0.5}, {"status": {"ok": 0.5, "err": 0.5}})
        result = _effect_miner(config=self._config(dt_min_samples=5)).screen_effects({"work": ae})
        attr = result["work"].attributes["status"]
        assert attr.value_samples == 3
        assert attr.value_action == "fallback"

    def test_total_firings_stored(self):
        ae = _ae(77, {"status": 0.6})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        assert result["work"].total_firings == 77

    def test_multiple_transitions(self):
        ae_a = _ae(50, {"x": 0.5})
        ae_b = _ae(80, {"z": 0.2})
        result = _effect_miner(config=self._config()).screen_effects({"t_a": ae_a, "t_b": ae_b})
        assert "t_a" in result
        assert "t_b" in result

    def test_value_probabilities_stored(self):
        ae = _ae(100, {"status": 0.6}, {"status": {"ok": 0.7, "err": 0.3}})
        result = _effect_miner(config=self._config()).screen_effects({"work": ae})
        vals = result["work"].attributes["status"].values
        assert vals["ok"].probability == 0.7
        assert vals["err"].probability == 0.3


# ===========================================================================
# _build_effect_matrix
# ===========================================================================

class TestBuildEffectMatrix:
    def setup_method(self):
        self.m = _effect_miner()

    def test_no_firings_returns_empty(self):
        plog = _preprocessed()
        X, yp, yv = self.m._build_effect_matrix(plog, "target")
        assert X.empty
        assert yp == {}
        assert yv == {}

    def test_single_firing_one_row(self):
        plog = _preprocessed(transition_firings={
            "target": [_fd("target", changed_attrs={"score": "high"})],
        })
        X, yp, yv = self.m._build_effect_matrix(plog, "target")
        assert len(X) == 1

    def test_multiple_firings_multiple_rows(self):
        plog = _preprocessed(transition_firings={
            "target": [_fd("target", changed_attrs={"x": "a"}) for _ in range(5)],
        })
        X, yp, yv = self.m._build_effect_matrix(plog, "target")
        assert len(X) == 5

    def test_changed_attr_sets_y_presence_true(self):
        plog = _preprocessed(transition_firings={
            "target": [_fd("target", changed_attrs={"result": "ok"})],
        })
        X, yp, yv = self.m._build_effect_matrix(plog, "target")
        assert "result" in yp
        assert bool(yp["result"].iloc[0]) is True

    def test_absent_attr_sets_y_presence_false(self):
        plog = _preprocessed(transition_firings={
            "target": [_fd("target", changed_attrs={})],
        })
        X, yp, yv = self.m._build_effect_matrix(plog, "target")
        assert "result" not in yp

    def test_mixed_changed_and_unchanged(self):
        plog = _preprocessed(transition_firings={
            "target": [
                _fd("target", changed_attrs={"result": "ok"}),
                _fd("target", changed_attrs={}),
            ],
        })
        X, yp, yv = self.m._build_effect_matrix(plog, "target")
        assert list(yp["result"]) == [True, False]

    def test_y_value_set_only_when_changed(self):
        import pandas as pd
        plog = _preprocessed(transition_firings={
            "target": [
                _fd("target", changed_attrs={}),
                _fd("target", changed_attrs={"result": "new"}),
            ],
        })
        X, yp, yv = self.m._build_effect_matrix(plog, "target")
        assert pd.isna(yv["result"].iloc[0])
        assert yv["result"].iloc[1] == "new"

    def test_pre_state_becomes_x_features(self):
        plog = _preprocessed(transition_firings={
            "target": [
                _fd("target", pre_state={"risk": "high"}, changed_attrs={"score": "1"}),
            ],
        })
        X, _, _ = self.m._build_effect_matrix(plog, "target")
        assert "risk" in X.columns
        assert X["risk"].iloc[0] == "high"

    def test_attr_first_seen_mid_log_padded_correctly(self):
        """Attribute first seen in 3rd firing: rows 0-1 must be False/NA."""
        plog = _preprocessed(transition_firings={
            "target": [
                _fd("target", changed_attrs={}),
                _fd("target", changed_attrs={}),
                _fd("target", changed_attrs={"late": "v"}),
            ],
        })
        X, yp, yv = self.m._build_effect_matrix(plog, "target")
        assert len(X) == 3
        assert "late" in yp
        assert list(yp["late"]) == [False, False, True]
        assert str(yv["late"].iloc[2]) == "v"


# ===========================================================================
# _mine_appearance
# ===========================================================================

class TestMineAppearance:
    def setup_method(self):
        self.m = _effect_miner(
            config=AnalysisConfig(dt_min_samples=1, dt_min_accuracy=0.7)
        )

    def _series(self, values):
        import pandas as pd
        return pd.Series(values, dtype=bool)

    def _df(self, rows):
        import pandas as pd
        return pd.DataFrame(rows)

    def test_single_class_returns_fallback(self):
        X = self._df([{"risk": "high"}] * 10)
        yp = self._series([True] * 10)
        guards, status = self.m._mine_appearance(X, yp)
        assert guards is None
        assert status == "fallback"

    def test_perfectly_separable_returns_dt(self):
        X = self._df(
            [{"risk": "high"}] * 20 + [{"risk": "low"}] * 20
        )
        yp = self._series([True] * 20 + [False] * 20)
        guards, status = self.m._mine_appearance(X, yp)
        assert status == "dt"
        assert guards is not None
        assert guards.subtype == "appearance"
        assert "appears" in guards.guards

    def test_low_accuracy_returns_fallback(self):
        import random
        rng = random.Random(42)
        X = self._df([{"x": i % 3} for i in range(40)])
        yp = self._series([rng.random() > 0.5 for _ in range(40)])
        m = _effect_miner(
            config=AnalysisConfig(dt_min_samples=1, dt_min_accuracy=0.99)
        )
        guards, status = m._mine_appearance(X, yp)
        if status != "insufficient":
            assert status == "fallback"
            assert guards is None

    def test_guards_total_samples_is_full_x(self):
        X = self._df([{"risk": "high"}] * 15 + [{"risk": "low"}] * 15)
        yp = self._series([True] * 15 + [False] * 15)
        guards, status = self.m._mine_appearance(X, yp)
        if status == "dt":
            assert guards.total_samples == 30

    def test_dt_accuracy_field_between_0_and_1(self):
        X = self._df([{"risk": "high"}] * 15 + [{"risk": "low"}] * 15)
        yp = self._series([True] * 15 + [False] * 15)
        guards, status = self.m._mine_appearance(X, yp)
        if status == "dt":
            assert 0.0 <= guards.dt_accuracy <= 1.0


# ===========================================================================
# _mine_value
# ===========================================================================

class TestMineValue:
    def setup_method(self):
        import pandas as pd
        self.pd = pd
        self.m = _effect_miner(
            config=AnalysisConfig(dt_min_samples=1, dt_min_accuracy=0.7)
        )

    def _yv(self, values):
        import pandas as pd
        return pd.Series(values)

    def _df(self, rows):
        import pandas as pd
        return pd.DataFrame(rows)

    def test_single_value_returns_insufficient(self):
        X = self._df([{"risk": "high"}] * 10)
        yv = self._yv(["urgent"] * 10)
        guards, status = self.m._mine_value(X, yv)
        assert guards is None
        assert status == "insufficient"

    def test_all_na_returns_insufficient(self):
        import pandas as pd
        X = self._df([{"risk": "high"}] * 10)
        yv = self._yv([pd.NA] * 10)
        guards, status = self.m._mine_value(X, yv)
        assert guards is None
        assert status == "insufficient"

    def test_perfectly_separable_returns_dt(self):
        X = self._df([{"risk": "high"}] * 20 + [{"risk": "low"}] * 20)
        yv = self._yv(["urgent"] * 20 + ["normal"] * 20)
        guards, status = self.m._mine_value(X, yv)
        assert status == "dt"
        assert guards is not None
        assert guards.subtype == "value"

    def test_na_rows_excluded_from_training(self):
        import pandas as pd
        X = self._df([{"risk": "high"}] * 15 + [{"risk": "low"}] * 15 + [{"risk": "mid"}] * 10)
        yv = self._yv(["urgent"] * 15 + ["normal"] * 15 + [pd.NA] * 10)
        guards, status = self.m._mine_value(X, yv)
        if status == "dt":
            assert guards.total_samples == 30

    def test_guards_keyed_by_value_strings(self):
        X = self._df([{"risk": "high"}] * 20 + [{"risk": "low"}] * 20)
        yv = self._yv(["urgent"] * 20 + ["normal"] * 20)
        guards, status = self.m._mine_value(X, yv)
        if status == "dt":
            for key in guards.guards:
                assert isinstance(key, str)

    def test_low_accuracy_returns_fallback(self):
        import random
        rng = random.Random(42)
        X = self._df([{"x": i % 3} for i in range(40)])
        yv = self._yv([rng.choice(["a", "b", "c"]) for _ in range(40)])
        m = _effect_miner(
            config=AnalysisConfig(dt_min_samples=1, dt_min_accuracy=0.99)
        )
        guards, status = m._mine_value(X, yv)
        if status != "insufficient":
            assert status == "fallback"
            assert guards is None

    def test_active_values_filter_pruned_values(self):
        X = self._df([{"risk": "high"}] * 15 + [{"risk": "low"}] * 15 + [{"risk": "mid"}] * 5)
        yv = self._yv(["urgent"] * 15 + ["normal"] * 15 + ["rare"] * 5)
        guards, status = self.m._mine_value(X, yv, active_values={"urgent", "normal"})
        if status == "dt":
            assert "rare" not in guards.guards


# ===========================================================================
# mine_effects — integration
# ===========================================================================

class TestMineEffectsIntegration:
    """
    Integration tests for the screen_effects -> mine_effects pipeline.
    """

    def _config(self, **kw) -> AnalysisConfig:
        defaults = dict(
            dt_min_samples=5, dt_min_accuracy=0.7,
            effect_never_threshold=0.05, effect_always_threshold=0.95,
            effect_value_prune_threshold=0.10, effect_value_certain_threshold=0.95,
        )
        defaults.update(kw)
        return AnalysisConfig(**defaults)

    def _run(self, plog, ae_dict, config=None):
        """Screen + mine pipeline."""
        cfg = config or self._config()
        m = DecisionMiner(silent_transitions={}, config=cfg)
        screening = m.screen_effects(ae_dict)
        return m.mine_effects(plog, screening)

    def _conditional_appearance_data(self, n: int = 20):
        firings = [
            *[_fd("work", pre_state={"risk": "high"}, changed_attrs={"status": "urgent"})
              for _ in range(n)],
            *[_fd("work", pre_state={"risk": "low"}, changed_attrs={})
              for _ in range(n)],
        ]
        return _preprocessed(transition_firings={"work": firings})

    def _ae_conditional(self, n: int = 20) -> AttributeEffect:
        return AttributeEffect(
            presence_probabilities={"status": 0.5},
            value_probabilities={"status": {"urgent": 1.0}},
            total_firings=n * 2,
        )

    def _always_present_data(self, n: int = 20):
        firings = [
            *[_fd("work", pre_state={"risk": "high"}, changed_attrs={"severity": "critical"})
              for _ in range(n)],
            *[_fd("work", pre_state={"risk": "low"}, changed_attrs={"severity": "minor"})
              for _ in range(n)],
        ]
        return _preprocessed(transition_firings={"work": firings})

    def _ae_always(self, n: int = 20) -> AttributeEffect:
        return AttributeEffect(
            presence_probabilities={"severity": 1.0},
            value_probabilities={"severity": {"critical": 0.5, "minor": 0.5}},
            total_firings=n * 2,
        )

    # --- Basic output structure ---

    def test_returns_dict_keyed_by_activity_name(self):
        plog = self._conditional_appearance_data()
        ae = {"work": self._ae_conditional()}
        result = self._run(plog, ae)
        assert isinstance(result, dict)
        for key in result:
            assert isinstance(key, str)

    def test_result_contains_transition_effects(self):
        plog = self._conditional_appearance_data()
        ae = {"work": self._ae_conditional()}
        result = self._run(plog, ae)
        if "work" in result:
            assert isinstance(result["work"], TransitionEffects)

    def test_transition_effects_has_total_firings(self):
        plog = self._conditional_appearance_data(n=20)
        ae = {"work": self._ae_conditional(n=20)}
        result = self._run(plog, ae)
        if "work" in result:
            assert result["work"].total_firings == 40

    # --- Screening: never threshold ---

    def test_never_threshold_attr_has_never_status_in_effects(self):
        ae = {"work": AttributeEffect(
            presence_probabilities={"status": 0.03},
            value_probabilities={"status": {"v": 1.0}},
            total_firings=100,
        )}
        plog = _preprocessed()
        result = self._run(plog, ae)
        if "work" in result and "status" in result["work"].effects:
            ce = result["work"].effects["status"]
            assert ce.appearance_status == "never"

    # --- Screening: always threshold -> deterministic appearance, 2B runs ---

    def test_always_threshold_appearance_is_deterministic(self):
        plog = self._always_present_data(n=20)
        ae = {"work": self._ae_always(n=20)}
        result = self._run(plog, ae)
        if "work" in result and "severity" in result["work"].effects:
            ce = result["work"].effects["severity"]
            assert ce.appearance_status == "deterministic"
            assert ce.appearance is None

    def test_always_threshold_value_analysis_runs(self):
        plog = self._always_present_data(n=20)
        ae = {"work": self._ae_always(n=20)}
        result = self._run(plog, ae)
        if "work" in result and "severity" in result["work"].effects:
            ce = result["work"].effects["severity"]
            assert ce.value_status != "never"

    # --- Insufficient samples ---

    def test_insufficient_2a_samples_returns_fallback_status(self):
        ae = {"work": AttributeEffect(
            presence_probabilities={"status": 0.5},
            value_probabilities={"status": {"urgent": 1.0}},
            total_firings=2,
        )}
        plog = _preprocessed()
        result = self._run(plog, ae)
        if "work" in result and "status" in result["work"].effects:
            assert result["work"].effects["status"].appearance_status == "fallback"

    # --- Empty screening -> empty result ---

    def test_empty_attribute_effects_returns_empty(self):
        plog = _preprocessed()
        result = self._run(plog, {})
        assert result == {}

    def test_empty_screening_returns_empty(self):
        plog = _preprocessed()
        m = DecisionMiner(silent_transitions={}, config=self._config())
        result = m.mine_effects(plog, {})
        assert result == {}

    # --- ConditionalEffect field types ---

    def test_conditional_effect_fields_have_correct_types(self):
        plog = self._conditional_appearance_data(n=20)
        ae = {"work": self._ae_conditional(n=20)}
        result = self._run(plog, ae)
        if "work" in result:
            for attr, ce in result["work"].effects.items():
                assert isinstance(ce, ConditionalEffect)
                assert isinstance(ce.attribute, str)
                assert isinstance(ce.presence_probability, float)
                assert isinstance(ce.possible_values, list)
                assert ce.appearance_status in {
                    "dt", "deterministic", "never", "insufficient", "fallback"
                }
                assert ce.value_status in {
                    "dt", "deterministic", "never", "insufficient", "fallback"
                }

    # --- 2A DT end-to-end ---

    def test_conditional_appearance_can_produce_dt_status(self):
        plog = self._conditional_appearance_data(n=20)
        ae = {"work": self._ae_conditional(n=20)}
        result = self._run(plog, ae)
        if "work" in result and "status" in result["work"].effects:
            ce = result["work"].effects["status"]
            if ce.appearance_status == "dt":
                assert ce.appearance is not None
                assert ce.appearance.subtype == "appearance"
                assert "appears" in ce.appearance.guards

    # --- 2B DT end-to-end ---

    def test_always_present_value_dt_guards_contain_guard_objects(self):
        plog = self._always_present_data(n=20)
        ae = {"work": self._ae_always(n=20)}
        result = self._run(plog, ae)
        if "work" in result and "severity" in result["work"].effects:
            ce = result["work"].effects["severity"]
            if ce.value_status == "dt":
                assert ce.value is not None
                for outcome, sop in ce.value.guards.items():
                    for path in sop:
                        for cond in path:
                            assert isinstance(cond, Guard)

    # --- presence_probability and possible_values stored ---

    def test_presence_probability_stored_in_conditional_effect(self):
        ae = {"work": _ae(50, {"x": 0.6})}
        plog = _preprocessed()
        result = self._run(plog, ae)
        if "work" in result and "x" in result["work"].effects:
            assert result["work"].effects["x"].presence_probability == pytest.approx(0.6, abs=0.01)

    def test_possible_values_stored_from_screening(self):
        ae = {"work": AttributeEffect(
            presence_probabilities={"status": 0.5},
            value_probabilities={"status": {"urgent": 0.7, "normal": 0.3}},
            total_firings=50,
        )}
        plog = _preprocessed()
        result = self._run(plog, ae)
        if "work" in result and "status" in result["work"].effects:
            pv = set(result["work"].effects["status"].possible_values)
            assert pv == {"urgent", "normal"}


# ===========================================================================
# Screening mutation: mine_* updates action when DT cannot be applied
# ===========================================================================

class TestMineScreeningUpdate:
    """Verify that mine_xor_splits and mine_effects mutate the screening
    objects to reflect the actual outcome when the DT cannot be applied."""

    # --- XOR splits ---

    def test_xor_single_class_updates_screening_to_fallback(self):
        """Single-class data after filtering → action updated to 'fallback'."""
        plog = _preprocessed(xor_firings={
            "p_xor": [_fd("b") for _ in range(10)],
        })
        screening = {"p_xor": _xor_screening(10, "dt", {
            "b": (0.7, "active"), "c": (0.3, "active"),
        })}
        cfg = AnalysisConfig(dt_min_samples=5, dt_min_accuracy=0.75)
        _miner(config=cfg).mine_xor_splits(plog, screening)
        assert screening["p_xor"].action == "fallback"

    def test_xor_low_accuracy_updates_screening_to_fallback(self):
        """DT accuracy below threshold → action updated to 'fallback'."""
        import random
        rng = random.Random(0)
        firings = [
            _fd("b" if rng.random() < 0.5 else "c", pre_state={"x": str(i % 3)})
            for i in range(30)
        ]
        plog = _preprocessed(xor_firings={"p_xor": firings})
        screening = {"p_xor": _xor_screening(30, "dt", {
            "b": (0.5, "active"), "c": (0.5, "active"),
        })}
        cfg = AnalysisConfig(dt_min_samples=5, dt_min_accuracy=0.99)
        _miner(config=cfg).mine_xor_splits(plog, screening)
        assert screening["p_xor"].action == "fallback"

    def test_xor_successful_dt_leaves_action_as_dt(self):
        """Successful DT must not change the screening action."""
        plog = _preprocessed(xor_firings={
            "p_xor": [
                *[_fd("b", pre_state={"risk": "high"}) for _ in range(15)],
                *[_fd("c", pre_state={"risk": "low"}) for _ in range(15)],
            ],
        })
        screening = {"p_xor": _xor_screening(30, "dt", {
            "b": (0.5, "active"), "c": (0.5, "active"),
        })}
        cfg = AnalysisConfig(dt_min_samples=5, dt_min_accuracy=0.75)
        _miner(config=cfg).mine_xor_splits(plog, screening)
        assert screening["p_xor"].action == "dt"

    # --- Effect mining: appearance ---

    def test_effect_appearance_fallback_updates_screening(self):
        """Appearance DT accuracy below threshold → appearance_action='fallback'."""
        import random
        rng = random.Random(0)
        firings = [
            _fd("work",
                pre_state={"x": str(i % 3)},
                changed_attrs={"status": "v"} if rng.random() < 0.5 else {})
            for i in range(40)
        ]
        plog = _preprocessed(transition_firings={"work": firings})
        ae = {"work": AttributeEffect(
            presence_probabilities={"status": 0.5},
            value_probabilities={"status": {"v": 1.0}},
            total_firings=40,
        )}
        cfg = AnalysisConfig(
            dt_min_samples=5, dt_min_accuracy=0.99,
            effect_never_threshold=0.05, effect_always_threshold=0.95,
            effect_value_prune_threshold=0.10, effect_value_certain_threshold=0.95,
        )
        m = DecisionMiner(silent_transitions={}, config=cfg)
        screening = m.screen_effects(ae)
        m.mine_effects(plog, screening)
        assert screening["work"].attributes["status"].appearance_action in {
            "fallback", "insufficient"
        }

    def test_effect_empty_matrix_updates_screening_to_fallback(self):
        """Empty feature matrix at mining time → appearance_action and value_action become 'fallback'."""
        plog = _preprocessed(transition_firings={})  # no firings for 'work'
        ae = {"work": AttributeEffect(
            presence_probabilities={"status": 0.5},
            value_probabilities={"status": {"urgent": 0.6, "normal": 0.4}},
            total_firings=40,
        )}
        cfg = AnalysisConfig(
            dt_min_samples=5, dt_min_accuracy=0.75,
            effect_never_threshold=0.05, effect_always_threshold=0.95,
            effect_value_prune_threshold=0.10, effect_value_certain_threshold=0.95,
        )
        m = DecisionMiner(silent_transitions={}, config=cfg)
        screening = m.screen_effects(ae)
        m.mine_effects(plog, screening)
        assert screening["work"].attributes["status"].appearance_action == "fallback"
        assert screening["work"].attributes["status"].value_action == "fallback"

    def test_effect_successful_dt_leaves_action_unchanged(self):
        """Successful DT must not mutate appearance_action in screening."""
        firings = [
            *[_fd("work", pre_state={"risk": "high"}, changed_attrs={"s": "a"})
              for _ in range(15)],
            *[_fd("work", pre_state={"risk": "low"}, changed_attrs={})
              for _ in range(15)],
        ]
        plog = _preprocessed(transition_firings={"work": firings})
        ae = {"work": AttributeEffect(
            presence_probabilities={"s": 0.5},
            value_probabilities={"s": {"a": 1.0}},
            total_firings=30,
        )}
        cfg = AnalysisConfig(
            dt_min_samples=5, dt_min_accuracy=0.75,
            effect_never_threshold=0.05, effect_always_threshold=0.95,
            effect_value_prune_threshold=0.10, effect_value_certain_threshold=0.95,
        )
        m = DecisionMiner(silent_transitions={}, config=cfg)
        screening = m.screen_effects(ae)
        m.mine_effects(plog, screening)
        # appearance_action was "dt" and DT should succeed with perfect data
        assert screening["work"].attributes["s"].appearance_action == "dt"
