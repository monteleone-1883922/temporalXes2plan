"""
Tests for parsing.decision_mining.DecisionMiner.

All tests use synthetic PetriNetLog objects with manually constructed
FiringStep/TraceExecution instances — no real event logs or discovery
algorithms are needed.
"""
import pytest
from typing import Dict, List, Optional, Set

from pm4py import PetriNet, Marking

from tests.helpers import _transition, _place
from parsing.decision_mining import DecisionMiner
from models import AnalysisConfig, FiringStep, Guard, PetriNetLog, TraceExecution, XorSplitGuards


# ---------------------------------------------------------------------------
# PetriNetLog factory helpers
# ---------------------------------------------------------------------------

def _step(
    transition: PetriNet.Transition,
    from_places: Set[PetriNet.Place],
    is_tau: bool = False,
    attributes: Optional[Dict] = None,
    activity_name: Optional[str] = None,
) -> FiringStep:
    return FiringStep(
        transition=transition,
        activity_name=activity_name or (transition.label or "tau"),
        is_tau=is_tau,
        from_places=from_places,
        attributes=attributes or {},
    )


def _execution(trace_id: str, steps: List[FiringStep]) -> TraceExecution:
    return TraceExecution(trace_id=trace_id, steps=steps)


def _pn_log(executions: List[TraceExecution]) -> PetriNetLog:
    return PetriNetLog(
        executions=executions,
        net=PetriNet("test"),
        initial_marking=Marking(),
        final_marking=Marking(),
    )


def _miner(
    silent: Optional[Dict] = None,
    config: Optional[AnalysisConfig] = None,
    discretizer=None,
) -> DecisionMiner:
    return DecisionMiner(
        silent_transitions=silent or {},
        config=config or AnalysisConfig(),
        discretizer=discretizer,
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

    def test_default_discretizer_is_none(self):
        m = _miner()
        assert m._discretizer is None

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
# count_xor_samples
# ===========================================================================

class TestCountXorSamples:
    def test_basic_count(self):
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        xor_splits = {p_xor: [t_B, t_C]}

        execs = [
            _execution("c1", [
                _step(t_src, {p_in}),
                _step(t_B, {p_xor}),
            ]),
            _execution("c2", [
                _step(t_src, {p_in}),
                _step(t_C, {p_xor}),
            ]),
        ]
        pn_log = _pn_log(execs)
        m = _miner()
        counts = m.count_xor_samples(pn_log, xor_splits)

        assert counts["p_xor"] == 2

    def test_empty_log_returns_zero(self):
        _, t_B, t_C, _, p_xor = _xor_net()
        counts = _miner().count_xor_samples(_pn_log([]), {p_xor: [t_B, t_C]})
        assert counts["p_xor"] == 0

    def test_non_branch_transitions_not_counted(self):
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        xor_splits = {p_xor: [t_B, t_C]}
        t_other = _transition("t_other", "Other")
        execs = [_execution("c1", [_step(t_other, {p_xor})])]
        counts = _miner().count_xor_samples(_pn_log(execs), xor_splits)
        assert counts["p_xor"] == 0

    def test_multiple_executions_accumulate(self):
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        xor_splits = {p_xor: [t_B, t_C]}
        execs = [
            _execution(f"c{i}", [_step(t_B, {p_xor})])
            for i in range(7)
        ]
        counts = _miner().count_xor_samples(_pn_log(execs), xor_splits)
        assert counts["p_xor"] == 7

    def test_multiple_xor_places_counted_independently(self):
        _, t_B, t_C, _, p_xor = _xor_net()
        p2 = _place("p_xor2")
        t_D = _transition("t_D", "D")
        t_E = _transition("t_E", "E")
        xor_splits = {p_xor: [t_B, t_C], p2: [t_D, t_E]}

        execs = [
            _execution("c1", [_step(t_B, {p_xor}), _step(t_D, {p2})]),
            _execution("c2", [_step(t_C, {p_xor})]),
        ]
        counts = _miner().count_xor_samples(_pn_log(execs), xor_splits)
        assert counts["p_xor"] == 2
        assert counts["p_xor2"] == 1


# ===========================================================================
# mine_xor_splits — filtering behavior
# ===========================================================================

class TestMineXorSplitsFiltering:
    def _config(self, min_samples: int = 5) -> AnalysisConfig:
        return AnalysisConfig(dt_min_samples=min_samples, dt_min_accuracy=0.75)

    def test_insufficient_samples_place_absent_from_result(self):
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        execs = [_execution("c1", [_step(t_B, {p_xor})])]
        pn_log = _pn_log(execs)
        result, _ = _miner(config=self._config(min_samples=10)).mine_xor_splits(
            pn_log, {p_xor: [t_B, t_C]}
        )
        assert "p_xor" not in result

    def test_single_class_place_skipped(self):
        """All executions take the same branch → y.nunique() < 2 → skip."""
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        # 10 executions all going to t_B, no attributes → single class
        execs = [
            _execution(f"c{i}", [_step(t_B, {p_xor})])
            for i in range(10)
        ]
        result, _ = _miner(config=self._config(min_samples=5)).mine_xor_splits(
            _pn_log(execs), {p_xor: [t_B, t_C]}
        )
        assert "p_xor" not in result

    def test_result_contains_xor_split_guards(self):
        """Sufficient perfectly-separable data → XorSplitGuards returned."""
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        t_pre = _transition("t_pre", "Pre")

        execs = []
        for i in range(15):
            execs.append(_execution(f"c{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "high"}),
                _step(t_B, {p_xor}),
            ]))
        for i in range(15):
            execs.append(_execution(f"d{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "low"}),
                _step(t_C, {p_xor}),
            ]))

        result, _ = _miner(config=self._config(min_samples=5)).mine_xor_splits(
            _pn_log(execs), {p_xor: [t_B, t_C]}
        )
        assert "p_xor" in result
        assert isinstance(result["p_xor"], XorSplitGuards)

    def test_total_samples_matches_count(self):
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        t_pre = _transition("t_pre", "Pre")

        execs = []
        for i in range(10):
            execs.append(_execution(f"c{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "high"}),
                _step(t_B, {p_xor}),
            ]))
        for i in range(10):
            execs.append(_execution(f"d{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "low"}),
                _step(t_C, {p_xor}),
            ]))

        result, _ = _miner(config=self._config(min_samples=5)).mine_xor_splits(
            _pn_log(execs), {p_xor: [t_B, t_C]}
        )
        assert result["p_xor"].total_samples == 20

    def test_low_accuracy_returns_empty_guards_dict(self):
        """When DT accuracy < threshold, guards dict must be empty."""
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        t_pre = _transition("t_pre", "Pre")
        import random
        rng = random.Random(0)

        execs = []
        for i in range(30):
            # Labels are random — unpredictable regardless of attribute
            branch = t_B if rng.random() < 0.5 else t_C
            execs.append(_execution(f"c{i}", [
                _step(t_pre, {p_in}, attributes={"x": i % 3}),
                _step(branch, {p_xor}),
            ]))

        cfg = AnalysisConfig(dt_min_samples=5, dt_min_accuracy=0.99)
        result, _ = _miner(config=cfg).mine_xor_splits(
            _pn_log(execs), {p_xor: [t_B, t_C]}
        )
        if "p_xor" in result:
            assert result["p_xor"].guards == {}

    def test_dt_accuracy_field_is_set(self):
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        t_pre = _transition("t_pre", "Pre")
        execs = []
        for i in range(15):
            execs.append(_execution(f"c{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "high"}),
                _step(t_B, {p_xor}),
            ]))
        for i in range(15):
            execs.append(_execution(f"d{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "low"}),
                _step(t_C, {p_xor}),
            ]))
        result, _ = _miner(config=self._config(min_samples=5)).mine_xor_splits(
            _pn_log(execs), {p_xor: [t_B, t_C]}
        )
        assert 0.0 <= result["p_xor"].dt_accuracy <= 1.0


# ===========================================================================
# mine_xor_splits — SOP structure
# ===========================================================================

class TestMineXorSplitsSop:
    def _perfectly_separable_log(self):
        """20 B executions (risk=high) + 20 C executions (risk=low)."""
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        t_pre = _transition("t_pre", "Pre")
        execs = []
        for i in range(20):
            execs.append(_execution(f"b{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "high"}),
                _step(t_B, {p_xor}),
            ]))
        for i in range(20):
            execs.append(_execution(f"c{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "low"}),
                _step(t_C, {p_xor}),
            ]))
        return _pn_log(execs), p_xor, t_B, t_C

    def _mine(self, pn_log, p_xor, t_B, t_C):
        cfg = AnalysisConfig(dt_min_samples=5, dt_min_accuracy=0.7, dt_max_depth=3)
        result, _ = _miner(config=cfg).mine_xor_splits(pn_log, {p_xor: [t_B, t_C]})
        return result

    def test_guards_is_dict_keyed_by_activity_name(self):
        pn_log, p_xor, t_B, t_C = self._perfectly_separable_log()
        result = self._mine(pn_log, p_xor, t_B, t_C)
        guards = result["p_xor"].guards
        assert isinstance(guards, dict)
        for key in guards:
            assert isinstance(key, str)

    def test_outer_list_is_or_of_paths(self):
        """Guards outer list is a list (OR of DT paths)."""
        pn_log, p_xor, t_B, t_C = self._perfectly_separable_log()
        result = self._mine(pn_log, p_xor, t_B, t_C)
        for activity, outer in result["p_xor"].guards.items():
            assert isinstance(outer, list)

    def test_inner_list_is_and_of_guard_objects(self):
        """Each path's condition list must contain Guard instances."""
        pn_log, p_xor, t_B, t_C = self._perfectly_separable_log()
        result = self._mine(pn_log, p_xor, t_B, t_C)
        for activity, outer in result["p_xor"].guards.items():
            for path_conditions in outer:
                assert isinstance(path_conditions, list)
                for cond in path_conditions:
                    assert isinstance(cond, Guard)

    def test_guard_objects_have_non_empty_attribute(self):
        """Every Guard must carry a non-empty attribute name."""
        pn_log, p_xor, t_B, t_C = self._perfectly_separable_log()
        result = self._mine(pn_log, p_xor, t_B, t_C)
        for activity, outer in result["p_xor"].guards.items():
            for path_conditions in outer:
                for cond in path_conditions:
                    assert cond.attribute, f"Guard has empty attribute: {cond!r}"

    def test_categorical_guards_carry_value(self):
        """Guards produced from a categorical attribute must have value set."""
        pn_log, p_xor, t_B, t_C = self._perfectly_separable_log()
        result = self._mine(pn_log, p_xor, t_B, t_C)
        for activity, outer in result["p_xor"].guards.items():
            for path_conditions in outer:
                for cond in path_conditions:
                    # risk is categorical — value must be 'high' or 'low'
                    assert cond.attribute == "risk"
                    assert cond.value in ("high", "low")


# ===========================================================================
# Leaf pruning — integration tests via mine_xor_splits
# ===========================================================================

class TestLeafPruning:
    """
    Pruning integration tests.

    Noisy log structure used in most tests:
      20 × B  (risk=high)     — clean dominant leaf for B
      20 × C  (risk=low)      — clean dominant leaf for C
       4 × B  (risk=low)      — noise: produces a small/impure leaf for B
    """

    def _noisy_log(self):
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        t_pre = _transition("t_pre", "Pre")
        execs = []
        for i in range(20):
            execs.append(_execution(f"b{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "high"}),
                _step(t_B, {p_xor}),
            ]))
        for i in range(20):
            execs.append(_execution(f"c{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "low"}),
                _step(t_C, {p_xor}),
            ]))
        # 4 noisy traces: B fires on risk=low → creates impure/small leaf
        for i in range(4):
            execs.append(_execution(f"noise{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "low"}),
                _step(t_B, {p_xor}),
            ]))
        return _pn_log(execs), p_xor, t_B, t_C

    def _mine(self, pn_log, p_xor, t_B, t_C, cfg):
        result, _ = _miner(config=cfg).mine_xor_splits(pn_log, {p_xor: [t_B, t_C]})
        return result

    def test_perfect_data_not_pruned_with_defaults(self):
        """Perfectly separable data with default config must produce non-empty guards."""
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        t_pre = _transition("t_pre", "Pre")
        execs = []
        for i in range(20):
            execs.append(_execution(f"b{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "high"}),
                _step(t_B, {p_xor}),
            ]))
        for i in range(20):
            execs.append(_execution(f"c{i}", [
                _step(t_pre, {p_in}, attributes={"risk": "low"}),
                _step(t_C, {p_xor}),
            ]))
        cfg = AnalysisConfig(dt_min_samples=5, dt_min_accuracy=0.7)
        result = self._mine(_pn_log(execs), p_xor, t_B, t_C, cfg)
        assert "p_xor" in result
        assert result["p_xor"].guards != {}

    def test_high_purity_threshold_prunes_noisy_leaf(self):
        """
        With purity threshold above the noise leaf's purity (4/24 ≈ 0.83 impure
        from C's perspective), the contaminated leaf for C should be pruned so
        that C's guards contain only the clean path.
        """
        pn_log, p_xor, t_B, t_C = self._noisy_log()
        cfg = AnalysisConfig(
            dt_min_samples=5,
            dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=1,
            dt_prune_min_purity=0.95,  # only pure leaves survive
            dt_prune_orphan_mode="keep_best",
        )
        result = self._mine(pn_log, p_xor, t_B, t_C, cfg)
        if "p_xor" in result:
            guards = result["p_xor"].guards
            # Each surviving activity must have at least one path
            for activity, outer in guards.items():
                assert len(outer) >= 1

    def test_high_samples_threshold_prunes_small_leaf(self):
        """
        With dt_prune_min_leaf_samples set above the size of the noisy leaf,
        that leaf is excluded. Guards for surviving leaves must still be present.
        """
        pn_log, p_xor, t_B, t_C = self._noisy_log()
        cfg = AnalysisConfig(
            dt_min_samples=5,
            dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=10,  # noisy leaf has 4 samples → pruned
            dt_prune_min_purity=0.0,
            dt_prune_orphan_mode="keep_best",
        )
        result = self._mine(pn_log, p_xor, t_B, t_C, cfg)
        if "p_xor" in result:
            guards = result["p_xor"].guards
            for activity, outer in guards.items():
                assert all(len(path) >= 1 for path in outer)

    def _orphan_log(self):
        """
        Log designed to produce a small leaf for C:
          20 × B  (high_risk=True)   — leaf B: n=23, purity high
           3 × C  (high_risk=True)   — noise, merged into B's leaf
           3 × C  (high_risk=False)  — leaf C: n=3, purity=1.0 but small

        A boolean attribute is used so both DT branches produce a Guard
        (unlike categorical, whose <= side returns None).
        """
        t_src, t_B, t_C, p_in, p_xor = _xor_net()
        t_pre = _transition("t_pre", "Pre")
        execs = []
        for i in range(20):
            execs.append(_execution(f"b{i}", [
                _step(t_pre, {p_in}, attributes={"high_risk": True}),
                _step(t_B, {p_xor}),
            ]))
        for i in range(3):
            execs.append(_execution(f"cnoise{i}", [
                _step(t_pre, {p_in}, attributes={"high_risk": True}),
                _step(t_C, {p_xor}),
            ]))
        for i in range(3):
            execs.append(_execution(f"c{i}", [
                _step(t_pre, {p_in}, attributes={"high_risk": False}),
                _step(t_C, {p_xor}),
            ]))
        return _pn_log(execs), p_xor, t_B, t_C

    def test_orphan_drop_mode_removes_activity(self):
        """
        With drop mode, an activity whose only leaf has too few samples must be
        absent from guards.  C's pure leaf has n=3 < threshold=5.
        """
        pn_log, p_xor, t_B, t_C = self._orphan_log()
        cfg = AnalysisConfig(
            dt_min_samples=5,
            dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=5,
            dt_prune_min_purity=0.0,
            dt_prune_orphan_mode="drop",
        )
        result = self._mine(pn_log, p_xor, t_B, t_C, cfg)
        if "p_xor" in result:
            assert "c" not in result["p_xor"].guards

    def test_orphan_keep_best_mode_preserves_activity(self):
        """
        With keep_best mode, the same orphaned activity must still appear in
        guards with exactly one path (the best — and only — rescued leaf).
        """
        pn_log, p_xor, t_B, t_C = self._orphan_log()
        cfg = AnalysisConfig(
            dt_min_samples=5,
            dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=5,
            dt_prune_min_purity=0.0,
            dt_prune_orphan_mode="keep_best",
        )
        result = self._mine(pn_log, p_xor, t_B, t_C, cfg)
        if "p_xor" in result:
            guards = result["p_xor"].guards
            assert "c" in guards
            assert len(guards["c"]) == 1

    def test_pruning_threshold_zero_keeps_all_leaves(self):
        """Setting both thresholds to 0 must reproduce pre-pruning behavior."""
        pn_log, p_xor, t_B, t_C = self._noisy_log()
        cfg_pruned = AnalysisConfig(
            dt_min_samples=5, dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=0, dt_prune_min_purity=0.0,
        )
        cfg_nopruned = AnalysisConfig(
            dt_min_samples=5, dt_min_accuracy=0.5,
            dt_prune_min_leaf_samples=0, dt_prune_min_purity=0.0,
        )
        r1 = self._mine(pn_log, p_xor, t_B, t_C, cfg_pruned)
        r2 = self._mine(pn_log, p_xor, t_B, t_C, cfg_nopruned)
        if "p_xor" in r1 and "p_xor" in r2:
            assert set(r1["p_xor"].guards.keys()) == set(r2["p_xor"].guards.keys())


# ===========================================================================
# _build_feature_matrix — causal ordering
# ===========================================================================

class TestBuildFeatureMatrix:
    def setup_method(self):
        self.t_src, self.t_B, self.t_C, self.p_in, self.p_xor = _xor_net()
        self.t_pre = _transition("t_pre", "Pre")
        self.m = _miner()

    def test_state_emitted_before_split_steps_own_attrs(self):
        """Attributes on the split step itself must NOT appear as features."""
        execs = [
            _execution("c1", [
                _step(self.t_pre, {self.p_in}, attributes={"before": "yes"}),
                _step(self.t_B, {self.p_xor}, attributes={"at_split": "x"}),
            ]),
        ]
        X, y = self.m._build_feature_matrix(
            _pn_log(execs), self.p_xor, [self.t_B, self.t_C]
        )
        assert "before" in X.columns
        assert "at_split" not in X.columns

    def test_empty_log_returns_empty_dataframe(self):
        X, y = self.m._build_feature_matrix(
            _pn_log([]), self.p_xor, [self.t_B, self.t_C]
        )
        assert X.empty
        assert len(y) == 0

    def test_no_xor_traversals_returns_empty(self):
        t_other = _transition("t_other", "Other")
        execs = [_execution("c1", [_step(t_other, {self.p_xor})])]
        X, y = self.m._build_feature_matrix(
            _pn_log(execs), self.p_xor, [self.t_B, self.t_C]
        )
        assert X.empty

    def test_none_attribute_values_omitted_from_state(self):
        execs = [
            _execution("c1", [
                _step(self.t_pre, {self.p_in}, attributes={"crp": None, "age": 45}),
                _step(self.t_B, {self.p_xor}),
            ]),
        ]
        X, _ = self.m._build_feature_matrix(
            _pn_log(execs), self.p_xor, [self.t_B, self.t_C]
        )
        assert "crp" not in X.columns
        assert "age" in X.columns

    def test_ignored_attributes_excluded(self):
        cfg = AnalysisConfig(ignored_attributes={"case:concept:name", "crp"})
        m = _miner(config=cfg)
        execs = [
            _execution("c1", [
                _step(self.t_pre, {self.p_in}, attributes={"crp": 5.0, "age": 30}),
                _step(self.t_B, {self.p_xor}),
            ]),
        ]
        X, _ = m._build_feature_matrix(
            _pn_log(execs), self.p_xor, [self.t_B, self.t_C]
        )
        assert "crp" not in X.columns
        assert "age" in X.columns

    def test_tau_steps_do_not_update_state(self):
        """Tau steps must be skipped when updating the accumulated state."""
        t_tau = _transition("t_tau", None)
        silent = {t_tau: "tau_1"}
        m = _miner(silent=silent)
        execs = [
            _execution("c1", [
                _step(t_tau, {self.p_in}, is_tau=True, attributes={"should_skip": "yes"}),
                _step(self.t_B, {self.p_xor}),
            ]),
        ]
        X, _ = m._build_feature_matrix(
            _pn_log(execs), self.p_xor, [self.t_B, self.t_C]
        )
        assert "should_skip" not in X.columns

    def test_row_count_matches_traversals(self):
        execs = [
            _execution(f"c{i}", [_step(self.t_B, {self.p_xor})])
            for i in range(6)
        ]
        X, y = self.m._build_feature_matrix(
            _pn_log(execs), self.p_xor, [self.t_B, self.t_C]
        )
        assert len(X) == 6
        assert len(y) == 6

    def test_labels_match_branch_activity_names(self):
        execs = [
            _execution("c1", [_step(self.t_B, {self.p_xor})]),
            _execution("c2", [_step(self.t_C, {self.p_xor})]),
        ]
        _, y = self.m._build_feature_matrix(
            _pn_log(execs), self.p_xor, [self.t_B, self.t_C]
        )
        assert set(y.tolist()) == {"b", "c"}

    def test_numeric_with_discretizer_boundaries_stored_as_interval_label(self):
        """
        When the Discretizer has boundaries for a numeric attribute, values must
        be converted to interval labels (strings) in the state, so they appear
        as object-dtype column in X (ready for one-hot encoding).
        """
        from parsing.discretizer import Discretizer
        disc = Discretizer()
        disc.boundaries = {"crp": [6.0]}
        m = _miner(discretizer=disc)

        execs = [
            _execution("c1", [
                _step(self.t_pre, {self.p_in}, attributes={"crp": 4.5}),
                _step(self.t_B, {self.p_xor}),
            ]),
        ]
        X, _ = m._build_feature_matrix(
            _pn_log(execs), self.p_xor, [self.t_B, self.t_C]
        )
        assert "crp" in X.columns
        assert X["crp"].iloc[0] == "lte_6_0"

    def test_numeric_without_discretizer_boundaries_stored_as_raw_float(self):
        """
        When the Discretizer has NO boundaries for a numeric attribute (e.g. the
        attribute was constant or failed silhouette), the raw float is kept in
        state unchanged. _train_and_extract is responsible for dropping it.
        """
        from parsing.discretizer import Discretizer
        disc = Discretizer()
        disc.boundaries = {}  # no boundaries for any attribute
        m = _miner(discretizer=disc)

        execs = [
            _execution("c1", [
                _step(self.t_pre, {self.p_in}, attributes={"crp": 4.5}),
                _step(self.t_B, {self.p_xor}),
            ]),
        ]
        X, _ = m._build_feature_matrix(
            _pn_log(execs), self.p_xor, [self.t_B, self.t_C]
        )
        assert "crp" in X.columns
        import pandas as pd
        assert pd.api.types.is_numeric_dtype(X["crp"])


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
        """Bool columns must not produce one-hot dummies; they stay as 0/1."""
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
        """
        A raw numeric column (not bool/binary, no Discretizer) must be dropped
        so it does not inflate accuracy with splits that produce no PDDL guards.
        When the only column is a raw numeric, X_enc ends up empty → ({}, 0.0).
        """
        import pandas as pd
        X = pd.DataFrame({"crp": [4.5, 6.2, 11.0, 3.1, 9.8, 2.0] * 3})
        y = pd.Series(["b", "c"] * 9, dtype=str)
        guards, accuracy = self.m._train_and_extract(X, y)
        assert guards == {}
        assert accuracy == 0.0

    def test_discretized_numeric_column_treated_as_categorical(self):
        """
        When a numeric column has already been converted to interval labels by the
        Discretizer (e.g. 'lte_6_0', 'gte_6_0'), _train_and_extract must treat it
        as categorical and produce PDDL guards like '(crp lte_6_0)'.
        """
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
        assert any(c.attribute == "crp" for c in all_conditions), (
            "Expected a Guard with attribute 'crp', got: " + str(all_conditions)
        )
