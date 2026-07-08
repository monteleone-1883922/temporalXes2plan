"""Unit tests for network_search.search_space.suggest_config —
docs/network_improvement_loop_plan.md §9 phase 3: every generated config
must always satisfy the two correlation constraints from
docs/network_improvement_loop.md §4.9, over a large random sample of trials.
"""
import optuna
import pytest

from models import AnalysisConfig
from network_search.search_space import (
    _DT_PRUNE_MIN_LEAF_SAMPLES_CHOICES,
    _EFFECT_MODE_CHOICES,
    suggest_config,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 500


def _generate_configs(n: int = N_TRIALS, seed: int = 0):
    configs = []

    def objective(trial: optuna.Trial) -> float:
        configs.append(suggest_config(trial))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    study.optimize(objective, n_trials=n)
    return configs


@pytest.fixture(scope="module")
def configs():
    return _generate_configs()


# ---------------------------------------------------------------------------
# Correlation constraints (§4.9) — must hold for every trial, no exceptions
# ---------------------------------------------------------------------------

class TestCorrelationConstraints:
    def test_dt_min_samples_always_at_least_probability_min_samples(self, configs):
        assert all(cfg.dt_min_samples >= cfg.probability_min_samples for cfg in configs)

    def test_dt_min_prob_to_use_always_at_least_xor_prune_threshold(self, configs):
        # Both are floats built from the same suggest_float grid -- an exact
        # comparison is safe here (no independent floating point ops mix in),
        # but allow a tiny epsilon for robustness against rounding.
        assert all(
            cfg.dt_min_prob_to_use >= cfg.xor_prune_threshold - 1e-9 for cfg in configs
        )

    def test_constraints_are_not_trivially_satisfied_by_a_constant(self, configs):
        # Guard against a degenerate implementation that always returns the
        # same dt_min_samples/probability_min_samples pair -- the search
        # space must actually vary both sides of each constraint.
        assert len({cfg.probability_min_samples for cfg in configs}) > 1
        assert len({cfg.dt_min_samples for cfg in configs}) > 1
        assert len({cfg.xor_prune_threshold for cfg in configs}) > 1
        assert len({cfg.dt_min_prob_to_use for cfg in configs}) > 1


# ---------------------------------------------------------------------------
# Every suggested value stays within its declared boundary (§4.1-§4.6)
# ---------------------------------------------------------------------------

class TestBoundaries:
    def test_probability_min_samples_in_range(self, configs):
        assert all(5 <= cfg.probability_min_samples <= 50 for cfg in configs)

    def test_static_appearances_min_samples_in_range(self, configs):
        assert all(10 <= cfg.static_appearances_min_samples <= 100 for cfg in configs)

    def test_static_attr_probability_in_range(self, configs):
        # Upper bound snapped to 0.97 -- see search_space.py's comment.
        assert all(0.85 <= cfg.static_attr_probability <= 0.97 for cfg in configs)

    def test_xor_prune_threshold_in_range(self, configs):
        assert all(0.01 <= cfg.xor_prune_threshold <= 0.15 for cfg in configs)

    def test_dt_min_accuracy_in_range(self, configs):
        assert all(0.6 <= cfg.dt_min_accuracy <= 0.95 for cfg in configs)

    def test_dt_max_depth_in_range(self, configs):
        assert all(1 <= cfg.dt_max_depth <= 5 for cfg in configs)

    def test_dt_prune_min_purity_in_range(self, configs):
        assert all(0.6 <= cfg.dt_prune_min_purity <= 0.95 for cfg in configs)

    def test_dt_prune_min_leaf_samples_from_explicit_choice_list(self, configs):
        assert all(
            cfg.dt_prune_min_leaf_samples in _DT_PRUNE_MIN_LEAF_SAMPLES_CHOICES for cfg in configs
        )

    def test_effect_never_threshold_in_range(self, configs):
        assert all(0.01 <= cfg.effect_never_threshold <= 0.15 for cfg in configs)

    def test_effect_always_threshold_in_range(self, configs):
        assert all(0.85 <= cfg.effect_always_threshold <= 0.97 for cfg in configs)

    def test_effect_value_prune_threshold_in_range(self, configs):
        assert all(0.01 <= cfg.effect_value_prune_threshold <= 0.16 for cfg in configs)

    def test_effect_value_certain_threshold_in_range(self, configs):
        assert all(0.85 <= cfg.effect_value_certain_threshold <= 0.97 for cfg in configs)

    def test_related_effect_prob_in_range(self, configs):
        assert all(0.75 <= cfg.related_effect_prob <= 0.95 for cfg in configs)

    def test_incompatible_effect_prob_in_range(self, configs):
        assert all(0.01 <= cfg.incompatible_effect_prob <= 0.11 for cfg in configs)

    def test_replay_min_fitness_in_range(self, configs):
        assert all(0.5 <= cfg.replay_min_fitness <= 0.95 for cfg in configs)

    def test_kmeans_max_k_in_range(self, configs):
        assert all(3 <= cfg.kmeans_max_k <= 8 for cfg in configs)

    def test_dominance_threshold_in_range(self, configs):
        assert all(0.15 <= cfg.dominance_threshold <= 0.50 for cfg in configs)

    def test_min_residual_points_in_range(self, configs):
        assert all(20 <= cfg.min_residual_points <= 100 for cfg in configs)

    def test_min_gvf_threshold_in_range(self, configs):
        assert all(0.50 <= cfg.min_gvf_threshold <= 0.85 for cfg in configs)

    def test_gvf_target_in_range(self, configs):
        assert all(0.80 <= cfg.gvf_target <= 0.98 for cfg in configs)

    def test_min_gvf_improvement_in_range(self, configs):
        assert all(0.005 <= cfg.min_gvf_improvement <= 0.03 for cfg in configs)


# ---------------------------------------------------------------------------
# Categorical fields
# ---------------------------------------------------------------------------

class TestCategoricalFields:
    def test_dt_prune_orphan_mode_valid(self, configs):
        assert all(cfg.dt_prune_orphan_mode in ("fallback", "keep_best", "drop") for cfg in configs)

    def test_xor_statistical_mode_valid(self, configs):
        assert all(
            cfg.xor_statistical_mode in ("majority_only", "weighted", "pruned_weighted")
            for cfg in configs
        )

    def test_effect_appearance_mode_valid(self, configs):
        assert all(cfg.effect_appearance_mode in _EFFECT_MODE_CHOICES for cfg in configs)

    def test_effect_value_mode_valid(self, configs):
        assert all(cfg.effect_value_mode in _EFFECT_MODE_CHOICES for cfg in configs)

    def test_all_categorical_choices_are_actually_reachable(self, configs):
        # Sanity: with 500 random trials every declared categorical option
        # should show up at least once, otherwise the space is narrower than
        # intended.
        assert {cfg.dt_prune_orphan_mode for cfg in configs} == {"fallback", "keep_best", "drop"}
        assert {cfg.xor_statistical_mode for cfg in configs} == {
            "majority_only", "weighted", "pruned_weighted",
        }
        assert set(_EFFECT_MODE_CHOICES) <= {cfg.effect_appearance_mode for cfg in configs}
        assert set(_EFFECT_MODE_CHOICES) <= {cfg.effect_value_mode for cfg in configs}


# ---------------------------------------------------------------------------
# Fields excluded from the search space stay at their AnalysisConfig() default
# ---------------------------------------------------------------------------

class TestExcludedFieldsStayAtDefault:
    def test_pre_discretization_performance_knobs_untouched(self, configs):
        # jenks_sample_size/kde_grid_points/kde_extrema_order stay fixed --
        # unlike the rest of §4.7, these are performance/resolution knobs,
        # not quality knobs (see the analysis doc).
        default = AnalysisConfig()
        for cfg in configs:
            assert cfg.jenks_sample_size == default.jenks_sample_size
            assert cfg.kde_grid_points == default.kde_grid_points
            assert cfg.kde_extrema_order == default.kde_extrema_order

    def test_replay_engine_and_variant_untouched(self, configs):
        default = AnalysisConfig()
        assert all(cfg.replay_engine == default.replay_engine for cfg in configs)
        assert all(cfg.replay_alignment_variant == default.replay_alignment_variant for cfg in configs)

    def test_dead_and_fixed_fields_untouched(self, configs):
        default = AnalysisConfig()
        for cfg in configs:
            assert cfg.attr_precondition_min_frequency == default.attr_precondition_min_frequency
            assert cfg.attr_precondition_min_firings == default.attr_precondition_min_firings
            assert cfg.xor_screen_prune_branches == default.xor_screen_prune_branches
            assert cfg.lower_bound_prob_actions == default.lower_bound_prob_actions
            assert cfg.ignored_attributes == default.ignored_attributes
            assert cfg.log_removed_effects == default.log_removed_effects
            assert cfg.snapshot_dir == default.snapshot_dir
