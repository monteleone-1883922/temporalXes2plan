"""Search space definition and per-trial AnalysisConfig construction.

See docs/network_improvement_loop.md §4 for the full analysis behind every
boundary/step chosen here — including why replay_engine/replay_alignment_variant
(§4.6) are excluded from this first version of the search space while the
pre-discretization parameters (§4.7) are included — and
docs/network_improvement_loop_plan.md §4 for this module's role in the loop.
"""

from __future__ import annotations

from typing import Optional

import optuna

from models import AnalysisConfig

# Explicit non-uniform step list for dt_prune_min_leaf_samples (§4.1) —
# Optuna's suggest_int only supports a uniform step, so the low-then-coarse
# progression proposed in the analysis is expressed as a categorical instead.
_DT_PRUNE_MIN_LEAF_SAMPLES_CHOICES = [1, 2, 3, 5, 8, 10, 15, 20]

# effect_appearance_mode/effect_value_mode: only 2 effective values today —
# see docs/network_improvement_loop.md §4 intro. "duplicate" stands in for
# every mode string other than "duplicate_no_cost" (all currently produce
# identical behavior in domain_builder.py).
_EFFECT_MODE_CHOICES = ["duplicate", "duplicate_no_cost"]


def suggest_config(trial: optuna.Trial, base_config: Optional[AnalysisConfig] = None) -> AnalysisConfig:
    """Build one candidate AnalysisConfig from an Optuna trial.

    Two pairs of parameters are reparametrized as base + non-negative offset
    (docs/network_improvement_loop.md §4.9, confirmed in §8) instead of
    being suggested independently, so every AnalysisConfig this function
    returns is guaranteed to satisfy both correlation constraints without
    ever needing to reject or clip a sampled combination:

    - dt_min_samples = probability_min_samples + dt_min_samples_extra
    - dt_min_prob_to_use = xor_prune_threshold + dt_min_prob_extra

    TPE observes and builds its surrogate model on the *_extra parameters,
    not on dt_min_samples/dt_min_prob_to_use directly — an accepted
    consequence of the reparametrization (see the analysis doc).

    Every AnalysisConfig field not set explicitly below
    (replay_engine/replay_alignment_variant, jenks_sample_size/kde_grid_points/
    kde_extrema_order, the dead
    attr_precondition_*/effect_appearance_threshold/xor_screen_prune_branches
    fields, and fixed/logging/path fields) is sourced from `base_config`
    (or from `AnalysisConfig()`'s own default when `base_config` is None) —
    this lets a caller (`Pipeline.run(search=True, config=...)`) supply
    user-chosen values for fields Optuna doesn't tune.
    """
    # --- §4.2 fallback statistico e attributi "statici" ---
    # probability_min_samples is also the base for the dt_min_samples
    # reparametrization below.
    probability_min_samples = trial.suggest_int("probability_min_samples", 5, 50, step=5)
    static_appearances_min_samples = trial.suggest_int(
        "static_appearances_min_samples", 10, 100, step=10
    )
    # Upper bound snapped to 0.97 (nearest value reachable from 0.85 at step
    # 0.03) — see the note above §4.4 for why.
    static_attr_probability = trial.suggest_float("static_attr_probability", 0.85, 0.97, step=0.03)

    # --- §4.3 XOR-specifici ---
    # xor_prune_threshold is also the base for the dt_min_prob_to_use
    # reparametrization below.
    xor_statistical_mode = trial.suggest_categorical(
        "xor_statistical_mode", ["majority_only", "weighted", "pruned_weighted"]
    )
    xor_prune_threshold = trial.suggest_float("xor_prune_threshold", 0.01, 0.15, step=0.02)

    # --- §4.1 cascade decision tree (condiviso XOR + effetti) ---
    dt_min_samples_extra = trial.suggest_int("dt_min_samples_extra", 0, 70, step=10)
    dt_min_samples = probability_min_samples + dt_min_samples_extra

    dt_min_prob_extra = trial.suggest_float("dt_min_prob_extra", 0.0, 0.15, step=0.05)
    dt_min_prob_to_use = xor_prune_threshold + dt_min_prob_extra

    dt_min_accuracy = trial.suggest_float("dt_min_accuracy", 0.6, 0.95, step=0.05)
    dt_max_depth = trial.suggest_int("dt_max_depth", 1, 5, step=1)
    dt_prune_min_leaf_samples = trial.suggest_categorical(
        "dt_prune_min_leaf_samples", _DT_PRUNE_MIN_LEAF_SAMPLES_CHOICES
    )
    dt_prune_min_purity = trial.suggest_float("dt_prune_min_purity", 0.6, 0.95, step=0.05)
    dt_prune_orphan_mode = trial.suggest_categorical(
        "dt_prune_orphan_mode", ["fallback", "keep_best", "drop"]
    )

    # --- §4.4 effects conditional ---
    # Upper bounds below are snapped to the nearest value reachable from the
    # lower bound at the given step (Optuna requires the range to be evenly
    # divisible by step; passing the analysis doc's nominal bound as-is
    # would only produce a UserWarning and silently apply the same snap) —
    # 0.99 -> 0.97 at step 0.03, 0.20 -> 0.16 at step 0.05.
    effect_never_threshold = trial.suggest_float("effect_never_threshold", 0.01, 0.15, step=0.02)
    effect_always_threshold = trial.suggest_float("effect_always_threshold", 0.85, 0.97, step=0.03)
    effect_value_prune_threshold = trial.suggest_float(
        "effect_value_prune_threshold", 0.01, 0.16, step=0.05
    )
    effect_value_certain_threshold = trial.suggest_float(
        "effect_value_certain_threshold", 0.85, 0.97, step=0.03
    )
    effect_appearance_mode = trial.suggest_categorical(
        "effect_appearance_mode", _EFFECT_MODE_CHOICES
    )
    effect_value_mode = trial.suggest_categorical("effect_value_mode", _EFFECT_MODE_CHOICES)

    # --- §4.5 co-occorrenza effetti ---
    # Same snapping as above: 0.99 -> 0.95 at step 0.05, 0.15 -> 0.11 at step 0.05.
    related_effect_prob = trial.suggest_float("related_effect_prob", 0.75, 0.95, step=0.05)
    incompatible_effect_prob = trial.suggest_float("incompatible_effect_prob", 0.01, 0.11, step=0.05)

    # --- §4.6 token replay (replay_engine/replay_alignment_variant excluded) ---
    replay_min_fitness = trial.suggest_float("replay_min_fitness", 0.5, 0.95, step=0.05)

    # --- §4.7 pre-discretizzazione numerica ---
    # jenks_sample_size/kde_grid_points/kde_extrema_order stay fixed at their
    # AnalysisConfig() default -- performance/numerical-resolution knobs, not
    # quality knobs (see the analysis doc).
    kmeans_max_k = trial.suggest_int("kmeans_max_k", 3, 8, step=1)
    dominance_threshold = trial.suggest_float("dominance_threshold", 0.15, 0.50, step=0.05)
    min_residual_points = trial.suggest_int("min_residual_points", 20, 100, step=10)
    min_gvf_threshold = trial.suggest_float("min_gvf_threshold", 0.50, 0.85, step=0.05)
    gvf_target = trial.suggest_float("gvf_target", 0.80, 0.98, step=0.02)
    min_gvf_improvement = trial.suggest_float("min_gvf_improvement", 0.005, 0.03, step=0.005)

    base = base_config if base_config is not None else AnalysisConfig()

    return AnalysisConfig(
        dt_min_samples=dt_min_samples,
        dt_min_prob_to_use=dt_min_prob_to_use,
        dt_min_accuracy=dt_min_accuracy,
        dt_max_depth=dt_max_depth,
        probability_min_samples=probability_min_samples,
        static_appearances_min_samples=static_appearances_min_samples,
        static_attr_probability=static_attr_probability,
        dt_prune_min_leaf_samples=dt_prune_min_leaf_samples,
        dt_prune_min_purity=dt_prune_min_purity,
        dt_prune_orphan_mode=dt_prune_orphan_mode,
        xor_statistical_mode=xor_statistical_mode,
        xor_prune_threshold=xor_prune_threshold,
        effect_never_threshold=effect_never_threshold,
        effect_always_threshold=effect_always_threshold,
        effect_value_prune_threshold=effect_value_prune_threshold,
        effect_value_certain_threshold=effect_value_certain_threshold,
        related_effect_prob=related_effect_prob,
        incompatible_effect_prob=incompatible_effect_prob,
        effect_appearance_mode=effect_appearance_mode,
        effect_value_mode=effect_value_mode,
        replay_min_fitness=replay_min_fitness,
        kmeans_max_k=kmeans_max_k,
        dominance_threshold=dominance_threshold,
        min_residual_points=min_residual_points,
        min_gvf_threshold=min_gvf_threshold,
        gvf_target=gvf_target,
        min_gvf_improvement=min_gvf_improvement,
        # --- fields Optuna doesn't tune: sourced from base_config, not a
        # hardcoded default (docs/network_improvement_loop.md §4.6/§4.7) ---
        jenks_sample_size=base.jenks_sample_size,
        kde_grid_points=base.kde_grid_points,
        kde_extrema_order=base.kde_extrema_order,
        xor_screen_prune_branches=base.xor_screen_prune_branches,
        lower_bound_prob_actions=base.lower_bound_prob_actions,
        replay_engine=base.replay_engine,
        replay_alignment_variant=base.replay_alignment_variant,
        ignored_attributes=base.ignored_attributes,
        attr_precondition_min_frequency=base.attr_precondition_min_frequency,
        attr_precondition_min_firings=base.attr_precondition_min_firings,
        log_removed_effects=base.log_removed_effects,
        snapshot_dir=base.snapshot_dir,
    )
