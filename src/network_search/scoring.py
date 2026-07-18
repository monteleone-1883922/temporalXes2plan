"""Score composition and the exact early-stopping bound for the replay phase.

See docs/network_improvement_loop.md §3 for the metric design and
docs/network_improvement_loop_plan.md §6 for this module's role in the
search loop. See docs/network_search_score_formula.md for the full
mathematical formula this module implements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreWeights:
    """Weights for combine() — placeholder starting values, not tuned yet.

    docs/network_improvement_loop.md §8 point 1 defers picking final
    numeric values to experimentation once the loop runs end-to-end on a
    real log. reproducibility_score is added to the final score with an
    implicit coefficient of 1 (no weight of its own — every other
    component already has one, so this one is deliberately left
    unweighted) and combine() is therefore always monotone increasing in
    it, unconditionally — see max_reachable_score()'s docstring for why
    that property is required.

    w_det_xor/w_det_eff/w_fb_xor/w_fb_eff/w_prune_xor **must be values in
    [0, 1]** — they parametrize compute_xor_score()/compute_effect_score()'s
    per-decision reward/penalty (0.5 +/- 0.5*weight), which is what keeps
    xor_score and effect_score themselves bounded in [0, 1] with 0.5 as
    neutral. w_det_xor/w_det_eff are separate weights for the deterministic
    reward on each axis (XOR routing vs. effect appearance/value) — they
    used to be a single shared w_det. w_xor, w_eff and w_dup instead scale
    those already-bounded [0, 1] scores inside the *final* score (see
    combine()) and are ordinary unrestricted weights, same role as w_dup
    already had before this axis split.

    w_full_replay scales full_replayability_score the same way w_xor/w_eff
    scale their own already-[0, 1]-bounded scores in combine(). Unlike
    reproducibility_score (implicit coefficient 1, unconditional),
    full_replayability_score's weight is explicit and **must stay >= 0**:
    max_reachable_score()'s exact-upper-bound guarantee depends on
    combine() being monotone increasing in full_replayability_score, which
    only holds for a non-negative weight — same documented-not-validated
    convention as the [0, 1]-constrained weights above.
    """

    w_det_xor: float = 1.0
    w_det_eff: float = 1.0
    w_fb_xor: float = 0.5
    w_fb_eff: float = 0.5
    w_prune_xor: float = 1.0
    w_xor: float = 1.0
    w_eff: float = 1.0
    w_dup: float = 0.2
    w_full_replay: float = 1.0


@dataclass
class TrialMetrics:
    """The metrics computed for one trial (candidate AnalysisConfig).

    xor_score, effect_score and duplication_score are known already at
    rung 1 (screening only, no replay) — see
    docs/network_improvement_loop_plan.md §7. reproducibility_score is None
    until rung 2 actually runs.

    full_replayability_score is likewise None until rung 2 — a stricter
    companion to reproducibility_score: it additionally requires the
    replayed trace's final attribute assignment to exactly match the
    log-observed final state ("full replayability", see
    TraceReplayer.replay_evaluation_split's matches_expected_final_state).
    """

    xor_score: float
    effect_score: float
    duplication_score: float
    reproducibility_score: Optional[float]
    n_test_replayed: int
    coverage: float
    full_replayability_score: Optional[float] = None


def combine(metrics: TrialMetrics, weights: ScoreWeights) -> float:
    """Final trial score — a weighted sum of the components.

    reproducibility_score has no weight of its own (implicit coefficient
    1), so combine() is unconditionally monotone increasing in it.
    full_replayability_score DOES have its own weight (w_full_replay,
    default 1.0) — combine() is monotone increasing in it as long as
    w_full_replay >= 0. See max_reachable_score()'s docstring for why both
    properties are required.
    """
    return (
        (metrics.reproducibility_score or 0.0)
        + weights.w_xor * metrics.xor_score
        + weights.w_eff * metrics.effect_score
        + weights.w_dup * metrics.duplication_score
        + weights.w_full_replay * (metrics.full_replayability_score or 0.0)
    )


def max_reachable_score(
    n_success_so_far: int,
    n_full_success_so_far: int,
    n_processed_so_far: int,
    n_test_total: int,
    xor_score: float,
    effect_score: float,
    duplication_score: float,
    weights: ScoreWeights,
) -> float:
    """Exact upper bound on the final score reachable from this point in a
    trial's replay loop — docs/network_improvement_loop.md §5.4.2.

    xor_score, effect_score and duplication_score are already fixed by
    rung 1 (they do not depend on the replay); the only thing still
    unknown is how many of the remaining test traces will replay
    successfully, in each of two senses: reproducibility_score's loose
    "reached_end" and full_replayability_score's stricter "reached_end AND
    matches_expected_final_state" (n_full_success_so_far <= n_success_so_far
    always, since full success requires loose success too). The best case
    for each is that every remaining trace succeeds, giving the maximum
    possible value of that score. Comparing this bound against the best
    score found so far tells the caller whether continuing the replay for
    this trial could still change the outcome.

    This is an exact bound, not a statistical estimate: it can never reject
    a trial that would go on to beat the current best, unlike an
    extrapolating pruner (e.g. Optuna's MedianPruner). That guarantee holds
    because combine() is linear in (reproducibility_score,
    full_replayability_score) for fixed xor/effect/duplication scores, and
    each term's own best-case value is *simultaneously* achievable by the
    same continuation ("every remaining trace both reaches the end and
    matches the expected final state" maximizes both counters at once,
    precisely because full success implies loose success) — so summing the
    two independently-optimistic terms yields a genuine, tight upper bound,
    not merely two separate overestimates. reproducibility_score's term is
    unconditionally monotone (implicit coefficient 1 in combine()).
    full_replayability_score's term is monotone only when weights.w_full_replay
    >= 0 — if combine() ever stops adding either term with a fixed
    non-negative coefficient, this bound must be revisited.
    """
    n_remaining = n_test_total - n_processed_so_far
    best_possible_reproducibility = (n_success_so_far + n_remaining) / n_test_total
    best_possible_full_replayability = (n_full_success_so_far + n_remaining) / n_test_total
    return combine(
        TrialMetrics(
            xor_score=xor_score,
            effect_score=effect_score,
            duplication_score=duplication_score,
            reproducibility_score=best_possible_reproducibility,
            n_test_replayed=n_processed_so_far,
            coverage=0.0,
            full_replayability_score=best_possible_full_replayability,
        ),
        weights,
    )
