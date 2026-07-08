"""Score composition and the exact early-stopping bound for the replay phase.

See docs/network_improvement_loop.md §3 for the metric design and
docs/network_improvement_loop_plan.md §6 for this module's role in the
search loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreWeights:
    """Weights for combine() — placeholder starting values, not tuned yet.

    docs/network_improvement_loop.md §8 point 1 defers picking final
    numeric values to experimentation once the loop runs end-to-end on a
    real log. w_repro must stay > 0 for combine() to be monotone
    increasing in reproducibility_score — see max_reachable_score()'s
    docstring for why that property is required.
    """

    w_det: float = 1.0
    w_fb_xor: float = 1.0
    w_fb_eff: float = 1.0
    w_prune_xor: float = 3.0
    w_dup: float = 0.2
    w_repro: float = 5.0


@dataclass
class TrialMetrics:
    """The metrics computed for one trial (candidate AnalysisConfig).

    fallback_score and duplication_penalty are known already at rung 1
    (screening only, no replay) — see docs/network_improvement_loop_plan.md
    §7. reproducibility_score is None until rung 2 actually runs.
    """

    fallback_score: float
    duplication_penalty: float
    reproducibility_score: Optional[float]
    n_test_replayed: int
    coverage: float


def combine(metrics: TrialMetrics, weights: ScoreWeights) -> float:
    """Final trial score — a weighted sum of the three axes.

    Must stay monotone increasing in metrics.reproducibility_score
    (weights.w_repro > 0) for max_reachable_score()'s exact bound to be
    valid — see that function's docstring.
    """
    return (
        weights.w_repro * (metrics.reproducibility_score or 0.0)
        + metrics.fallback_score
        - weights.w_dup * metrics.duplication_penalty
    )


def max_reachable_score(
    n_success_so_far: int,
    n_processed_so_far: int,
    n_test_total: int,
    fallback_score: float,
    duplication_penalty: float,
    weights: ScoreWeights,
) -> float:
    """Exact upper bound on the final score reachable from this point in a
    trial's replay loop — docs/network_improvement_loop.md §5.4.2.

    fallback_score and duplication_penalty are already fixed by rung 1 (they
    do not depend on the replay); the only thing still unknown is how many
    of the remaining test traces will replay successfully. The best case is
    all of them do, giving the maximum possible reproducibility_score used
    here. Comparing this bound against the best score found so far tells
    the caller whether continuing the replay for this trial could still
    change the outcome.

    This is an exact bound, not a statistical estimate: it can never reject
    a trial that would go on to beat the current best, unlike an
    extrapolating pruner (e.g. Optuna's MedianPruner). That guarantee only
    holds because combine() is monotone increasing in
    reproducibility_score (weights.w_repro > 0) — if combine() ever stops
    being monotone in that term, this bound must be revisited.
    """
    n_remaining = n_test_total - n_processed_so_far
    best_possible_reproducibility = (n_success_so_far + n_remaining) / n_test_total
    return combine(
        TrialMetrics(
            fallback_score=fallback_score,
            duplication_penalty=duplication_penalty,
            reproducibility_score=best_possible_reproducibility,
            n_test_replayed=n_processed_so_far,
            coverage=0.0,
        ),
        weights,
    )
