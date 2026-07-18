"""Pure metric functions for the network-quality search loop.

No I/O, no dependency on Optuna — every function takes already-built
in-memory objects and returns a float. See docs/network_improvement_loop.md
§3 for the design these implement.

Source of truth for the fallback/determinism classification is the
pre-pruning screening (XorSplitScreening/TransitionScreening), not the
final ParseResult — see §3.2 of that document for why. Note two caveats
found while implementing, worth keeping in mind if this ever needs to match
the constructed network exactly rather than the screening decision:

- `_filter_xor_splits` in xes_parser.py never actually calls
  `prune_transitions()` on the branches it collects into its `pruned` set
  (the call site has a `# TODO remove pruned` and is not implemented) — a
  branch screened as "pruned" is *classified* as eliminated by this module,
  but is not currently removed from the discovered Petri net. If that TODO
  gets implemented, this classification remains correct; if it never does,
  this metric is still measuring the right thing (the screening intended
  the branch to disappear), just ahead of the network construction code
  actually enforcing it.
- Under `xor_statistical_mode == "majority_only"`, `_apply_fallback` only
  ever assigns cascade info to the single highest-probability active
  branch — the other active (non-pruned, non-winning) branches receive no
  `XorBranchInfo` at all, but are also not added to the `pruned` set. This
  module classifies them by their screening `status` ("active"), i.e. as
  neutral, not as eliminated or fallback — a branch-status-only
  classification cannot detect this specific case without cross-referencing
  the constructed `ParseResult`, which this module deliberately avoids
  (see module docstring above).
"""

from __future__ import annotations

from typing import Dict, List, Set

from encoding.prepared_input import PreparedTransition
from models import ParseResult, TransitionScreening, XorSplitScreening
from replay.trace_replayer import ReplayOutcome

from network_search.scoring import ScoreWeights

# EffectAttrScreening.appearance_action / value_action and
# XorBranchScreening.status values that are excluded from the score
# entirely (not a routing/effect decision made about the network).
_EFFECT_NEVER = "never"


def _bounded_reward(weight: float) -> float:
    """0.5 + 0.5*weight -- maps a reward weight in [0,1] to [0.5, 1]."""
    return 0.5 + 0.5 * weight


def _bounded_penalty(weight: float) -> float:
    """0.5 - 0.5*weight -- maps a penalty weight in [0,1] to [0, 0.5]."""
    return 0.5 - 0.5 * weight


def compute_xor_score(
    xor_screening: Dict[str, XorSplitScreening],
    weights: ScoreWeights,
) -> float:
    """Asse A (routing XOR) — docs/network_improvement_loop.md §3.2.

    Every decision contributes a value in [0, 1], with 0.5 as the neutral
    point (0 = worst possible, 1 = best possible) — see
    docs/network_search_score_formula.md.
    weights.w_det_xor/w_fb_xor/w_prune_xor must themselves be in [0, 1]: a
    reward contributes 0.5 + 0.5*w_det_xor, a penalty contributes
    0.5 - 0.5*w_prune_xor (or w_fb_xor), so the average over all decisions
    always stays in [0, 1].

    Returns 0.5 (neutral) when there are no XOR splits to classify at all,
    rather than dividing by zero.
    """
    total = 0.0
    n_decisions = 0

    for split_screening in xor_screening.values():
        for branch in split_screening.branches.values():
            if branch.status == "pruned":
                value = _bounded_penalty(weights.w_prune_xor)
            elif branch.status == "certain":
                value = _bounded_reward(weights.w_det_xor)
            elif branch.status == "active":
                if split_screening.action == "fallback":
                    #todo change this kind of fallback, its weight should be less than normal fallback
                    value = _bounded_penalty(weights.w_fb_xor)
                else:
                    # action == "dt" (or, defensively, any other action while
                    # status is "active"): fully rewarded (1.0, the best
                    # possible score) — the routing decision is explained by
                    # a mined guard, not merely "not eliminated" — see module
                    # docstring's first caveat for the DT-guard-missing edge
                    # case this simplification does not distinguish.
                    value = 1
            elif branch.status == "fallback":
                value = _bounded_penalty(weights.w_fb_xor)
            else:
                raise ValueError(f"Unexpected XorBranchScreening.status: {branch.status!r}")
            total += value
            n_decisions += 1

    if n_decisions == 0:
        return 0.5
    return total / n_decisions


def compute_effect_score(
    effect_screening: Dict[str, TransitionScreening],
    weights: ScoreWeights,
) -> float:
    """Asse B (effetti) — docs/network_improvement_loop.md §3.2.

    Every decision contributes a value in [0, 1], with 0.5 as the neutral
    point, exactly like compute_xor_score — weights.w_det_eff/w_fb_eff must
    themselves be in [0, 1].

    Returns 0.5 (neutral) when there are no conditional effects to classify
    at all, rather than dividing by zero.
    """
    total = 0.0
    n_decisions = 0

    for transition_screening in effect_screening.values():
        for attr_screening in transition_screening.attributes.values():
            for action in (attr_screening.appearance_action, attr_screening.value_action):
                #todo what about dt leaves?
                if action == _EFFECT_NEVER:
                    continue
                if action == "deterministic":
                    value = _bounded_reward(weights.w_det_eff)
                elif action == "dt":
                    value = 0.5  # neutral
                elif action == "fallback":
                    value = _bounded_penalty(weights.w_fb_eff)
                else:
                    raise ValueError(f"Unexpected effect action: {action!r}")
                total += value
                n_decisions += 1

    if n_decisions == 0:
        return 0.5
    return total / n_decisions


def compute_duplication_score(transitions: Dict[str, PreparedTransition]) -> float:
    """docs/network_improvement_loop.md §3.3 — inverse mean effect-group count.

    Takes PreparedDomainInput.transitions (PreparedTransition, built by
    DomainBuilder), not ParseResult.transitions (TransitionInfo) — the
    latter's own `effect_groups` field is never populated by Parser (always
    `[]`, see xes_parser.py's TransitionInfo construction); the actual
    Cartesian-combined effect groups used to generate PDDL variant actions
    only exist on PreparedTransition, produced later by
    DomainBuilder.build_prepared_input().

    len(transitions) / max(1, total_groups), where total_groups is the sum
    over all transitions of len(effect_groups). This is a *reward*, not a
    penalty: fewer effect groups relative to the size of the net yields a
    higher score. Every real transition has at least one effect group, so
    total_groups is normally >= len(transitions), keeping the result in
    (0, 1] — max(1, total_groups) only guards the degenerate case where no
    transition has any effect group at all (total_groups == 0), which would
    otherwise divide by zero.
    """
    if not transitions:
        return 0.0
    total_groups = sum(len(t.effect_groups) for t in transitions.values())
    return len(transitions) / max(1, total_groups)


def compute_reproducibility(replay_outcomes: List[ReplayOutcome]) -> float:
    """docs/network_improvement_loop.md §3.1 — fraction of replayable test traces.

    Returns 0.0 for an empty list rather than raising — an empty test set
    is a configuration problem for the caller to catch, not something this
    pure function should decide how to handle.
    """
    if not replay_outcomes:
        return 0.0
    return sum(1 for outcome in replay_outcomes if outcome.is_replayable) / len(replay_outcomes)


def compute_coverage(parse_result: ParseResult, original_activity_names: Set[str]) -> float:
    """docs/network_improvement_loop.md §3.4 — reporting-only, not part of the score.

    Fraction of the original log's activities that survive as labeled
    transitions in the final (pruned) ParseResult.
    """
    if not original_activity_names:
        return 1.0
    present = set(parse_result.transitions.keys())
    return len(present & original_activity_names) / len(original_activity_names)
