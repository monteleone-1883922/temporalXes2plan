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


def compute_fallback_score(
    xor_screening: Dict[str, XorSplitScreening],
    effect_screening: Dict[str, TransitionScreening],
    weights: ScoreWeights,
) -> float:
    """Asse A (routing XOR) + Asse B (effetti) — docs/network_improvement_loop.md §3.2.

    Returns 0.0 (neutral) when there are no decisions to classify at all
    (e.g. a net with no XOR splits and no conditional effects) rather than
    dividing by zero.
    """
    total_weight = 0.0
    n_decisions = 0

    for split_screening in xor_screening.values():
        for branch in split_screening.branches.values():
            if branch.status == "pruned":
                total_weight -= weights.w_prune_xor
            elif branch.status == "certain":
                total_weight += weights.w_det
            elif branch.status == "active":
                if split_screening.action == "fallback":
                    #todo change this kind of fallback, its weight should be less than normal fallback
                    total_weight -= weights.w_fb_xor
                # action == "dt" (or, defensively, any other action while
                # status is "active"): neutral — see module docstring's
                # first caveat for the DT-guard-missing edge case this
                # simplification does not distinguish.
            elif branch.status == "fallback":
                total_weight -= weights.w_fb_xor
            else:
                raise ValueError(f"Unexpected XorBranchScreening.status: {branch.status!r}")
            n_decisions += 1

    for transition_screening in effect_screening.values():
        for attr_screening in transition_screening.attributes.values():
            for action in (attr_screening.appearance_action, attr_screening.value_action):
                #todo what about dt leaves?
                if action == _EFFECT_NEVER:
                    continue
                if action == "deterministic":
                    total_weight += weights.w_det
                elif action == "dt":
                    pass  # neutral
                elif action == "fallback":
                    total_weight -= weights.w_fb_eff
                else:
                    raise ValueError(f"Unexpected effect action: {action!r}")
                n_decisions += 1

    if n_decisions == 0:
        return 0.0
    return total_weight / n_decisions


def compute_duplication_penalty(transitions: Dict[str, PreparedTransition]) -> float:
    """docs/network_improvement_loop.md §3.3 — mean excess effect-group count.

    Takes PreparedDomainInput.transitions (PreparedTransition, built by
    DomainBuilder), not ParseResult.transitions (TransitionInfo) — the
    latter's own `effect_groups` field is never populated by Parser (always
    `[]`, see xes_parser.py's TransitionInfo construction); the actual
    Cartesian-combined effect groups used to generate PDDL variant actions
    only exist on PreparedTransition, produced later by
    DomainBuilder.build_prepared_input().

    max(0, len(effect_groups) - 1) per transition (a single effect group is
    not a duplication — it is the normal case), averaged over all
    transitions so the penalty is comparable across nets of different
    sizes rather than growing with the number of activities.
    """
    if not transitions:
        return 0.0
    excess = sum(max(0, len(t.effect_groups) - 1) for t in transitions.values())
    return excess / len(transitions)


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
