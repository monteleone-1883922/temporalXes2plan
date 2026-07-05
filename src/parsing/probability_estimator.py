from collections import defaultdict
from typing import Any, Dict, List, Optional

from pm4py import PetriNet
from pm4py.objects.powl.obj import Transition

import core_utils as utils
from models import (
    AnalysisConfig,
    AttributeEffect,
    PreprocessedLog,
    XorSplitStats,
)

logger = utils.get_logger(__name__)


class ProbabilityEstimator:
    """Computes XOR-split branch and attribute-effect probabilities.

    Operates on a PreprocessedLog (produced by LogPreprocessor) instead of
    scanning the raw PetriNetLog.  All sanitization, discretization, and
    change-detection has already been performed during preprocessing — this
    class is a pure counter + normalizer.
    """

    def __init__(
        self,
        silent_transitions: Dict[Transition, str],
        config: Optional[AnalysisConfig] = None,
    ) -> None:
        self.silent_transitions = silent_transitions
        self.config = config or AnalysisConfig()

    # ------------------------------------------------------------------
    # XOR-split branch probabilities
    # ------------------------------------------------------------------

    def compute_xor_probabilities(
        self,
        preprocessed_log: PreprocessedLog,
        decision_points: Dict[PetriNet.Place, List[Transition]],
    ) -> Dict[str, XorSplitStats]:
        """Compute branch probabilities for XOR-split decision points.

        Counts branch activations from the preprocessed XOR-firings index
        rather than iterating the raw log.

        Args:
            preprocessed_log: Single-pass preprocessed log with xor_firings.
            decision_points: XOR-split places mapped to their outgoing
                transitions (from PetriNetModel.xor_splits).

        Returns:
            Dictionary mapping place names to XorSplitStats.
        """
        branch_counts: Dict[str, Dict[str, int]] = {}

        for place in decision_points:
            counts: Dict[str, int] = defaultdict(int)
            for fd in preprocessed_log.xor_firings.get(place.name, []):
                counts[fd.activity_name] += 1
            branch_counts[place.name] = dict(counts)

        return self._normalize_branch_counts(branch_counts, decision_points)

    def _normalize_branch_counts(
        self,
        branch_counts: Dict[str, Dict[str, int]],
        decision_points: Dict[PetriNet.Place, List[Transition]],
    ) -> Dict[str, XorSplitStats]:
        """Normalize branch activation counts into probabilities.

        Branches with zero total count fall back to equal probabilities.

        Args:
            branch_counts: place_name → activity_name → activation count.
            decision_points: XOR-split places mapped to outgoing transitions.

        Returns:
            Dictionary mapping place names to XorSplitStats.
        """
        result: Dict[str, XorSplitStats] = {}

        for place, transitions in decision_points.items():
            counts = branch_counts.get(place.name, {})
            total = sum(counts.values())

            if total > 0:
                place_probs = {
                    self._get_activity_name_for_transition(t): round(
                        counts.get(self._get_activity_name_for_transition(t), 0) / total, 2
                    )
                    for t in transitions
                }
            else:
                branch_names = [self._get_activity_name_for_transition(t) for t in transitions]
                logger.warning(
                    f"No replay data for XOR-split '{place.name}' "
                    f"(branches: {branch_names}). Assigning equal probabilities."
                )
                equal_prob = round(1.0 / len(transitions), 2) if transitions else 0.0
                place_probs = {name: equal_prob for name in branch_names}

            if place_probs:
                result[place.name] = XorSplitStats(
                    probabilities=place_probs,
                    total_executions=total,
                )

        return result

    # ------------------------------------------------------------------
    # Attribute-effect probabilities
    # ------------------------------------------------------------------

    def compute_attribute_effect_probabilities(
        self,
        preprocessed_log: PreprocessedLog,
    ) -> Dict[str, AttributeEffect]:
        """Compute attribute effect probabilities for each labeled transition.

        For each transition T visible in the preprocessed log:
        - P(attr X changes | T fires) = fraction of T firings where X is in
          changed_attrs.
        - P(X = v | X changes after T) = value distribution over changed_attrs
          values.

        All sanitization, discretization, and ignored-attribute filtering has
        already been applied by LogPreprocessor.

        Args:
            preprocessed_log: Single-pass preprocessed log with
                transition_firings.

        Returns:
            Dict mapping sanitized activity name to AttributeEffect.
        """
        result: Dict[str, AttributeEffect] = {}

        for act, firings in preprocessed_log.transition_firings.items():
            total = len(firings)
            effect_counts: Dict[str, int] = defaultdict(int)
            value_counts: Dict[str, Dict[Any, int]] = defaultdict(
                lambda: defaultdict(int)
            )

            for fd in firings:
                for attr, val in fd.changed_attrs.items():
                    # Ignore static attributes for effects
                    if attr in preprocessed_log.static_attributes:
                        continue
                    effect_counts[attr] += 1
                    val_key = val  # values are pre-sanitized by LogPreprocessor
                    value_counts[attr][val_key] += 1

            presence_probs: Dict[str, float] = {
                attr: round(count / total, 2)
                for attr, count in effect_counts.items()
            }
            value_probs: Dict[str, Dict[Any, float]] = {}
            for attr, count in effect_counts.items():
                attr_total = sum(value_counts[attr].values())
                value_probs[attr] = {
                    val: round(cnt / attr_total, 2)
                    for val, cnt in value_counts[attr].items()
                }

            result[act] = AttributeEffect(
                presence_probabilities=presence_probs,
                value_probabilities=value_probs,
                total_firings=total,
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_activity_name_for_transition(self, transition: Transition) -> str:
        """Resolve the activity name for a transition.

        Returns the sanitized label for labeled transitions, or the
        pre-assigned tau name for silent transitions.
        """
        if transition.label is not None:
            return transition.label
        fallback_name = f"tau_unknown_{id(transition)}"
        return self.silent_transitions.get(transition, fallback_name)
