from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from pm4py import PetriNet
from pm4py.objects.powl.obj import Transition

import core_utils as utils
from models import AttributeEffect, PetriNetLog, XorSplitStats
from parsing.discretizer import Discretizer

logger = utils.get_logger(__name__)


class ProbabilityEstimator:
    """
    Computes XOR-split branch probabilities from a PetriNetLog.

    Instead of running its own replay and marking simulation, this class reads
    the pre-built FiringStep.from_places to count branch activations: if a XOR-split
    place appears in a step's from_places, the step's transition is the chosen branch.

    This keeps the estimator as a pure counter + normalizer with no I/O concerns.
    Replay filtering and marking simulation are entirely handled by PetriNetLogBuilder.
    """

    def __init__(self, silent_transitions: Dict[Transition, str]) -> None:
        """
        Initialize ProbabilityEstimator.

        Args:
            silent_transitions: Mapping from Transition objects to their tau names,
                used to resolve activity names for silent transitions.
        """
        self.silent_transitions = silent_transitions

    def compute_from_petri_net_log(
        self,
        petri_net_log: PetriNetLog,
        decision_points: Dict[PetriNet.Place, List[Transition]],
    ) -> Dict[str, XorSplitStats]:
        """
        Compute branch probabilities for XOR-split decision points from a PetriNetLog.

        For each FiringStep, if any of its from_places is a XOR-split place, the
        step's transition is counted as a branch activation for that split.

        Args:
            petri_net_log: Pre-built PetriNetLog (replay already done and filtered).
            decision_points: XOR-split places mapped to their outgoing transitions,
                as returned by StructureAnalyzer.identify_xor_splits().

        Returns:
            Dictionary mapping place names to XorSplitStats (probabilities +
            total executions through that split).
        """
        branch_counts: Dict[PetriNet.Place, Dict[Transition, int]] = {
            place: defaultdict(int) for place in decision_points
        }

        for execution in petri_net_log.executions:
            for step in execution.steps:
                for place in step.from_places:
                    if place in decision_points:
                        branch_counts[place][step.transition] += 1

        return self._normalize_branch_counts(branch_counts, decision_points)

    def _normalize_branch_counts(
        self,
        branch_counts: Dict[PetriNet.Place, Dict[Transition, int]],
        decision_points: Dict[PetriNet.Place, List[Transition]],
    ) -> Dict[str, XorSplitStats]:
        """
        Normalize branch activation counts into probabilities and capture sample sizes.

        Branches with zero total count fall back to equal probabilities and a
        warning is logged.

        Args:
            branch_counts: Activation counts per place and transition.
            decision_points: XOR-split places mapped to their outgoing transitions.

        Returns:
            Dictionary mapping place names to XorSplitStats (probabilities + total
            number of executions through that split).
        """
        result: Dict[str, XorSplitStats] = {}

        for place, transitions in decision_points.items():
            counts = branch_counts[place]
            total = sum(counts.values())

            if total > 0:
                place_probs = {
                    self._get_activity_name_for_transition(t): round(counts[t] / total, 2)
                    for t in transitions
                }
            else:
                branch_names = [self._get_activity_name_for_transition(t) for t in transitions]
                logger.warning(
                    f"No replay data for XOR-split '{place.name}' "
                    f"(branches: {branch_names}). Assigning equal probabilities."
                )
                equal_prob = round(1.0 / len(transitions), 2) if transitions else 0.0
                place_probs = {
                    self._get_activity_name_for_transition(t): equal_prob
                    for t in transitions
                }

            if place_probs:
                result[place.name] = XorSplitStats(
                    probabilities=place_probs,
                    total_executions=total,
                )

        return result

    def compute_attribute_effect_probabilities(
        self,
        petri_net_log: PetriNetLog,
        discretizer: Optional[Discretizer] = None,
    ) -> Dict[str, AttributeEffect]:
        """
        Compute attribute effect probabilities for each non-tau transition.

        For each labeled transition T, determines:
        - P(attribute X changes after T): fraction of T firings where X changed value
          or appeared for the first time in the execution state.
        - P(X = v | X changes after T): value distribution conditioned on change.

        An attribute is considered changed if its value in the current step differs
        from the last seen value in the same execution, or if it appears for the
        first time. Attributes absent from a step's attributes are not considered.
        State resets at the start of each execution.

        Attribute names are sanitized via utils.sanitize_name. Numerical attribute
        values are discretized via the discretizer (if provided) before comparison
        and storage, so two raw values that fall in the same interval are treated
        as identical.

        Args:
            petri_net_log: Pre-built PetriNetLog.
            discretizer: Optional pre-fitted Discretizer for numerical attributes.
                When None, raw values are used as-is.

        Returns:
            Dict mapping activity_name to AttributeEffect. Tau transitions excluded.
        """
        effect_counts, value_counts, total_firings = self._count_attribute_effects(
            petri_net_log, discretizer
        )
        return self._normalize_effect_counts(effect_counts, value_counts, total_firings)

    def _count_attribute_effects(
        self,
        petri_net_log: PetriNetLog,
        discretizer: Optional[Discretizer] = None,
    ) -> Tuple[
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, Dict[Any, int]]],
        Dict[str, int],
    ]:
        """
        Accumulate raw effect counts across all executions.

        Attribute names are sanitized. Values are discretized when a discretizer
        is provided; otherwise raw values are used for comparison and storage.

        Args:
            petri_net_log: Pre-built PetriNetLog.
            discretizer: Optional pre-fitted Discretizer.

        Returns:
            Tuple of (effect_counts, value_counts, total_firings):
            - effect_counts:  activity → sanitized_attr → # firings where attribute changed
            - value_counts:   activity → sanitized_attr → value → # occurrences
            - total_firings:  activity → total # labeled firings
        """
        effect_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        value_counts: Dict[str, Dict[str, Dict[Any, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        total_firings: Dict[str, int] = defaultdict(int)

        for execution in petri_net_log.executions:
            state: Dict[str, Any] = {}
            for step in execution.steps:
                if step.is_tau:
                    continue
                act = step.activity_name
                total_firings[act] += 1
                for attr, val in step.attributes.items():
                    sanitized_attr = utils.sanitize_name(attr)
                    transformed_val = (
                        discretizer.transform_value(sanitized_attr, val)
                        if discretizer is not None
                        else val
                    )
                    if sanitized_attr not in state or state[sanitized_attr] != transformed_val:
                        effect_counts[act][sanitized_attr] += 1
                        value_counts[act][sanitized_attr][transformed_val] += 1
                    state[sanitized_attr] = transformed_val

        return effect_counts, value_counts, total_firings

    def _normalize_effect_counts(
        self,
        effect_counts: Dict[str, Dict[str, int]],
        value_counts: Dict[str, Dict[str, Dict[Any, int]]],
        total_firings: Dict[str, int],
    ) -> Dict[str, AttributeEffect]:
        """
        Normalize raw effect counts into probabilities.

        Args:
            effect_counts: Activity → attribute → # firings where attribute changed.
            value_counts:  Activity → attribute → value → # occurrences.
            total_firings: Activity → total labeled firings.

        Returns:
            Dict mapping activity_name to AttributeEffect.
        """
        result: Dict[str, AttributeEffect] = {}
        for act, total in total_firings.items():
            presence_probs: Dict[str, float] = {}
            value_probs: Dict[str, Dict[Any, float]] = {}
            for attr, count in effect_counts[act].items():
                presence_probs[attr] = round(count / total, 2) if total > 0 else 0.0
                attr_total = sum(value_counts[act][attr].values())
                value_probs[attr] = {
                    val: round(cnt / attr_total, 2)
                    for val, cnt in value_counts[act][attr].items()
                }
            result[act] = AttributeEffect(
                presence_probabilities=presence_probs,
                value_probabilities=value_probs,
                total_firings=total,
            )
        return result

    def _get_activity_name_for_transition(self, transition: Transition) -> str:
        """
        Resolve the activity name for a transition.

        Args:
            transition: The Transition object to resolve.

        Returns:
            Sanitized label for labeled transitions, or the pre-assigned tau name
            for silent transitions. Falls back to 'tau_unknown_<id>' if not found.
        """
        if transition.label is not None:
            return utils.sanitize_name(transition.label)
        fallback_name = f"tau_unknown_{id(transition)}"
        return self.silent_transitions.get(transition, fallback_name)
