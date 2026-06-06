from collections import defaultdict
from typing import Dict, List, Set

from pm4py import PetriNet
from pm4py.objects.powl.obj import Transition

import core_utils as utils
from models import PetriNetLog, XorSplitStats

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
