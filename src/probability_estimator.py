import logging
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union, Optional, Any
import pm4py
from pm4py import PetriNet
from pm4py.objects.powl.obj import Transition
import utils

logger = utils.get_logger(__name__)

class ProbabilityEstimator:
    """
    Computes branch/decision probabilities based on the event log data.
    """
    def __init__(
        self, 
        edges: Set[PetriNet.Arc], 
        predecessors: Dict[str, List[str]], 
        silent_transitions: Dict[Transition, str]
    ) -> None:
        """
        Initialize ProbabilityEstimator.

        Args:
            edges: Set of arcs (edges) in the Petri Net.
            predecessors: Mapping of activity names to their predecessors.
            silent_transitions: Mapping from Transition objects to their tau names.
        """
        self.edges = edges
        self.predecessors = predecessors
        self.silent_transitions = silent_transitions

    def compute_decision_points_probabilities(self, full_log: Any) -> Dict[str, Dict[str, float]]:
        """
        Identify decision points (XOR-splits) in the Petri net and compute their branch probabilities
        by analyzing transition frequencies in the event log.

        Args:
            full_log: The full event log to analyze frequencies from.

        Returns:
            Dictionary mapping place names to branch probabilities:
            PlaceName -> BranchActivityName -> Probability

            ESEMPIO DI OUTPUT RITORNATO:
            {
                "place_XOR_1": {
                    "ER_Triage": 0.40,
                    "ER_Sepsis_Triage": 0.60
                },
                "place_XOR_2": {
                    "Admission_ICU": 0.15,
                    "Release_A": 0.85
                }
            }
        """
        # Step 1: Get place-to-outgoing-transitions mapping
        place_outgoing_transitions = self._get_place_outgoing_transitions()
        
        # Step 2: Keep only places with more than 1 outgoing path (XOR-splits)
        decision_points = {
            place: transitions
            for place, transitions in place_outgoing_transitions.items()
            if len(transitions) > 1
        }

        # Step 3: Compute how often activity A is directly followed by activity B in the log
        df_counts = self._compute_activity_frequencies(full_log)
        
        # Step 4: Calculate branch probabilities based on frequencies
        decision_probabilities = self._compute_decision_probabilities(decision_points, df_counts)

        return decision_probabilities

    def _get_place_outgoing_transitions(self) -> Dict[PetriNet.Place, List[Transition]]:
        return self._get_place_outgoing_transitions_activities()

    def _get_place_outgoing_transitions_activities(self, return_act_names: bool = False) -> \
            Dict[PetriNet.Place, Union[List[Transition], List[str]]]:
        """
        Maps each Petri Net Place to its list of outgoing Transitions or activity names.

        Args:
            return_act_names: If True, resolves and returns sanitized activity names instead of Transition objects.

        Returns:
            Dictionary mapping Places to lists of Transitions or sanitized activity name strings.
            E.g. (when return_act_names=False):
            {
                Place("place_1"): [Transition("t1"), Transition("t2")]
            }
            E.g. (when return_act_names=True):
            {
                Place("place_1"): ["ER_Triage", "ER_Sepsis_Triage"]
            }
        """
        place_outgoing_transitions = defaultdict(list)
        for arc in self.edges:
            source, target = arc.source, arc.target
            if isinstance(source, pm4py.objects.petri_net.obj.PetriNet.Place) and \
               isinstance(target, pm4py.objects.petri_net.obj.PetriNet.Transition):
                place_outgoing_transitions[source].append(
                    self._get_activity_name_for_transition(target)
                    if return_act_names
                    else target
                )
        return place_outgoing_transitions

    def _compute_activity_frequencies(self, full_log: Any) -> Dict[str, Dict[str, int]]:
        """
        Computes the direct succession (directly-follows) frequencies from the event log.

        Args:
            full_log: The event log log to iterate over.

        Returns:
            Nested dictionary representing:
            SourceActivity -> TargetActivity -> AbsoluteFrequencyCount

            ESEMPIO DI OUTPUT RITORNATO:
            {
                "ER_Registration": {
                    "ER_Triage": 450,
                    "ER_Sepsis_Triage": 50
                },
                "ER_Triage": {
                    "LacticAcid_Measurement": 380,
                    "Release_A": 70
                }
            }
        """
        df_counts = defaultdict(lambda: defaultdict(int))
        for trace in tqdm(full_log, desc="Computing activity frequencies"):
            for i in range(len(trace) - 1):
                current_activity = utils.sanitize_name(trace[i]["concept:name"])
                next_activity = utils.sanitize_name(trace[i+1]["concept:name"])
                df_counts[current_activity][next_activity] += 1
        return df_counts

    def _compute_decision_probabilities(
        self,
        decision_points: Dict[PetriNet.Place, List[Transition]],
        df_counts: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimates conditional branch probabilities at each choice place.
        For a choice place with branches B1, B2...:
        We find the incoming activities I1, I2... that structurally precede the branches.
        We count how often B1, B2... execute directly after I1, I2... in the event log.
        Finally we divide by the total count to get the branch probability.

        Args:
            decision_points: Dictionary mapping Places to list of outgoing branch transitions.
            df_counts: Directly-follows frequency counts mapped by source -> target activity.

        Returns:
            Dictionary mapping place names to branch probabilities:
            PlaceName -> BranchActivityName -> Probability
            E.g. {"place_XOR_1": {"ER_Triage": 0.40, "ER_Sepsis_Triage": 0.60}}
        """
        decision_probabilities = {}

        for place, outgoing_transitions in decision_points.items():
            # Get the transitions outgoing from this choice point
            # all_transitions is a list of tuples: (TransitionObj, raw_label, sanitized_label)
            all_transitions = [
                (t, self._get_activity_name_for_transition(t),
                 utils.sanitize_name(self._get_activity_name_for_transition(t)))
                for t in outgoing_transitions
            ]

            # Collect structural predecessor activities that lead to this choice point
            incoming_activities: set[str] = set()
            for _, _, sanitized_out in all_transitions:
                preds = self.predecessors.get(sanitized_out, [])
                if not preds:
                    logger.debug(f"Activity {sanitized_out} has no predecessors in the Petri net")
                for pred in preds:
                    self._add_non_tau_predecessors(pred, incoming_activities)
            
            if not incoming_activities:
                logger.debug(f"No incoming activities found for decision point {place.name}")

            # Sum up observed transitions in the log from any incoming activity to each choice branch
            transition_counts: Dict[Transition, int] = defaultdict(int)
            total_count = 0
            for in_activity in incoming_activities:
                for out_transition, _, sanitized_out in all_transitions:
                    count = df_counts.get(in_activity, {}).get(sanitized_out, 0)
                    transition_counts[out_transition] += count
                    total_count += count

            # Calculate branch probabilities
            place_probs = {}
            if total_count > 0:
                for transition, count in transition_counts.items():
                    action_name = self._get_activity_name_for_transition(transition)
                    place_probs[action_name] = round(count / total_count, 2)
            else:
                # Fallback: if no logs matched this structural choice, assign equal probabilities
                logger.warning(f"No log data found for decision point {place.name} (incoming: {incoming_activities}, outgoing: {[t[2] for t in all_transitions]}). Assigning equal probabilities.")
                equal_prob = round(1.0 / len(all_transitions), 2) if all_transitions else 0.0
                for _, action_name, _ in all_transitions:
                    place_probs[action_name] = equal_prob

            if place_probs:
                decision_probabilities[place.name] = place_probs

        return decision_probabilities

    def _add_non_tau_predecessors(
        self, 
        activity: str, 
        incoming_set: Set[str], 
        visited: Optional[Set[str]] = None
    ) -> None:
        """
        Recursively traces back through silent ('tau') transitions to find 
        the nearest actual (labeled) predecessor activities.

        Args:
            activity: The predecessor activity name (possibly silent).
            incoming_set: The output set where found labeled predecessor names are collected.
            visited: Set to track visited activity names to prevent infinite recursion in cyclic nets.
        """
        if visited is None:
            visited = set()
        
        if activity in visited:
            logger.debug(f"Cycle detected in tau transitions at activity: {activity}")
            return
        visited.add(activity)
        
        if not activity.startswith('tau'):
            incoming_set.add(activity)
        else:
            for pred in self.predecessors.get(activity, []):
                self._add_non_tau_predecessors(pred, incoming_set, visited)

    def _get_activity_name_for_transition(self, transition: Transition) -> str:
        """
        Resolves the label for a transition (returns its sanitized label or assigned tau name).

        Args:
            transition: The Transition object to check.

        Returns:
            Sanitized label name, or the pre-assigned unique silent transition ID name.
            E.g. "ER_Triage" or "tau_1"
        """
        if transition.label is not None:
            return utils.sanitize_name(transition.label)
        else:
            fallback_name = f"tau_unknown_{id(transition)}"
            return self.silent_transitions.get(transition, fallback_name)
