import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union
import pm4py
from pm4py import PetriNet
from pm4py.objects.powl.obj import Transition
import utils

logger = utils.get_logger(__name__)

class StructureAnalyzer:
    """
    Analyzes the structure of the Petri net to extract relationships 
    (predecessors, parallels, direct transition graphs).
    """
    def __init__(
        self, 
        edges: Set[PetriNet.Arc], 
        places: Set[PetriNet.Place], 
        silent_transitions: Dict[Transition, str]
    ) -> None:
        """
        Initialize StructureAnalyzer.

        Args:
            edges: Set of arcs (edges) in the Petri Net.
            places: Set of places in the Petri Net.
            silent_transitions: Map from Transition objects to their tau names.
        """
        self.edges = edges
        self.places = places
        self.silent_transitions = silent_transitions

    def extract_predecessors(self) -> Dict[str, List[str]]:
        """
        Extract a mapping of activities to their direct predecessors in the Petri net structure.
        Identifies predecessor activities by tracing through the input places of transitions.

        Returns:
            Dictionary mapping activity names to their list of predecessor activity names:
            ActivityName -> List[PredecessorActivityNames]

            ESEMPIO DI OUTPUT RITORNATO:
            {
                "ER_Triage": ["ER_Registration"],
                "LacticAcid_Measurement": ["ER_Triage"],
                "Admission_ICU": ["ER_Triage"],
                "tau_1": ["ER_Registration"]
            }
        """
        place_to_inputs = self._build_place_input_mapping()
        return self._build_predecessor_mapping(place_to_inputs)

    def _build_place_input_mapping(self) -> Dict[PetriNet.Place, List[Transition]]:
        """
        Maps each place to its incoming transitions (transitions that put tokens in it).

        Returns:
            Dictionary mapping Places to list of Transition objects that lead into them.
            E.g. {Place("p1"): [Transition("t1"), Transition("tau_1")]}
        """
        place_to_inputs = defaultdict(list)
        for arc in self.edges:
            if isinstance(arc.source, pm4py.objects.petri_net.obj.PetriNet.Transition) and \
               isinstance(arc.target, pm4py.objects.petri_net.obj.PetriNet.Place):
                place_to_inputs[arc.target].append(arc.source)
        return place_to_inputs

    def _build_predecessor_mapping(
        self, 
        place_to_inputs: Dict[PetriNet.Place, List[Transition]],
        remove_duplicates: bool = True
    ) -> Dict[str, List[str]]:
        """
        Builds the predecessor mapping by connecting outgoing transitions of a place to its incoming transitions.

        Args:
            place_to_inputs: Pre-computed map of places to their incoming transitions.
            remove_duplicates: If True, returns predecessor lists with duplicate elements removed.

        Returns:
            Dictionary mapping sanitized target activity name to list of predecessor activity names.
            E.g. {"ER_Triage": ["ER_Registration"]}
        """
        predecessors: Dict[str, Union[set[str], list[str]]] = defaultdict(set if remove_duplicates else list)
        for arc in self.edges:
            if (isinstance(arc.source, pm4py.objects.petri_net.obj.PetriNet.Place) and
                    isinstance(arc.target, pm4py.objects.petri_net.obj.PetriNet.Transition)):

                target_activity = self._get_activity_name_for_transition(arc.target)
                for input_transition in place_to_inputs[arc.source]:
                    input_activity = self._get_activity_name_for_transition(input_transition)
                    if remove_duplicates:
                        predecessors[target_activity].add(input_activity)
                    else:
                        predecessors[target_activity].append(input_activity)
        return {act: list(prec_act) for act, prec_act in predecessors.items()} if remove_duplicates else predecessors

    def identify_parallels(self) -> Dict[str, List[str]]:
        """
        Identify transitions that split into parallel paths (AND-splits).
        An AND-split is detected when a transition has multiple outgoing places,
        and these places lead to distinct parallel paths rather than choices.

        Returns:
            Dictionary mapping parallel-splitting activity names to their parallel successor activities:
            AndSplitActivityName -> List[ParallelSuccessorActivityNames]

            ESEMPIO DI OUTPUT RITORNATO:
            {
                "ER_Registration": ["ER_Triage", "tau_1"],
                "LacticAcid_Measurement": ["CRP_Measurement", "Leucocytes_Measurement"]
            }
        """
        place_outgoing_transitions = self._get_place_outgoing_transitions()
        transition_outgoing_places = self._get_transition_outgoing_places()
        
        # Choice/XOR-split places have more than one outgoing transitions
        decision_point_places = self._identify_decision_point_places(place_outgoing_transitions)
        
        return self._extract_parallel_patterns(
            transition_outgoing_places, 
            place_outgoing_transitions, 
            decision_point_places
        )

    def _identify_decision_point_places(
        self, 
        place_outgoing_transitions: Dict[PetriNet.Place, List[Transition]]
    ) -> Set[PetriNet.Place]:
        """
        Identifies choice places (places with multiple outgoing transitions).

        Args:
            place_outgoing_transitions: Pre-computed map of places to their outgoing transitions.

        Returns:
            Set of Place objects representing XOR-splits/choices.
        """
        decision_point_places = set()
        for place, transitions in place_outgoing_transitions.items():
            if len(transitions) > 1:
                decision_point_places.add(place)
        return decision_point_places

    def _extract_parallel_patterns(
        self, 
        transition_outgoing_places: Dict[Transition, List[PetriNet.Place]],
        place_outgoing_transitions: Dict[PetriNet.Place, List[Transition]],
        decision_point_places: Set[PetriNet.Place]
    ) -> Dict[str, List[str]]:
        """
        Identifies and returns parallel execution patterns (AND-splits) from transitions leading to distinct places.

        Args:
            transition_outgoing_places: Maps each Transition to its target Places.
            place_outgoing_transitions: Maps each Place to its outgoing Transitions.
            decision_point_places: Set of places that represent choices (exclusive OR-splits).

        Returns:
            Dictionary mapping AND-split activity names to their parallel successor activity names.
            E.g. {"LacticAcid_Measurement": ["CRP_Measurement", "Leucocytes_Measurement"]}
        """
        parallels = {}
        for transition, places in transition_outgoing_places.items():
            if len(places) > 1:
                # Skip transitions that lead to decision points (exclusive choices)
                if any(place in decision_point_places for place in places):
                    continue
                
                unique_subsequent_labels = self._get_subsequent_actions(places, place_outgoing_transitions)
                
                if len(unique_subsequent_labels) > 1:
                    transition_key = self._get_activity_name_for_transition(transition)
                    parallels[transition_key] = list(unique_subsequent_labels)
                else:
                    logger.debug(f"Transition {transition.label or transition.name} has multiple outgoing places but they all lead to the same activity set: {unique_subsequent_labels}")
        
        return parallels

    def _get_subsequent_actions(
            self,
            places: List[PetriNet.Place],
            place_outgoing_transitions: Dict[PetriNet.Place, List[Transition]]
    ) -> Set[str]:
        return self._get_subsequent_transitions(places, place_outgoing_transitions)

    def _get_subsequent_transitions(
        self, 
        places: List[PetriNet.Place],
        place_outgoing_transitions: Dict[PetriNet.Place, List[Transition]],
        return_activity: bool = True
    ) -> Union[Set[Transition], Set[str]]:
        """
        Resolves transitions reachable immediately after a set of places.

        Args:
            places: List of Place objects to trace from.
            place_outgoing_transitions: Pre-computed map of places to their outgoing transitions.
            return_activity: If True, returns set of sanitized activity name strings. If False, returns set of Transition objects.

        Returns:
            Set of Transition objects or set of sanitized activity name strings.
            E.g. (return_activity=True): {"CRP_Measurement", "Leucocytes_Measurement"}
        """
        return {
            self._get_activity_name_for_transition(next_transition)
            if return_activity
            else next_transition
            for place in places
            for next_transition in place_outgoing_transitions[place]
        }

    def _get_place_outgoing_activities(self) -> \
            Dict[PetriNet.Place, List[str]]:
        return self._get_place_outgoing_transitions_activities(True)

    def _get_place_outgoing_transitions(self) -> \
            Dict[PetriNet.Place, List[Transition]]:
        return self._get_place_outgoing_transitions_activities()

    def _get_place_outgoing_transitions_activities(self, return_act_names: bool = False) -> \
            Dict[PetriNet.Place, Union[List[Transition], List[str]]]:
        """
        Maps each place to its outgoing transitions (transitions that consume tokens from it).

        Args:
            return_act_names: If True, returns sanitized activity names instead of Transition objects.

        Returns:
            Dictionary mapping Places to list of Transitions/ActivityNames.
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

    def _get_transition_outgoing_places(self) -> Dict[Transition, List[PetriNet.Place]]:
        """
        Maps each transition to its outgoing places (places where it puts tokens).

        Returns:
            Dictionary mapping Transitions to lists of Place objects.
            E.g. {Transition("t1"): [Place("p1"), Place("p2")]}
        """
        transition_outgoing_places = defaultdict(list)
        for arc in self.edges:
            source, target = arc.source, arc.target
            if isinstance(source, pm4py.objects.petri_net.obj.PetriNet.Transition) and \
               isinstance(target, pm4py.objects.petri_net.obj.PetriNet.Place):
                transition_outgoing_places[source].append(target)
        return transition_outgoing_places

    def build_direct_transition_graph(self) -> Dict[str, List[str]]:
        """
        Build a direct transition graph by removing Petri net places and connecting transitions directly.
        This provides a clear directly-follows abstraction of the Petri net structure.

        Returns:
            Dictionary representing an adjacency list: {activity_name: [list of successor activities]}
            ActivityName -> List[SuccessorActivityNames]

            ESEMPIO DI OUTPUT RITORNATO:
            {
                "ER_Registration": ["ER_Triage", "tau_1"],
                "ER_Triage": ["LacticAcid_Measurement", "Admission_ICU"],
                "LacticAcid_Measurement": ["CRP_Measurement"]
            }
        """
        graph = defaultdict(list)
        place_incoming_activities = self._get_place_incoming_actions()
        place_outgoing_activities = self._get_place_outgoing_activities()
        
        for place in self.places:
            incoming_activities = place_incoming_activities.get(place, [])
            outgoing_activities = place_outgoing_activities.get(place, [])
            # Skip start/end places which don't sit between transitions
            if not incoming_activities or not outgoing_activities:
                continue
            
            # Connect all incoming transitions of this place to all outgoing transitions
            for incoming_a in incoming_activities:
                for outgoing_a in outgoing_activities:
                    graph[incoming_a].append(outgoing_a)
        
        return dict(graph)


    def _get_place_incoming_actions(self) -> \
            Dict[PetriNet.Place, List[str]]:
        """
        Maps each place to its incoming transitions.

        Args:
            return_act_names: If True, returns list of sanitized activity names instead of Transition objects.

        Returns:
            Dictionary mapping Places to list of incoming Transitions or activity names.
        """
        place_incoming_transitions = defaultdict(list)
        for arc in self.edges:
            if isinstance(arc.target, pm4py.objects.petri_net.obj.PetriNet.Place) and \
               isinstance(arc.source, pm4py.objects.petri_net.obj.PetriNet.Transition):
                place_incoming_transitions[arc.target].append(
                    arc.source
                )
        return place_incoming_transitions

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
