import logging
from typing import Set, Dict, Tuple, Any, Optional
import pm4py
from pm4py import PetriNet, Marking
from pm4py.objects.powl.obj import Transition
import utils

logger = utils.get_logger(__name__)

class ModelDiscoverer:
    """
    Handles Petri net model discovery and basic structural extraction.
    """
    DISCOVERY_ALGORITHMS = {
        'alpha': pm4py.discovery.discover_petri_net_alpha,
        'inductive': pm4py.discovery.discover_petri_net_inductive,
        'heuristics': pm4py.discovery.discover_petri_net_heuristics,
        'ilp': pm4py.discovery.discover_petri_net_ilp
    }

    def __init__(self, discovery_algorithm: str = 'inductive') -> None:
        """
        Initialize ModelDiscoverer with selected discovery algorithm.

        Args:
            discovery_algorithm: Petri net discovery algorithm.
        """
        if discovery_algorithm not in self.DISCOVERY_ALGORITHMS:
            raise ValueError(f"Unsupported discovery algorithm: {discovery_algorithm}. "
                             f"Supported algorithms: {list(self.DISCOVERY_ALGORITHMS.keys())}")
        self.discovery_algorithm = discovery_algorithm

    def discover(self, train_log: Any) -> Tuple[
        PetriNet, 
        Marking, 
        Marking, 
        Set[Transition], 
        Set[PetriNet.Place], 
        Set[PetriNet.Arc]
    ]:
        """
        Discover the Petri net from the log and return the structural sets.

        Args:
            train_log: The training event log.

        Returns:
            Tuple containing:
            - petrinet: The discovered PetriNet.
            - initial_marking: Initial Marking of the Petri Net.
            - final_marking: Final Marking of the Petri Net.
            - transitions: Set of Transition objects in the Petri net.
            - places: Set of Place objects in the Petri net.
            - edges: Set of Arc (edge) objects in the Petri net.
        """
        discovery_function = self.DISCOVERY_ALGORITHMS[self.discovery_algorithm]
        try:
            petrinet, initial_marking, final_marking = discovery_function(train_log)
            if not initial_marking:
                logger.warning("Discovered Petri net has an empty initial marking")
            if not final_marking:
                logger.warning("Discovered Petri net has an empty final marking")
            
            transitions = set(petrinet.transitions)
            places = set(petrinet.places)
            edges = set(petrinet.arcs)
            logger.info(f"Successfully discovered Petri net using {self.discovery_algorithm} algorithm")
            return petrinet, initial_marking, final_marking, transitions, places, edges
        except Exception as e:
            logger.error(f"Error discovering Petri net with {self.discovery_algorithm} algorithm: {e}")
            raise e

    def extract_basic_properties(
        self, 
        transitions: Set[Transition], 
        full_log: Any
    ) -> Tuple[Set[str], Dict[Transition, str], Dict[str, int], Dict[str, int]]:
        """
        Extract core properties from the discovered transitions and event log.

        Args:
            transitions: Set of all Petri net transitions.
            full_log: Full event log to calculate start and end frequencies.

        Returns:
            Tuple containing:
            - activities: Set of all sanitized unique activity names.
            - silent_transitions: Map of Petri Net Transitions to tau names.
            - start_activities: Frequency map of start activities.
            - end_activities: Frequency map of end activities.
        """
        silent_transitions = {}
        activities_set = set()
        tau_counter = 1
        
        for transition in transitions:
            if transition.label is None:  # Create a tau action for a silent transition
                tau_name = f"tau_{tau_counter}"
                silent_transitions[transition] = tau_name
                activities_set.add(tau_name)
                tau_counter += 1
            else:  # Regular labeled transition
                sanitized_name = utils.sanitize_name(transition.label)
                activities_set.add(sanitized_name)
        
        start_activities = {utils.sanitize_name(activity): freq 
                             for activity, freq in pm4py.get_start_activities(full_log).items()}
        end_activities = {utils.sanitize_name(activity): freq 
                           for activity, freq in pm4py.get_end_activities(full_log).items()}
                           
        return activities_set, silent_transitions, start_activities, end_activities
