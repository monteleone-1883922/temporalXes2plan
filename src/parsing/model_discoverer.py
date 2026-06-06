from typing import Set, Dict, Tuple, Any, Collection
import pm4py
from pm4py.objects.powl.obj import Transition
import core_utils as utils
from models import PetriNetModel

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

    def discover(self, train_log: Any) -> PetriNetModel:
        """
        Discover the Petri net from the log and return a complete structural model.

        Args:
            train_log: The training event log.

        Returns:
            PetriNetModel containing the Petri net, markings(initial and final,
            activity names, and silent transition mapping.
        """
        discovery_function = self.DISCOVERY_ALGORITHMS[self.discovery_algorithm]
        try:
            petrinet, initial_marking, final_marking = discovery_function(train_log)
            if not initial_marking:
                logger.warning("Discovered Petri net has an empty initial marking")
            if not final_marking:
                logger.warning("Discovered Petri net has an empty final marking")

            activities, silent_transitions = self._extract_activities_and_silent(petrinet.transitions)

            logger.info(f"Successfully discovered Petri net using {self.discovery_algorithm} algorithm")
            return PetriNetModel(
                petrinet=petrinet,
                initial_marking=initial_marking,
                final_marking=final_marking,
                activities=activities,
                silent_transitions=silent_transitions,
            )
        except Exception as e:
            logger.error(f"Error discovering Petri net with {self.discovery_algorithm} algorithm: {e}")
            raise e

    def _extract_activities_and_silent(
        self,
        transitions: Collection[Transition]
    ) -> Tuple[Set[str], Dict[Transition, str]]:
        """
        Build the activity name set and silent transition mapping from the transition set.

        Args:
            transitions: Set of all Petri net transitions.

        Returns:
            Tuple of (activities, silent_transitions).
        """
        silent_transitions: Dict[Transition, str] = {}
        activities: Set[str] = set()
        tau_counter = 1

        for transition in transitions:
            if transition.label is None:
                tau_name = f"tau_{tau_counter}"
                silent_transitions[transition] = tau_name
                activities.add(tau_name)
                tau_counter += 1
            else:
                activities.add(utils.sanitize_name(transition.label))

        return activities, silent_transitions

