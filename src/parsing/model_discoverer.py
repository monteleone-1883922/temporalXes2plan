from collections import defaultdict
from typing import Collection, Dict, List, Set, Tuple, Any

import pm4py
from pm4py import PetriNet
from pm4py.objects.powl.obj import Transition

import core_utils as utils
from models import PetriNetModel

logger = utils.get_logger(__name__)


class ModelDiscoverer:
    """
    Discovers a Petri net from an event log and extracts its full structural model.

    All structural analysis (arc maps, XOR splits, AND joins) is performed inline
    so that the returned PetriNetModel is immediately usable by downstream
    components without additional passes over the net.
    """

    DISCOVERY_ALGORITHMS = {
        'alpha': pm4py.discovery.discover_petri_net_alpha,
        'inductive': pm4py.discovery.discover_petri_net_inductive,
        'heuristics': pm4py.discovery.discover_petri_net_heuristics,
        'ilp': pm4py.discovery.discover_petri_net_ilp,
    }

    def __init__(self, discovery_algorithm: str = 'inductive') -> None:
        """
        Initialize ModelDiscoverer with selected discovery algorithm.

        Args:
            discovery_algorithm: Name of the Petri net discovery algorithm.

        Raises:
            ValueError: If discovery_algorithm is not one of the supported values.
        """
        if discovery_algorithm not in self.DISCOVERY_ALGORITHMS:
            raise ValueError(
                f"Unsupported discovery algorithm: {discovery_algorithm}. "
                f"Supported algorithms: {list(self.DISCOVERY_ALGORITHMS.keys())}"
            )
        self.discovery_algorithm = discovery_algorithm

    def discover(self, train_log: Any) -> PetriNetModel:
        """
        Discover the Petri net from the log and return a complete structural model.

        Computes arc maps, XOR-split places, and AND-join activities so they are
        immediately available on the returned PetriNetModel.

        Args:
            train_log: The training event log (pm4py EventLog).

        Returns:
            PetriNetModel with petrinet, markings, activity names, silent
            transition mapping, arc maps, XOR splits, and AND joins.
        """
        discovery_function = self.DISCOVERY_ALGORITHMS[self.discovery_algorithm]
        try:
            petrinet, initial_marking, final_marking = discovery_function(train_log)
            if not initial_marking:
                logger.warning("Discovered Petri net has an empty initial marking")
            if not final_marking:
                logger.warning("Discovered Petri net has an empty final marking")

            activities, silent_transitions = self._extract_activities_and_silent(
                petrinet.transitions
            )
            trans_inputs, trans_outputs = self._build_arc_maps(petrinet.arcs)
            xor_splits = self._identify_xor_splits(petrinet.arcs)
            place_inputs = self._build_place_inputs(petrinet.arcs)

            logger.info(
                f"Successfully discovered Petri net using {self.discovery_algorithm} algorithm"
            )
            return PetriNetModel(
                petrinet=petrinet,
                initial_marking=initial_marking,
                final_marking=final_marking,
                activities=activities,
                silent_transitions=silent_transitions,
                trans_inputs=trans_inputs,
                trans_outputs=trans_outputs,
                xor_splits=xor_splits,
                place_inputs=place_inputs,
            )
        except Exception as e:
            logger.error(
                f"Error discovering Petri net with {self.discovery_algorithm} algorithm: {e}"
            )
            raise

    # ---------------------------------------------------------------------------
    # Private structural helpers
    # ---------------------------------------------------------------------------

    def _extract_activities_and_silent(
        self,
        transitions: Collection[Transition],
    ) -> Tuple[Set[str], Dict[Transition, str]]:
        """
        Build the activity name set and silent transition mapping.

        Args:
            transitions: All transitions in the discovered net.

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

    def _build_arc_maps(
        self,
        arcs: Collection[PetriNet.Arc],
    ) -> Tuple[
        Dict[Transition, Set[PetriNet.Place]],
        Dict[Transition, Set[PetriNet.Place]],
    ]:
        """
        Build input and output place maps for every transition.

        Args:
            arcs: All arcs in the discovered net.

        Returns:
            Tuple of (trans_inputs, trans_outputs):
            - trans_inputs:  Transition -> set of input Places
            - trans_outputs: Transition -> set of output Places
        """
        trans_inputs: Dict[Transition, Set[PetriNet.Place]] = defaultdict(set)
        trans_outputs: Dict[Transition, Set[PetriNet.Place]] = defaultdict(set)
        Place = pm4py.objects.petri_net.obj.PetriNet.Place
        Trans = pm4py.objects.petri_net.obj.PetriNet.Transition

        for arc in arcs:
            if isinstance(arc.source, Place) and isinstance(arc.target, Trans):
                trans_inputs[arc.target].add(arc.source)
            elif isinstance(arc.source, Trans) and isinstance(arc.target, Place):
                trans_outputs[arc.source].add(arc.target)

        return dict(trans_inputs), dict(trans_outputs)

    def _identify_xor_splits(
        self,
        arcs: Collection[PetriNet.Arc],
    ) -> Dict[PetriNet.Place, List[Transition]]:
        """
        Find XOR-split places: places with more than one outgoing transition.

        Args:
            arcs: All arcs in the discovered net.

        Returns:
            Dict mapping each XOR-split Place to its list of outgoing Transitions.
        """
        Place = pm4py.objects.petri_net.obj.PetriNet.Place
        Trans = pm4py.objects.petri_net.obj.PetriNet.Transition

        place_outgoing: Dict[PetriNet.Place, List[Transition]] = defaultdict(list)
        for arc in arcs:
            if isinstance(arc.source, Place) and isinstance(arc.target, Trans):
                place_outgoing[arc.source].append(arc.target)

        return {place: ts for place, ts in place_outgoing.items() if len(ts) > 1}

    def _build_place_inputs(
        self,
        arcs: Collection[PetriNet.Arc],
    ) -> Dict[PetriNet.Place, List[Transition]]:
        """
        Build a mapping from each place to the transitions that produce tokens into it.

        This is the inverse of trans_outputs: place_inputs[p] lists every transition t
        such that p is an output place of t (Trans→Place arcs).

        Args:
            arcs: All arcs in the discovered net.

        Returns:
            Dict mapping each Place to its list of input Transitions.
        """
        Place = pm4py.objects.petri_net.obj.PetriNet.Place
        Trans = pm4py.objects.petri_net.obj.PetriNet.Transition

        place_inputs: Dict[PetriNet.Place, List[Transition]] = defaultdict(list)
        for arc in arcs:
            if isinstance(arc.source, Trans) and isinstance(arc.target, Place):
                place_inputs[arc.target].append(arc.source)

        return dict(place_inputs)

    @staticmethod
    def _activity_name(transition: Transition, silent_transitions: Dict[Transition, str]) -> str:
        if transition.label is not None:
            return utils.sanitize_name(transition.label)
        return silent_transitions.get(transition, f"tau_unknown_{id(transition)}")
