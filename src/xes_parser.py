import os
import logging
from tqdm import tqdm

import numpy as np
from collections import defaultdict
import pm4py
import random
from typing import List, Dict, Optional, Any, Tuple, Set, Union, Callable

from pandas import DataFrame
from pm4py import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from pm4py.objects.powl.obj import Transition

from decision_mining import discover_all_decision_rules, get_attribute_domains
import utils
import copy

SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

logger = utils.get_logger(__name__)

class Parser:
    """
    XES Parser for discovering Petri nets and extracting process properties.

    Attributes:
        log_path (str): Path to the XES event log file.
        coverage_percentage (float): Coverage percentage for variant filtering.
        discovery_algorithm (str): Petri net discovery algorithm name.
        intervals (Dict[str, List[float]]): Attribute discretization intervals.
        use_activity_classifier (bool): Whether to use Activity classifier.
        log (pm4py.objects.log.obj.EventLog): The loaded and filtered event log.
        full_log (pm4py.objects.log.obj.EventLog): The full event log (pre-filtering).
        petrinet (pm4py.objects.petri_net.obj.PetriNet): Discovered Petri net.
        initial_marking (pm4py.objects.petri_net.obj.Marking): Initial marking.
        final_marking (pm4py.objects.petri_net.obj.Marking): Final marking.
        transitions (Set[pm4py.objects.petri_net.obj.PetriNet.Transition]): Transitions.
        places (Set[pm4py.objects.petri_net.obj.PetriNet.Place]): Places.
        edges (Set[pm4py.objects.petri_net.obj.PetriNet.Arc]): Arcs.
        activities (Set[str]): Set of activity names.
        silent_transitions (List[Tuple[pm4py.objects.petri_net.obj.PetriNet.Transition, str]]): Tau transitions.
        start_activities (Dict[str, int]): Map of start activities to frequencies.
        end_activities (Dict[str, int]): Map of end activities to frequencies.
        attributes (Set[str]): Set of event attributes.
        attribute_categories (Dict[str, str]): Map of attributes to categories.
        predecessors (Dict[str, List[str]]): Map of activities to predecessors.
        decision_points_probabilities (Dict[str, Dict[str, float]]): Probabilities at decision points.
        parallels (Dict[str, List[str]]): Map of AND-splits to parallel activities.
        direct_transition_graph (Dict[str, List[str]]): Direct transition graph.
        attribute_domains (Dict[str, Set[str]]): Possible values for attributes.
        decision_samples (List[Dict[str, Any]]): Samples for decision mining.
    """

    log_path: str
    coverage_percentage: float
    discovery_algorithm: str
    intervals: Dict[str, List[float]]
    use_activity_classifier: bool
    log: Any
    full_log: Any
    train_df: Union[tuple[EventLog, EventLog], tuple[DataFrame, DataFrame]]
    test_df: Union[tuple[EventLog, EventLog], tuple[DataFrame, DataFrame]]
    petrinet: PetriNet
    initial_marking: Marking
    final_marking: Marking
    transitions: Set[Transition]
    places: Set[PetriNet.Place]
    edges: Set[PetriNet.Arc]
    activities: Set[str]
    silent_transitions: Dict[Transition, str]
    start_activities: Dict[str, int]
    end_activities: Dict[str, int]
    attributes: Set[str]
    attribute_categories: Dict[str, str]
    predecessors: Dict[str, List[str]] #activities and the list of preceding activities (activities executed right before)
    decision_points_probabilities: Dict[str, Dict[str, float]]
    parallels: Dict[str, List[str]]
    direct_transition_graph: Dict[str, List[str]]
    attribute_domains: Dict[str, Set[Union[str, bool]]]
    decision_samples: List[Dict[str, Union[str, Set[str]]]]
    IGNORED_ATTRIBUTES = {'case:concept:name', 'concept:name', 'time:timestamp', 'lifecycle:transition', 'org:resource', 'org:group', 'variant-index', 'Resource', 'org:role'}
    MIN_PROBABILITY_THRESHOLD = 0.1
    STRONG_PROBABILITY_THRESHOLD = 0.2
    MIN_SUPPORT_COUNT = 1
    MIN_SUPPORT_FOR_RELATIONSHIPS = 1

    DISCOVERY_ALGORITHMS = {
        'alpha': pm4py.discovery.discover_petri_net_alpha,
        'inductive': pm4py.discovery.discover_petri_net_inductive,
        'heuristics': pm4py.discovery.discover_petri_net_heuristics,
        'ilp': pm4py.discovery.discover_petri_net_ilp
    }

    def __init__(
        self, 
        log_path: str, 
        coverage_percentage: float, 
        discovery_algorithm: str = 'inductive', 
        intervals: Optional[Dict[str, List[float]]] = None,
        use_activity_classifier: bool = False
    ) -> None:
        """
        Initialize the Parser with a specified discovery algorithm and load the event log.

        This constructor orchestrates the full parsing pipeline:
        1. Loading and filtering the XES log.
        2. Splitting into train/test sets.
        3. Discovering the Petri net structure.
        4. Extracting basic properties (activities, transitions).
        5. Initializing attribute metadata.
        6. Computing complex structural properties and probabilities.
        
        Args:
            log_path: Path to the XES event log file (.xes).
            coverage_percentage: Minimum cumulative coverage percentage for variant filtering.
            discovery_algorithm: Algorithm to use for Petri net discovery ('alpha', 'inductive', 'heuristics', 'ilp').
            intervals: Pre-defined discretization intervals for numeric attributes.
            use_activity_classifier: If True, combine concept:name with lifecycle:transition
                                     to create activity names (Activity classifier). Needed for logs
                                     where events are not uniquely identified by concept:name alone.
        """
        if discovery_algorithm not in self.DISCOVERY_ALGORITHMS:
            raise ValueError(f"Unsupported discovery algorithm: {discovery_algorithm}. "
                           f"Supported algorithms: {list(self.DISCOVERY_ALGORITHMS.keys())}")
        self.intervals = intervals or {}
        self.discovery_algorithm = discovery_algorithm
        self.use_activity_classifier = use_activity_classifier
        self.log_path = log_path
        self.log = self._load_and_filter_log(coverage_percentage)
        self._split_train_test()
        self._discover_petri_net()
        self._extract_basic_properties() # Extract activities, start/end activities, and identify silent transitions
        self._initialize_attributes() # Extract attributes datatypes (numeric, boolean, categorical)
        self._compute_structure_and_probabilities()

    def _split_train_test(self) -> None:
        """
        Split the loaded event log into training (80%) and testing (20%) sets.
        
        The training set is used for Petri net discovery and property extraction,
        while the test set is reserved for evaluation.
        """
        self.train_df, self.test_df = pm4py.split_train_test(self.log, train_percentage=0.8)
        self.log = self.train_df

    def _discover_petri_net(self) -> None:
        """
        Discover the Petri net from the training log using the selected algorithm.
        
        This method populates self.petrinet, self.initial_marking, self.final_marking,
        and extracts the sets of transitions, places, and edges.
        """
        discovery_function = self.DISCOVERY_ALGORITHMS[self.discovery_algorithm]
        try:
            self.petrinet, self.initial_marking, self.final_marking = discovery_function(self.log)
            if not self.initial_marking:
                logger.warning(f"Discovered Petri net has an empty initial marking")
            if not self.final_marking:
                logger.warning(f"Discovered Petri net has an empty final marking")
            self.transitions = set(self.petrinet.transitions)
            self.places = set(self.petrinet.places)
            self.edges = set(self.petrinet.arcs)
            #pm4py.vis.view_petri_net(self.petrinet, self.initial_marking, self.final_marking)
            logger.info(f"Successfully discovered Petri net using {self.discovery_algorithm} algorithm")
        except Exception as e:
            logger.error(f"Error discovering Petri net with {self.discovery_algorithm} algorithm: {e}")
            

    def _extract_basic_properties(self) -> None:
        """
        Extract core process properties from the discovered Petri net and full log.
        
        Handles:
        - Mapping Petri net transitions to sanitized activity names.
        - Identifying and naming silent (tau) transitions.
        - Identifying start and end activities with their frequencies.
        """
        self.silent_transitions = {}
        activities_set = set()
        tau_counter = 1
        
        for transition in self.transitions:
            if transition.label is None: # Create a tau action for a silent transition
                tau_name = f"tau_{tau_counter}"
                self.silent_transitions[transition] = tau_name
                activities_set.add(tau_name)
                tau_counter += 1
            else: # Regular labeled transition
                sanitized_name = utils.sanitize_name(transition.label)
                activities_set.add(sanitized_name)
        
        self.activities = activities_set
        self.start_activities = {utils.sanitize_name(activity): freq 
                               for activity, freq in pm4py.get_start_activities(self.full_log).items()}
        self.end_activities = {utils.sanitize_name(activity): freq 
                             for activity, freq in pm4py.get_end_activities(self.full_log).items()}


    def _initialize_attributes(self) -> None:
        """
        Extract and categorize event and trace attributes.
        
        Identifies all non-ignored attributes and determines if they are 
        'boolean', 'numerical', or 'categorical'.
        """
        self.attributes = {utils.sanitize_name(attr) for attr in pm4py.get_event_attributes(self.full_log) 
                          if attr not in Parser.IGNORED_ATTRIBUTES}
        self.attribute_categories = self._categorize_attributes()

    def _compute_structure_and_probabilities(self) -> None:
        """
        Compute high-level structural properties and data-driven probabilities.
        
        This method performs:
        1. Predecessor extraction (activity dependencies).
        2. Branch probability computation for decision points (XOR-splits).
        3. Parallel pattern identification (AND-splits).
        4. Direct transition graph construction.
        5. Decision mining for attribute rules and domains.
        """
        self.predecessors = self.extract_predecessors()
        self.decision_points_probabilities = self.compute_decision_points_probabilities()
        self.parallels = self.identify_parallels()
        self.direct_transition_graph = self._build_direct_transition_graph()
        # print("Probabilities at decision points:", self.decision_points_probabilities)
        # print("Graph of direct transitions:", dict(self.direct_transition_graph))
        
        # Run decision mining to get attribute domains
        # Pass the already-transformed log (with Activity classifier applied if enabled)
        rules, intervals, samples = discover_all_decision_rules(self.log_path, log=self.full_log)
        self.attribute_domains = get_attribute_domains(samples, intervals)
        self.decision_samples = samples
        # Update intervals with the ones discovered from decision mining
        if intervals:
            self.intervals = dict(intervals)  # Convert defaultdict to regular dict

    def _load_and_filter_log(self, coverage_percentage: float) -> Any:
        """
        Load the XES log, apply classifiers, and filter by coverage.

        Args:
            coverage_percentage: Minimum cumulative coverage for variant filtering.

        Returns:
            The loaded and filtered event log object.
        """
        log = pm4py.objects.log.importer.xes.importer.apply(self.log_path)
        self.full_lifecycle_log = copy.deepcopy(log)

        if self.use_activity_classifier:
            # When using Activity classifier, combine concept:name + lifecycle:transition
            # This is needed for logs where events are not uniquely identified by concept:name
            # alone (e.g., BPIC 2013)
            for trace in tqdm(log, desc="Applying Activity classifier"):
                for event in trace:
                    lifecycle = event.get('lifecycle:transition', '')
                    if lifecycle:
                        concept_name = event.get('concept:name', '')
                        # Combine as "ActivityName (Lifecycle)" - e.g., "Accepted (In Progress)"
                        event['concept:name'] = f"{concept_name} ({lifecycle})"
            logger.info("Using Activity classifier: combining concept:name + lifecycle:transition")
        else:
            # Filter full log for lifecycle (keep only 'complete' events)
            try:
                lifecycle_values = pm4py.get_event_attribute_values(log, 'lifecycle:transition')
                if 'complete' in lifecycle_values:
                    log = pm4py.filter_event_attribute_values(log, 'lifecycle:transition', 'complete')
            except:
                pass

        self.full_log = log  # Store full log for frequency computation
        
        try:
            filtered_log = pm4py.filter_variants_by_coverage_percentage(log, coverage_percentage)
            if len(filtered_log) > 0:
                log = filtered_log
            else:
                logger.warning(f"Warning: Coverage filter removed all traces, using full log")
        except Exception as e:
            logger.warning(f"Warning: Coverage filtering failed: {e}, using full log")
            
        return log
    
    
    def _categorize_attributes(self) -> Dict[str, str]:
        """
        Categorize attributes into 'boolean', 'numerical', or 'categorical'.

        Returns:
            Dictionary mapping attribute names to categories.
        """
        attr_categories = {}
        
        # Include trace (case) attributes
        for attr in pm4py.get_trace_attributes(self.full_log):
            sanitized_name = utils.sanitize_name(f"case:{attr}")
            values = [trace.attributes[attr] for trace in tqdm(self.full_log, desc=f"Categorizing trace attribute {attr}", leave=False) if attr in trace.attributes]
            if values:
                attr_categories[sanitized_name] = 'boolean' \
                    if all(isinstance(val, bool) for val in values) \
                    else 'numerical' \
                    if all(isinstance(val, (int, float)) and not isinstance(val, bool) for val in values) \
                    else 'categorical'
        
        # Include event attributes
        for attr in pm4py.get_event_attributes(self.full_log):
            if attr in Parser.IGNORED_ATTRIBUTES:
                continue 
            sanitized_name = utils.sanitize_name(attr)
            values = pm4py.get_event_attribute_values(self.full_log, attr).keys()
            attr_categories[sanitized_name] = 'boolean' \
                if all(isinstance(val, bool) for val in values) \
                else 'numerical' \
                if all(isinstance(val, (int, float)) and not isinstance(val, bool) for val in values) \
                else 'categorical'
            
            # Log if mixed types detected
            if attr_categories[sanitized_name] == 'categorical':
                has_numeric = any(isinstance(val, (int, float)) and not isinstance(val, bool) for val in values)
                has_bool = any(isinstance(val, bool) for val in values)
                if has_numeric or has_bool:
                    logger.warning(f"Attribute {attr} has mixed data types, forcing to 'categorical'")
        
        return attr_categories
    
    
    def _get_place_outgoing_transitions(self, return_act_names: bool = False) -> \
            Dict[PetriNet.Place, Union[List[Transition], List[str]]]:
        """
        Get a mapping of places to their outgoing transitions.
        
        Args:
            return_act_names: If True, return sanitized activity names instead of Transition objects.
            
        Returns:
            Dictionary where keys are PetriNet.Place and values are lists of outgoing transitions/names.
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
        Get a mapping of transitions to their outgoing places.
        
        Returns:
            Dictionary where keys are Transition objects and values are lists of subsequent PetriNet.Place objects.
        """
        transition_outgoing_places = defaultdict(list)
        for arc in self.edges:
            source, target = arc.source, arc.target
            if isinstance(source, pm4py.objects.petri_net.obj.PetriNet.Transition) and \
               isinstance(target, pm4py.objects.petri_net.obj.PetriNet.Place):
                transition_outgoing_places[source].append(target)
        return transition_outgoing_places

    def _compute_activity_frequencies(self) -> Dict[str, Dict[str, int]]:
        """
        Compute the frequency of direct transitions between activities from the event log.
        
        This builds a transition matrix representing how many times activity B follows activity A.
        
        Returns:
            A nested dictionary: {current_activity: {next_activity: frequency_count}}
        """
        df_counts = defaultdict(lambda: defaultdict(int))
        for trace in tqdm(self.full_log, desc="Computing activity frequencies"):
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
        Compute execution probabilities for each branch at decision points (XOR-splits).

        Probabilities are estimated based on how often the subsequent activities follow 
        the activities that precede the decision point in the event log.

        Args:
            decision_points: Dictionary mapping places (split points) to their outgoing transitions.
            df_counts: Nested dictionary of activity transition frequencies.

        Returns:
            Mapping of place names to branch probabilities.
            Example: {'p1': {'Activity_A': 0.7, 'Activity_B': 0.3}}
        """
        # For each place contains a dict[next executable activity, probability to be executed]
        decision_probabilities = {}

        for place, outgoing_transitions in decision_points.items():
            #  activities executable from a certain place
            all_transitions = [
                (t, self._get_activity_name_for_transition(t),
                 utils.sanitize_name(self._get_activity_name_for_transition(t)))
                for t in outgoing_transitions
            ]

            # predecessor activities (in the PetriNet without tau activities)
            # set of all predecessors forall next executable actions
            incoming_activities: set[str] = set()
            for _, _, sanitized_out in all_transitions:
                # iterate activities preceding the executable activities for this place
                preds = self.predecessors.get(sanitized_out, [])
                if not preds:
                    logger.debug(f"Activity {sanitized_out} has no predecessors in the Petri net")
                for pred in preds:
                    self._add_non_tau_predecessors(pred, incoming_activities)
            
            if not incoming_activities:
                logger.debug(f"No incoming activities found for decision point {place.name}")

            # count for each next activity how many times it happens after the incoming activities (in the log)
            transition_counts: Dict[Transition, int] = defaultdict(int)
            # sum of all transition_counts
            total_count = 0
            # for preceding activity
            for in_activity in incoming_activities:
                # for next activities
                for out_transition, _, sanitized_out in all_transitions:
                    count = df_counts.get(in_activity, {}).get(sanitized_out, 0)
                    transition_counts[out_transition] += count
                    total_count += count

            # dict[next executable activity, probability to be executed]
            place_probs = {}
            if total_count > 0:
                for transition, count in transition_counts.items():
                    action_name = self._get_activity_name_for_transition(transition)
                    place_probs[action_name] = round(count / total_count, 2)
            else:
                # Assign equal probabilities if no data available
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
        Recursively find non-tau predecessors, skipping intermediate tau transitions.
        
        This is used to find "real" activities that precede a place, even if the 
        immediate predecessors are silent transitions.
        """
        if visited is None:
            visited = set()
        
        if activity in visited:
            logger.debug(f"Cycle detected in tau transitions at activity: {activity}")
            return  # Prevent infinite recursion on cycles
        visited.add(activity)
        
        if not activity.startswith('tau'):
            incoming_set.add(activity)
        else:
            for pred in self.predecessors.get(activity, []):
                self._add_non_tau_predecessors(pred, incoming_set, visited)

    def compute_decision_points_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Identify decision points (XOR-splits) and compute their branch probabilities.
        
        A decision point is a Petri net place with multiple outgoing transitions.
        
        Returns:
            Dictionary mapping place names to their branch probability maps.
        """
        place_outgoing_transitions = self._get_place_outgoing_transitions()
        
        decision_points = {
            place : transitions
            for place, transitions in place_outgoing_transitions.items()
            if len(transitions) > 1
        }

        df_counts = self._compute_activity_frequencies()
        decision_probabilities = self._compute_decision_probabilities(decision_points, df_counts)

        return decision_probabilities
        
        
    def identify_parallels(self) -> Dict[str, List[str]]:
        """
        Identify transitions that split into parallel paths (AND-splits).
        
        A parallel split (AND-split) occurs when a transition has multiple outgoing 
        places that lead to different subsequent transitions which are not exclusive choices.
        
        Returns:
            Dictionary mapping parallel-splitting activity names to their subsequent activities.
            Example: {'Fork_Activity': ['Parallel_Branch_A', 'Parallel_Branch_B']}
        """
        place_outgoing_transitions = self._get_place_outgoing_transitions()
        transition_outgoing_places = self._get_transition_outgoing_places()
        
        decision_point_places = self._identify_decision_point_places(place_outgoing_transitions)
        
        return self._extract_parallel_patterns(transition_outgoing_places, place_outgoing_transitions, decision_point_places)

    def _identify_decision_point_places(self, place_outgoing_transitions: Dict[PetriNet.Place, List[Transition]]) -> Set[PetriNet.Place]:
        """
        Identify places that represent decision points (XOR-splits).
        
        Args:
            place_outgoing_transitions: Map of places to their outgoing transitions.
            
        Returns:
            Set of PetriNet.Place objects that have more than one output.
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
        Extract parallel execution patterns (AND-splits) from the Petri net structure.

        Args:
            transition_outgoing_places: Map of transitions to their output places.
            place_outgoing_transitions: Map of places to their output transitions.
            decision_point_places: Set of places identified as XOR-splits (to be excluded).

        Returns:
            Dictionary: {activity_name: [list of parallel successor activity names]}
        """
        parallels = {}
        for transition, places in transition_outgoing_places.items():
            if len(places) > 1:
                # Skip transitions that lead to decision points (exclusive choices)
                if any(place in decision_point_places for place in places):
                    continue
                
                unique_subsequent_labels = self._get_subsequent_transitions(places, place_outgoing_transitions)
                
                if len(unique_subsequent_labels) > 1:
                    transition_key = self._get_activity_name_for_transition(transition)
                    parallels[transition_key] = list(unique_subsequent_labels)
                else:
                    logger.debug(f"Transition {transition.label or transition.name} has multiple outgoing places but they all lead to the same activity set: {unique_subsequent_labels}")
        
        return parallels

    def _get_subsequent_transitions(
        self, 
        places: List[PetriNet.Place],
        place_outgoing_transitions: Dict[PetriNet.Place, List[Transition]],
        return_activity: bool = True
    ) -> Union[Set[Transition], Set[str]]:
        """
        Get all transitions following a list of places.
        
        Args:
            places: List of PetriNet.Place objects to find successors for.
            place_outgoing_transitions: Map of places to their outgoing transitions.
            return_activity: If True, return activity names instead of Transition objects.
            
        Returns:
            Set of transitions (or activity names) that follow the given places.
        """
        return {
            self._get_activity_name_for_transition(next_transition)
            if return_activity
            else next_transition
            for place in places
            for next_transition in place_outgoing_transitions[place]
        }

    

    def extract_predecessors(self) -> Dict[str, List[str]]:
        """
        Extract a mapping of activities to their direct predecessors in the Petri net structure.
        
        This builds a dependency map representing which activities must be completed 
        immediately before another activity can start.
        
        Returns:
            Dictionary: {activity_name: [list of predecessor activity names]}
        """
        place_to_inputs = self._build_place_input_mapping()
        # predecessor activities
        return self._build_predecessor_mapping(place_to_inputs)


    def _build_place_input_mapping(self) -> Dict[PetriNet.Place, List[Transition]]:
        """
        Build a mapping from Petri net places to their input transitions.
        
        Returns:
            Dictionary where keys are PetriNet.Place and values are lists of incoming Transition objects.
        """
        place_to_inputs = defaultdict(list)
        for arc in self.edges:
            if isinstance(arc.source, pm4py.objects.petri_net.obj.PetriNet.Transition) and \
               isinstance(arc.target, pm4py.objects.petri_net.obj.PetriNet.Place):
                place_to_inputs[arc.target].append(arc.source)
        return place_to_inputs

    def _build_predecessor_mapping(self, place_to_inputs: Dict[PetriNet.Place, List[Transition]],
                                   remove_duplicates: bool = True) -> Dict[str, List[str]]:
        """
        Build a predecessor activity mapping from a place-to-input transition mapping.
        
        Args:
            place_to_inputs: Map of places to their incoming transitions.
            remove_duplicates: If True, ensure the predecessor lists contain unique activity names.
            
        Returns:
            Dictionary: {activity_name: [list of preceding activity names]}
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

    def _get_activity_name_for_transition(self, transition: Transition) -> str:
        """
        Get the sanitized activity name for a given Petri net transition.
        
        This handles both regular transitions (using their labels) and 
        silent (tau) transitions (using the names assigned during basic property extraction).
        
        Args:
            transition: The PetriNet.Transition object.
            
        Returns:
            The sanitized string name of the activity.
        """
        if transition.label is not None:
            return utils.sanitize_name(transition.label)
        else:
            # Find the tau name for this silent
            # If not found, create a new tau name (shouldn't happen if _extract_basic_properties ran)
            fallback_name = f"tau_unknown_{id(transition)}"
            logger.warning(f"Using fallback name for unlabeled transition: {fallback_name}")
            return self.silent_transitions.get(transition, fallback_name)


    def discover_attribute_activity_relationships(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Discover how specific attribute values influence subsequent activities.
        
        This analysis identifies cases where an attribute value significantly correlates 
        with the execution of certain activities over others.
        
        Returns:
            A nested dictionary of significant relationships.
            Example: {
                'age': {
                    'old': {
                        'Admit_ICU': {'count': 45, 'probability': 0.8}
                    }
                }
            }
        """
        relationships = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for trace in tqdm(self.full_log, desc="Discovering attribute-activity relationships"):
            for i in range(len(trace) - 1):
                current_event = trace[i]
                next_event = trace[i+1]
                next_activity = next_event['concept:name']
                source_activity = current_event['concept:name']
                sanitized_source = utils.sanitize_name(source_activity)
                sanitized_target = utils.sanitize_name(next_activity)
                
                if sanitized_target in self.predecessors and sanitized_source in self.predecessors[sanitized_target]:
                    for attr, val in current_event.items():
                        if attr in self.IGNORED_ATTRIBUTES:
                            continue
                        
                        sanitized_attr = utils.sanitize_name(attr)
                        relationships[sanitized_attr][str(val)][sanitized_target] += 1
        significant_relationships = {}
        for attr, values in relationships.items():
            if len(values) <= 1:
                logger.debug(f"Skipping attribute {attr} in relationship discovery (only {len(values)} unique value(s) found)")
                continue
            common_activities = set.intersection(*[set(activities.keys()) for activities in values.values()])
            non_discriminative = {activity for activity in common_activities 
                                 if all(activities[activity] / sum(activities.values()) >= self.STRONG_PROBABILITY_THRESHOLD 
                                       for activities in values.values())}
            significant_attr_relationships = {}
            
            for val, activities in values.items():
                total = sum(activities.values())
                if total >= self.MIN_SUPPORT_FOR_RELATIONSHIPS:
                    significant_activities = {
                        activity: {'count': count, 'probability': count/total}
                        for activity, count in activities.items()
                        if activity not in non_discriminative and count/total >= 0.2
                    }
                    if significant_activities:
                        significant_attr_relationships[val] = significant_activities
            
            if significant_attr_relationships:
                significant_relationships[attr] = significant_attr_relationships
        
        return significant_relationships
    

    def discover_activity_attribute_effects(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover how activities affect attribute values.
        
        This analysis captures both:
        - Changes to existing attribute values (e.g., 'status' changing from 'open' to 'closed').
        - Introductions of new attribute values (e.g., a measurement value first appearing).
        
        Returns:
            Nested dictionary of significant attribute effects.
            Example: {
                'Confirm_Payment': {
                    'payment_status': {
                        'pending': {'cleared': {'count': 100, 'probability': 1.0}}
                    },
                    'invoice_id': {
                        'NEW': {'INV-001': {'count': 1, 'probability': 1.0}}
                    }
                }
            }
        """
        effects = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        new_value_introductions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        valid_transitions = self._get_valid_transitions()
        
        self._collect_attribute_effects(effects, new_value_introductions, valid_transitions)
        
        return self._build_significant_effects(effects, new_value_introductions)

    def _get_valid_transitions(self) -> Set[Tuple[str, str]]:
        """
        Identify valid activity transitions (A -> B) from the Petri net structure.
        
        Returns:
            Set of tuples (source_activity, target_activity).
        """
        valid_transitions = set()
        for target_activity, source_activities in self.predecessors.items():
            for source_activity in source_activities:
                valid_transitions.add((source_activity, target_activity))
        return valid_transitions

    def _collect_attribute_effects(
        self, 
        effects: Dict[str, Dict[str, Dict[str, Dict[str, int]]]], 
        new_value_introductions: Dict[str, Dict[str, Dict[str, int]]], 
        valid_transitions: Set[Tuple[str, str]]
    ) -> None:
        """
        Traverse the event log to collect empirical evidence of attribute changes and introductions.
        
        Args:
            effects: Accumulator for attribute transitions.
            new_value_introductions: Accumulator for first-time attribute appearances.
            valid_transitions: Set of valid transitions to restrict the analysis.
        """
        for trace in tqdm(self.full_log, desc="Collecting attribute effects"):
            attributes_seen_in_trace = set()
            prev_event = None
            
            for event in trace:
                activity = event['concept:name']
                sanitized_activity = utils.sanitize_name(activity)
                self._track_new_attribute_introductions(
                    event, sanitized_activity, attributes_seen_in_trace, new_value_introductions
                )
                if prev_event is not None:
                    self._track_attribute_changes(
                        prev_event, event, sanitized_activity, valid_transitions, effects
                    )
                
                prev_event = event

    def _track_new_attribute_introductions(
        self, 
        event: Dict[str, Any], 
        sanitized_activity: str, 
        attributes_seen_in_trace: Set[str], 
        new_value_introductions: Dict[str, Dict[str, Dict[str, int]]]
    ) -> None:
        """
        Identify and record attributes that are appearing for the first time in a trace.
        
        Args:
            event: The current event dictionary.
            sanitized_activity: The current activity name.
            attributes_seen_in_trace: Set of attributes already encountered in this trace.
            new_value_introductions: Accumulator for recording the introduction.
        """
        for attr, val in event.items():
            if attr in self.IGNORED_ATTRIBUTES:
                continue
            sanitized_attr = utils.sanitize_name(attr)
            if sanitized_attr not in attributes_seen_in_trace:
                discretized_val = utils.discretize_value(sanitized_attr, val, self.intervals)
                new_value_introductions[sanitized_activity][sanitized_attr][discretized_val] += 1
                attributes_seen_in_trace.add(sanitized_attr)

    def _track_attribute_changes(
        self, 
        prev_event: Dict[str, Any], 
        event: Dict[str, Any], 
        sanitized_activity: str, 
        valid_transitions: Set[Tuple[str, str]], 
        effects: Dict[str, Dict[str, Dict[str, Dict[str, int]]]]
    ) -> None:
        """
        Detect and record changes in attribute values between subsequent events.
        
        Args:
            prev_event: The preceding event dictionary.
            event: The current event dictionary.
            sanitized_activity: The current activity name.
            valid_transitions: Valid Petri net transitions.
            effects: Accumulator for recording the changes.
        """
        prev_activity = prev_event['concept:name']
        sanitized_prev_activity = utils.sanitize_name(prev_activity)
        
        if (sanitized_prev_activity, sanitized_activity) not in valid_transitions:
            return
        
        # Look for attribute values that changed after the previous activity
        for attr in set(prev_event.keys()).union(event.keys()):
            if attr in self.IGNORED_ATTRIBUTES:
                continue
                
            prev_val = prev_event.get(attr)
            curr_val = event.get(attr)
            if prev_val is None or curr_val is None or prev_val == curr_val:
                continue
                
            sanitized_attr = utils.sanitize_name(attr)
            prev_val_str = str(prev_val).lower() if isinstance(prev_val, bool) else str(prev_val)
            curr_val_str = str(curr_val).lower() if isinstance(curr_val, bool) else str(curr_val)
            effects[sanitized_prev_activity][sanitized_attr][prev_val_str][curr_val_str] += 1

    def _build_significant_effects(
        self, 
        effects: Dict[str, Dict[str, Dict[str, Dict[str, int]]]], 
        new_value_introductions: Dict[str, Dict[str, Dict[str, int]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process the raw counters into significant attribute effect probabilities.
        
        Args:
            effects: Raw counters for attribute transitions.
            new_value_introductions: Raw counters for attribute introductions.
            
        Returns:
            Dictionary of filtered and structured effects.
        """
        significant_effects = {}
        for activity, attributes in effects.items():
            significant_effects[activity] = {}
            for attr, from_values in attributes.items():
                significant_from_values = self._filter_significant_transitions(from_values)
                if significant_from_values:
                    significant_effects[activity][attr] = significant_from_values
        for activity, attributes in new_value_introductions.items():
            if activity not in significant_effects:
                significant_effects[activity] = {}
            
            for attr, values in attributes.items():
                total_introductions = sum(values.values())
                if total_introductions >= 1:
                    if attr not in significant_effects[activity]:
                        significant_effects[activity][attr] = {}
                    
                    significant_effects[activity][attr]['NEW'] = {}
                    for val, count in values.items():
                        probability = count / total_introductions
                        if probability >= self.MIN_PROBABILITY_THRESHOLD:
                            significant_effects[activity][attr]['NEW'][val] = {
                                'count': count,
                                'probability': probability
                            }
        
        return significant_effects

    def _filter_significant_transitions(self, from_values: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
        """
        Filter attribute transitions based on minimum probability and support count thresholds.
        
        Args:
            from_values: Map of destination values to their occurrence counts.
            
        Returns:
            Filtered map with probabilities.
        """
        significant_from_values = {}
        
        for from_val, to_values in from_values.items():
            total_transitions = sum(to_values.values())
            if total_transitions >= self.MIN_SUPPORT_COUNT:
                significant_to_values = {}
                for to_val, count in to_values.items():
                    probability = count / total_transitions
                    if probability >= self.MIN_PROBABILITY_THRESHOLD:
                        significant_to_values[to_val] = {
                            'count': count,
                            'probability': probability
                        }
                
                if significant_to_values:
                    significant_from_values[from_val] = significant_to_values
        
        return significant_from_values

    def _build_direct_transition_graph(self) -> Dict[str, List[str]]:
        """
        Build a direct transition graph by removing Petri net places and connecting transitions directly.
        
        Silent (tau) transitions are included as explicit nodes. This graph is used 
        for reachability analysis and plan validation.
        
        Returns:
            Dictionary representing an adjacency list: {activity_name: [list of successor activities]}
        """
        graph = defaultdict(list)
        # get transactions outgoing and ingoing place already translated to activities
        place_incoming_activities = self._get_place_incoming_transitions(return_act_names=True)
        place_outgoing_activities = self._get_place_outgoing_transitions(return_act_names=True)
        
        for place in self.places:
            incoming_activities = place_incoming_activities.get(place, [])
            outgoing_activities = place_outgoing_activities.get(place, [])
            # skip starting or ending places
            if not incoming_activities or not outgoing_activities:
                continue
            
            # For each incoming activity, connect to all outgoing activities
            for incoming_a in incoming_activities:
                for outgoing_a in outgoing_activities:
                    graph[incoming_a].append(outgoing_a)
        
        return dict(graph)


    def _get_place_incoming_transitions(self, return_act_names: bool = False) -> \
            Dict[PetriNet.Place, Union[List[Transition], List[str]]]:
        """
        Get a mapping of Petri net places to their incoming transitions.
        
        Args:
            return_act_names: If True, return activity names instead of Transition objects.
            
        Returns:
            Dictionary where keys are PetriNet.Place and values are lists of incoming transitions.
        """
        place_incoming_transitions = defaultdict(list)
        for arc in self.edges:
            if isinstance(arc.target, pm4py.objects.petri_net.obj.PetriNet.Place) and \
               isinstance(arc.source, pm4py.objects.petri_net.obj.PetriNet.Transition):
                place_incoming_transitions[arc.target].append(
                    self._get_activity_name_for_transition(arc.source)
                    if return_act_names
                    else arc.source
                )
        return place_incoming_transitions