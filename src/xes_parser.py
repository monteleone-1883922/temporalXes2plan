import os
import logging
import random
from typing import List, Dict, Optional, Any, Tuple, Set, Union

import numpy as np
import pm4py
from pm4py import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from pm4py.objects.powl.obj import Transition

from decision_mining import discover_all_decision_rules, get_attribute_domains
from log_processor import LogProcessor
from model_discoverer import ModelDiscoverer
from structure_analyzer import StructureAnalyzer
from probability_estimator import ProbabilityEstimator
from correlation_miner import CorrelationMiner, RelationshipMetrics
import utils

SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

logger = utils.get_logger(__name__)

class Parser:
    """
    XES Parser for discovering Petri nets and extracting process properties.
    Acts as a Facade/Controller coordinating specialized modules.
    """

    log_path: str  # Path to the XES event log file. E.g. "../sepsisLog/Sepsis Cases - Event Log.xes"
    coverage_percentage: float  # Minimum cumulative coverage for variant filtering. E.g. 0.001
    discovery_algorithm: str  # Name of the Petri net discovery algorithm to use. E.g. "inductive"
    intervals: Dict[str, List[float]]  # Discretization split points for numerical attributes. E.g. {"crp": [10.5, 45.2]}
    use_activity_classifier: bool  # Whether to combine concept:name and lifecycle:transition as activity names
    log: Any  # The loaded and filtered event log used for discovery (training set)
    full_log: Any  # The full event log after applying classifiers but before variant filtering
    full_lifecycle_log: Any  # Copy of the log before filtering for 'complete' events
    train_df: Any  # Training set split (80% of log)
    test_df: Any  # Testing set split (20% of log)
    petrinet: PetriNet  # The discovered Petri net model
    initial_marking: Marking  # The initial token marking of the Petri net
    final_marking: Marking  # The final token marking of the Petri net
    transitions: Set[Transition]  # Set of transitions present in the Petri net
    places: Set[PetriNet.Place]  # Set of places present in the Petri net
    edges: Set[PetriNet.Arc]  # Set of arcs (edges) connecting places and transitions
    activities: Set[str]  # Set of all unique sanitized activity names in the process
    silent_transitions: Dict[Transition, str]  # Mapping from Petri net Transitions to their assigned tau names
    start_activities: Dict[str, int]  # Frequency map of activities that start a process trace. E.g. {"ER_Registration": 1000}
    end_activities: Dict[str, int]  # Frequency map of activities that end a process trace. E.g. {"Release_A": 850}
    attributes: Set[str]  # Set of all non-ignored event attributes in the log. E.g. {"crp", "age"}
    attribute_categories: Dict[str, str]  # Mapping from attribute names to types. E.g. {"crp": "numerical", "age": "categorical"}
    
    # Map of activities to their direct preceding activities in the Petri net.
    # E.g. {"LacticAcid_Measurement": ["ER_Triage"], "Admission_ICU": ["ER_Triage", "tau_1"]}
    predecessors: Dict[str, List[str]]
    
    # Execution probabilities for branches at XOR-splits.
    # E.g. {"place_XOR_1": {"ER_Triage": 0.40, "ER_Sepsis_Triage": 0.60}}
    decision_points_probabilities: Dict[str, Dict[str, float]]
    
    # Mapping of AND-splits to the set of parallel activities they enable.
    # E.g. {"LacticAcid_Measurement": ["CRP_Measurement", "Leucocytes_Measurement"]}
    parallels: Dict[str, List[str]]
    
    # Graph representation of direct succession between activities (removed places from petrinet).
    # E.g. {"ER_Registration": ["ER_Triage", "tau_1"], "ER_Triage": ["CRP_Measurement"]}
    direct_transition_graph: Dict[str, List[str]]
    
    # Set of possible values or discretized intervals for each attribute.
    # E.g. {"diagnosticlacticacid": {"0.0-1.5", "1.5-3.0", "3.0-5.0"}, "diagnose": {"A", "B", "C"}}
    attribute_domains: Dict[str, Set[Union[str, bool]]]
    
    # Extracted data samples used for decision mining (maps events to executed activities and their preconditions).
    # E.g. [{"activity": "exec_ER_Triage", "preconditions": {"(completed exec_ER_Registration)"}, "case:Age": "young"}]
    decision_samples: List[Dict[str, Any]]

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
        Initialize the Parser and orchestrate the full parsing pipeline by delegating to specialized classes.
        """
        if discovery_algorithm not in self.DISCOVERY_ALGORITHMS:
            raise ValueError(f"Unsupported discovery algorithm: {discovery_algorithm}. "
                             f"Supported algorithms: {list(self.DISCOVERY_ALGORITHMS.keys())}")
        
        self.intervals = intervals or {}
        self.discovery_algorithm = discovery_algorithm
        self.use_activity_classifier = use_activity_classifier
        self.log_path = log_path

        # 1. Log Processor: loading, filtering, and splitting
        processor = LogProcessor(self.log_path, self.use_activity_classifier)
        self.log, self.full_log, self.full_lifecycle_log = processor.load_and_filter_log(coverage_percentage)
        
        self.train_df, self.test_df = processor.split_train_test(self.log)
        self.log = self.train_df

        # 2. Model Discoverer: Petri net discovery & basic properties
        discoverer = ModelDiscoverer(self.discovery_algorithm)
        self.petrinet, self.initial_marking, self.final_marking, self.transitions, self.places, self.edges = \
            discoverer.discover(self.log)
            
        self.activities, self.silent_transitions, self.start_activities, self.end_activities = \
            discoverer.extract_basic_properties(self.transitions, self.full_log)

        # 3. Attributes initialization & categorization
        self.attributes, self.attribute_categories = processor.initialize_attributes(self.full_log)

        # 4. Structure & Probabilities computation
        self._compute_structure_and_probabilities()

    def _compute_structure_and_probabilities(self) -> None:
        """
        Compute high-level structural properties and data-driven probabilities.
        """
        self.predecessors = self.extract_predecessors()
        self.decision_points_probabilities = self.compute_decision_points_probabilities()
        self.parallels = self.identify_parallels()
        self.direct_transition_graph = self._build_direct_transition_graph()
        
        # Run decision mining to get attribute domains
        rules, intervals, samples = discover_all_decision_rules(self.log_path, log=self.full_log)
        self.attribute_domains = get_attribute_domains(samples, intervals)
        self.decision_samples = samples
        
        # Update intervals with the ones discovered from decision mining
        if intervals:
            self.intervals = dict(intervals)

    # --- Delegated Structural and Statistical Methods ---

    def extract_predecessors(self) -> Dict[str, List[str]]:
        """
        Extract a mapping of activities to their direct predecessors in the Petri net structure.
        """
        analyzer = StructureAnalyzer(self.edges, self.places, self.silent_transitions)
        return analyzer.extract_predecessors()

    def compute_decision_points_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Identify decision points (XOR-splits) and compute their branch probabilities.
        """
        estimator = ProbabilityEstimator(self.edges, self.predecessors, self.silent_transitions)
        return estimator.compute_decision_points_probabilities(self.full_log)

    def identify_parallels(self) -> Dict[str, List[str]]:
        """
        Identify transitions that split into parallel paths (AND-splits).
        """
        analyzer = StructureAnalyzer(self.edges, self.places, self.silent_transitions)
        return analyzer.identify_parallels()

    def _build_direct_transition_graph(self) -> Dict[str, List[str]]:
        """
        Build a direct transition graph by removing Petri net places and connecting transitions directly.
        """
        analyzer = StructureAnalyzer(self.edges, self.places, self.silent_transitions)
        return analyzer.build_direct_transition_graph()

    # --- Delegated Correlation Mining Methods ---

    def discover_attribute_activity_relationships(self) -> Dict[str, Dict[str, Dict[str, RelationshipMetrics]]]:
        """
        Discover how specific attribute values influence subsequent activities.
        """
        miner = CorrelationMiner(self.full_log, self.predecessors, self.intervals)
        return miner.discover_attribute_activity_relationships()

    def discover_activity_attribute_effects(self) -> Dict[str, Dict[str, Dict[str, Dict[str, RelationshipMetrics]]]]:
        """
        Discover how activities affect attribute values.
        """
        miner = CorrelationMiner(self.full_log, self.predecessors, self.intervals)
        return miner.discover_activity_attribute_effects()