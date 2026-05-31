import os
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple, Set, Union
import core_utils as utils
from parsing.xes_parser import Parser

from .encoder_models import PDDLEncodingContext
from .pddl_precondition_collector import PreconditionCollector
from .pddl_effect_collector import EffectCollector
from .pddl_action_builder import ActionBuilder
from .pddl_domain_builder import DomainBuilder
from .pddl_problem_builder import ProblemBuilder

class Encoder:
    """
    Encoder for transforming process models and event logs into PDDL domains and problems.
    This class orchestrates the generation process using specialized builder classes.
    """

    def __init__(
        self, 
        parser: Parser, 
        domain_name: str = "process_domain", 
        min_confidence: float = 0.1, 
        init: Optional[List[str]] = None, 
        goal: Optional[List[str]] = None, 
        minimal_preconditions: bool = False
    ) -> None:
        """
        Initialize the Encoder with process data and generation settings.
        
        Args:
            parser: XES parser instance containing discovered process properties.
            domain_name: Name to assign to the generated PDDL domain.
            min_confidence: Threshold for including uncertain attribute relationships.
            init: Custom PDDL predicates to include in the initial state.
            goal: Custom PDDL predicates to include in the goal condition.
            minimal_preconditions: If True, generate lightweight preconditions suitable 
                                   for suffix planning.
        """
        self.parser = parser
        self.context = PDDLEncodingContext(
            parser=parser,
            domain_name=domain_name,
            min_confidence=min_confidence,
            minimal_preconditions=minimal_preconditions,
            init=init,
            goal=goal
        )
        
        # Populate context fields from the parser
        self.context.activities = parser.activities
        self.context.attribute_categories = parser.attribute_categories
        self.context.value_mappings = {}
        self.context.attr_activity_relationships = parser.discover_attribute_activity_relationships()
        self.context.activity_attr_effects = parser.discover_activity_attribute_effects()
        self.context.tau_activities = {activity for activity in self.context.activities if activity.startswith('tau_')}
        
        self.context.decision_points = parser.decision_points_probabilities
        self.context.decision_preconditions = defaultdict(list)
        self.context.decision_attributes = set()
        # sample is the field containing the actions with their preconditions  produced by discover_all_decision_rules
        for sample in getattr(parser, 'decision_samples', []):
            activity = sample['activity'].replace('exec_', '')
            preconds = set(p.replace('exec_', '') for p in sample['preconditions'])
            self.context.decision_preconditions[activity].append(preconds)
            for p in preconds:
                if p.startswith('(') and p.endswith(')'):
                    inner = p[1:-1]
                    parts = inner.split()
                    attr = parts[0]
                    if attr in self.context.attribute_categories:
                        self.context.decision_attributes.add(attr)

        self.context.parallel_activities = parser.parallels
        self._build_parallel_activity_map()
        self._process_decision_points()
        
        self.context.split_actions = {}
        self.context.effect_alternatives = {}
        
        # Initialize sub-components
        self.precondition_collector = PreconditionCollector(self.context)
        self.effect_collector = EffectCollector(self.context)
        self.action_builder = ActionBuilder(self.context, self.precondition_collector, self.effect_collector)
        self.domain_builder = DomainBuilder(self.context, self.action_builder)
        self.problem_builder = ProblemBuilder(self.context, self.action_builder)

    def generate_domain(self, output_path: Optional[str] = None) -> str:
        """Generate PDDL domain string and optionally save to file."""
        return self.domain_builder.generate_domain(output_path)
        
    def generate_problem(self, problem_name: str = "process_problem", output_path: Optional[str] = None) -> str:
        """Generate PDDL problem file with initial state and goal."""
        return self.problem_builder.generate_problem(problem_name, output_path)

    def _build_parallel_activity_map(self) -> None:
        """
        Builds a map of activities that can be executed in parallel.
        
        This method processes the 'parallel_activities' dictionary, which maps a source 
        activity to a list of target activities that occur in parallel. It computes the 
        reciprocal relationships between all parallel targets to ensure a complete undirected 
        mapping. The result is stored in 'self.context.parallel_activity_map'.
        """
        parallel_map = defaultdict(set)
        for source_activity, parallel_targets in self.context.parallel_activities.items():
            for activity1 in parallel_targets:
                for activity2 in parallel_targets:
                    if activity1 != activity2:
                        parallel_map[activity1].add(activity2)
                        parallel_map[activity2].add(activity1)
                parallel_map[source_activity].add(activity1)
                parallel_map[activity1].add(source_activity)
        self.context.parallel_activity_map = parallel_map
        
    def _process_decision_points(self) -> None:
        """
        Processes and normalizes decision point probabilities for internal mapping.
        
        The parser provides decision points where keys might be complex string representations 
        (like sets or dictionaries) representing the source of the decision. This method 
        cleans these keys into sanitized activity names and stores the resulting probabilities 
        in 'self.context.processed_decision_points', mapping a single activity to its branch probabilities.
        """
        self.context.processed_decision_points = {}
        for key_str, probs in self.context.decision_points.items():
            if isinstance(probs, dict):
                if isinstance(key_str, str):
                    activity_name = self._extract_activity_name(key_str)
                    if activity_name:
                        self.context.processed_decision_points[activity_name] = probs
            elif isinstance(key_str, str) and not key_str.startswith("{"):
                self.context.processed_decision_points[key_str] = probs
            elif isinstance(key_str, str) and key_str.startswith("{"):
                try:
                    source_part = key_str.split(",")[0].replace("{(", "").replace("{'", "").replace("'}", "")
                    sources = [s.strip().strip("'") for s in source_part.split(",")]
                    for source in sources:
                        source_activity = utils.sanitize_name(source)
                        if source_activity:
                            self.context.processed_decision_points[source_activity] = probs
                except Exception as e:
                    print(f"Error processing complex key {key_str}: {e}")
                    
    def _extract_activity_name(self, key_str: str) -> Optional[str]:
        """
        Extracts a valid activity name from a potentially complex dictionary key string.
        
        Args:
            key_str (str): A string that might contain an activity name mixed with other characters.
            
        Returns:
            Optional[str]: The sanitized activity name if found in the known activities set, otherwise None.
        """
        key_lower = str(key_str).lower()
        for activity in self.context.activities:
            if activity is not None and activity.lower() in key_lower:
                return activity
        return None

    # Properties to maintain backward compatibility for direct attribute access
    @property
    def domain_name(self): return self.context.domain_name
    @property
    def init(self): return self.context.init
    @property
    def goal(self): return self.context.goal
    @property
    def min_confidence(self): return self.context.min_confidence
    @property
    def activities(self): return self.context.activities
    @property
    def attribute_categories(self): return self.context.attribute_categories
    @property
    def _value_mappings(self): return self.context.value_mappings
    @property
    def attr_activity_relationships(self): return self.context.attr_activity_relationships
    @property
    def activity_attr_effects(self): return self.context.activity_attr_effects
    @property
    def tau_activities(self): return self.context.tau_activities
    @property
    def decision_points(self): return self.context.decision_points
    @property
    def decision_preconditions(self): return self.context.decision_preconditions
    @property
    def decision_attributes(self): return self.context.decision_attributes
    @property
    def parallel_activities(self): return self.context.parallel_activities
    @property
    def parallel_activity_map(self): return self.context.parallel_activity_map
    @property
    def split_actions(self): return self.context.split_actions
    @property
    def effect_alternatives(self): return self.context.effect_alternatives
    @property
    def processed_decision_points(self): return self.context.processed_decision_points
    @property
    def decision_points_conditions(self): return self.context.decision_points_conditions
    @property
    def minimal_preconditions(self): return self.context.minimal_preconditions
