import os
import random
from typing import Optional, List
from encoder_models import PDDLEncodingContext
from pddl_action_builder import ActionBuilder

class ProblemBuilder:
    """
    Responsible for generating the PDDL problem definition,
    including the initial state and the goal condition.
    """
    def __init__(self, context: PDDLEncodingContext, action_builder: ActionBuilder):
        self.context = context
        self.action_builder = action_builder

    def generate_problem(self, problem_name: str = "process_problem", output_path: Optional[str] = None) -> str:
        """
        Generate PDDL problem file with initial state and goal.
        
        Args:
            problem_name: Name of the problem
            output_path: Path to save the problem file
        """
        custom_init = self.context.init
        custom_goal = self.context.goal
        
        problem = [
            f"(define (problem {problem_name}_prediction)",
            f"  (:domain {self.context.domain_name})",
            ""
        ]
        
        problem.append(self._generate_init_section(custom_init))
        problem.append("")
        
        problem.append(self._generate_goal_section(custom_goal))
        problem.append("")
        problem.append(")")
        
        problem_content = "\n".join(problem)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(problem_content)
        
        return problem_content

    def _generate_init_section(self, custom_init: Optional[List[str]]) -> str:
        """
        Generates the `:init` section of the PDDL problem.
        
        The initial state defines the starting facts before any actions are executed.
        This method will include any custom initial predicates provided during initialization.
        If no custom predicates are provided, it automatically infers the starting condition by 
        enabling the first start activity discovered in the process log.
        
        Args:
            custom_init (Optional[List[str]]): A list of custom PDDL predicates to include in the initial state.
            
        Returns:
            str: The formatted PDDL `:init` block.
        """
        init = ["  (:init"]
        
        relevant_attrs = set()
        for attr in self.context.attr_activity_relationships:
            relevant_attrs.add(attr)
        for activity, effects in self.context.activity_attr_effects.items():
            for effect in effects:
                if 'attribute' in effect:
                    relevant_attrs.add(effect['attribute'])
        
        for activity in self.context.activities:
            variants = self.action_builder._generate_measurement_action_variants(activity)
            if variants:
                sanitized_action = activity.lower()
                matching_attribute = next((attr for attr in self.context.attribute_categories 
                                        if attr.lower() == sanitized_action), None)
                if matching_attribute:
                    relevant_attrs.add(matching_attribute)
        
        if custom_init:
            for condition in custom_init:
                init.append(f"    {condition}")
        else:
            if self.context.parser.start_activities:
                first_start_activity = next(iter(self.context.parser.start_activities))
                init.append(f"    (enabled {first_start_activity})")
        
        init.append("  )")
        return "\n".join(init)
    
    def _generate_goal_section(self, custom_goal: Optional[List[str]]) -> str:
        """
        Generates the `:goal` section of the PDDL problem.
        
        The goal state defines the condition that the planner must satisfy to successfully complete the problem.
        If custom goals are provided, they are added to the section. Otherwise, it falls back to 
        generating a default goal based on discovered end activities.
        
        Args:
            custom_goal (Optional[List[str]]): A list of custom PDDL predicates required in the goal state.
            
        Returns:
            str: The formatted PDDL `:goal` block.
        """
        goal = ["  (:goal", "    (and"]
        
        if custom_goal:
            for condition in custom_goal:
                goal.append(f"      {condition}")
        else:
            goal_condition = self._create_default_goal()
            goal.append(f"      {goal_condition}")
        
        goal.append("    ) )")
        return "\n".join(goal)

    def _create_default_goal(self) -> str:
        """
        Creates a default goal condition based on process end activities.
        
        Randomly selects one valid end activity from the set of discovered end activities 
        and creates a `(completed activity_name)` condition for it.
        
        Returns:
            str: The default goal string, or an empty tuple "()" if no valid end activities exist.
        """
        if self.context.parser.end_activities:
            valid_end_activities = [activity for activity in self.context.parser.end_activities.keys() if activity in self.context.activities]
            if valid_end_activities:
                random_end = random.choice(valid_end_activities)
                return f"(completed {random_end})"
        return "()"  # No valid end activities found
