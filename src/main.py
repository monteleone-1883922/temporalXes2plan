from argparse import ArgumentParser, Namespace
from typing import List, Dict, Optional, Any, Tuple, Set, Union
import os
from xes_parser import Parser
from pddl_encoder import Encoder
import subprocess
import pddl_helper as pddl_builder


def parse_arguments() -> Namespace:
    """
    Parse command-line arguments for the XES2PDDL framework.

    Returns:
        The parsed arguments as a Namespace object.
    """
    parser = ArgumentParser(description="Run the framework for XES encoding in PDDL.")
    parser.add_argument('--log_coverage', type=float, default='0.001', help='Minimum cumulative coverage percentage for variant filtering (pm4py).')
    parser.add_argument('--xes_name', type=str, default='sepsis.xes', help='Name of the XES event log to use.')
    parser.add_argument('--pddl_name', type=str, default='xes_log', help='Name of the PDDL domain to generate.')
    parser.add_argument('--init', type=str, default='../pddl/input/init.pddl', help='Path to the file containing initial state predicates.')
    parser.add_argument('--goal', type=str, default='../pddl/input/goal.pddl', help='Path to the file containing goal state predicates.')
    parser.add_argument('--discovery_algorithm', type=str, default='inductive',
                        choices=['alpha', 'inductive', 'heuristics', 'ilp'],
                        help='Discovery algorithm to use for Petri net discovery. '
                             'Options: alpha (Alpha Miner), inductive (Inductive Miner), '
                             'heuristics (Heuristics Miner), ilp (ILP Miner)')
    parser.add_argument('--use_activity_classifier', action='store_true',
                        help='Use Activity classifier (combine concept:name + lifecycle:transition) '
                             'to identify activities. Use this for logs where events are not '
                             'uniquely identified by concept:name alone (e.g., BPIC 2013).')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    print("Initialization of the PDDL encoding process for the XES event log...")
    args = parse_arguments()
    pddl_name = args.xes_name.split('.')[0]

    script_dir = os.path.dirname(__file__)
    init_file_path = os.path.join(script_dir, args.init)
    goal_file_path = os.path.join(script_dir, args.goal)
    domain_file_path = os.path.join(script_dir, '..', 'pddl', f'domain.pddl')
    problem_file_path = os.path.join(script_dir, '..', 'pddl', f'problem.pddl')

    parser = Parser(args.xes_name, args.log_coverage, args.discovery_algorithm,
                    use_activity_classifier=args.use_activity_classifier)

    custom_init = pddl_builder.read_state_file(init_file_path)
    custom_goal = pddl_builder.read_state_file(goal_file_path)
    if not custom_init:
        print("Using default empty initial state.")
    if not custom_goal:
        print("Using default empty goal state.")

    encoder = Encoder(parser,
                      pddl_name,
                      init=custom_init if custom_init else None,
                      goal=custom_goal if custom_goal else None)

    encoder.generate_domain(output_path=domain_file_path)
    print(f"Domain file generated: {domain_file_path}")

    encoder.generate_problem(problem_name=pddl_name, output_path=problem_file_path)
    print(f"Problem file generated: {problem_file_path}")

    call_script_path = os.path.join(script_dir, 'call_planner.py')
    print(f"Executing {call_script_path}...")
    try:
        subprocess.run(['python', call_script_path], check=True)
        print("call_planner.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing call_planner.py: {e}")
    
    plan_path = os.path.join(script_dir, '..', 'pddl', 'plan_problem.txt')
    print(f"Reading plan file: {plan_path}")
    try:
        with open(plan_path, 'r') as f:
            plan_content = f.read()
            print("\n--- Plan Content ---")
            print(plan_content)
            print("--- End of Plan ---")
    except FileNotFoundError:
        print(f"Error: Plan file not found at {plan_path}. Was the planner successful?")

    print("PDDL encoding and planning process finished.")