import argparse
import os
import sys
import subprocess
from typing import List, Dict, Optional, Any, Tuple, Set, Union

_module_dir = os.path.dirname(os.path.abspath(__file__))
DOWNWARD_SCRIPT = os.path.join(_module_dir, 'downward', 'fast-downward.py')
DOWNWARD_CMD = [sys.executable, DOWNWARD_SCRIPT]

def get_search_configs() -> Dict[str, Dict[str, str]]:
    """
    Return dictionary of supported search configurations for Fast Downward.

    Returns:
        Mapping of search configuration keys to their type and command-line string.
    """
    """Return dictionary of supported search configurations"""
    return {
        # Optimal (with reopening)
        "astar_blind": {"type": "search", "config": "astar(blind(), cost_type=one)"}, 
        "astar_hadd": {"type": "search", "config": "astar(add(), cost_type=one)"},
        "astar_hff": {"type": "search", "config": "astar(ff(), cost_type=one)"},
        "astar_lmcut": {"type": "search", "config": "astar(lmcut(), cost_type=one)"},
        
        # Satisficing
        "eager_greedy_blind": {"type": "search", "config": "eager_greedy([blind()], cost_type=one)"},
        "eager_greedy_hadd": {"type": "search", "config": "eager_greedy([add()], cost_type=one)"},
        "eager_greedy_hff": {"type": "search", "config": "eager_greedy([ff()], cost_type=one)"},
        "eager_greedy_lmcut": {"type": "search", "config": "eager_greedy([lmcut()], cost_type=one)"},
        "seq_sat_lama_2011": {"type": "alias", "config": "seq-sat-lama-2011"},
        "seq_opt_bjolp": {"type": "alias", "config": "seq-opt-bjolp"},
        "lama_first": {"type": "alias", "config": "lama-first"}
    }

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for running the planner.

    Returns:
        The parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description='Run Fast Downward planner with configurable search algorithms')
    search_configs = get_search_configs()
    search_help = "Search algorithm to use. Available options:\n"
    search_help += "OPTIMAL:\n"
    search_help += "  astar_blind: A* with blind heuristic\n"
    search_help += "  astar_hadd: A* with additive heuristic\n"
    search_help += "  astar_hff: A* with FF heuristic\n"
    search_help += "  astar_lmcut: A* with LM-cut heuristic (default)\n"
    search_help += "SATISFICING:\n"
    search_help += "  eager_greedy_blind: Eager greedy with blind heuristic\n"
    search_help += "  eager_greedy_hadd: Eager greedy with additive heuristic\n"
    search_help += "  eager_greedy_hff: Eager greedy with FF heuristic\n"
    search_help += "  eager_greedy_lmcut: Eager greedy with LM-cut heuristic\n"
    search_help += "ALIAS-BASED CONFIGURATIONS:\n"
    search_help += "  seq_sat_lama_2011: LAMA 2011 satisficing configuration\n"
    search_help += "  seq_opt_bjolp: Optimal BJOLP configuration\n"
    search_help += "  lama_first: LAMA first solution configuration\n"
    parser.add_argument('--search', 
                       choices=search_configs.keys(),
                       default='astar_lmcut',
                       help=search_help)
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout in seconds (default: 30)')
    
    return parser.parse_args()

def build_command(search_key: str) -> List[str]:
    """
    Build the full command-line list for executing Fast Downward.

    Args:
        search_key: The search configuration identifier (from get_search_configs).

    Returns:
        List of command-line arguments to be executed.
    """
    search_configs = get_search_configs()
    search_config = search_configs[search_key]
    
    if search_config["type"] == "alias":
        command = DOWNWARD_CMD + ["--alias", search_config["config"], "domain.pddl", "problem.pddl"]
    elif search_config["type"] == "search":
        base_command = DOWNWARD_CMD + ["domain.pddl", "problem.pddl"]
        if isinstance(search_config["config"], list):  # Multiple arguments
            command = base_command + search_config["config"]
        else: # Just --search
            command = base_command + ["--search", search_config["config"]]
    else:
        raise ValueError(f"Unknown configuration type: {search_config['type']}")
    
    return command

args = parse_arguments()

if not os.path.exists(DOWNWARD_SCRIPT):
    print("Error: Fast Downward script not found at: {}".format(DOWNWARD_SCRIPT))
    print("Please build Fast Downward:")
    print("  cd src/downward && ./build.py")
    sys.exit(1)

script_dir = os.path.dirname(__file__)
host_pddl_path = os.path.join(script_dir, '..', 'pddl')
command = build_command(args.search)
print(f"Target PDDL directory: {host_pddl_path}")
print(f"Selected search algorithm: {args.search}")

original_cwd = os.getcwd()
solution_found = False
try:
    os.chdir(host_pddl_path)
    print(f"\nRunning command: {' '.join(command)}")
    result = subprocess.run(command, 
                            capture_output=True, 
                            text=True, 
                            timeout=args.timeout)

    print("\n--- Command Output ---")
    print("Return Code:", result.returncode)
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    print("---------------------------\n")

    if result.returncode == 0 and "Solution found." in result.stdout:
        print("Planner found a solution.")
        
        plan_files = []
        for filename in os.listdir(host_pddl_path):
            if filename == 'sas_plan' or filename.startswith('sas_plan.'):
                plan_files.append(filename)
        
        if plan_files:
            plan_files.sort()
            source_plan_path = os.path.join(host_pddl_path, plan_files[0])
            target_plan_path = os.path.join(host_pddl_path, 'plan_problem.txt')
            with open(source_plan_path, 'r') as src, open(target_plan_path, 'w') as dst:
                dst.write(src.read())
            print(f"Plan saved to: {target_plan_path}")
            if len(plan_files) > 1:
                print(f"Note: Found {len(plan_files)} plan files. Used {plan_files[0]}.")
                print(f"All plan files: {', '.join(plan_files)}")
        else:
            print("Warning: Solution found but no plan file detected.")
        
        essential_files = {'plan_problem.txt', 'domain.pddl', 'problem.pddl'}
        cleanup_count = 0
        print("Cleaning up temporary files...")
        for filename in os.listdir(host_pddl_path):
            file_path = os.path.join(host_pddl_path, filename)
            if os.path.isfile(file_path) and filename not in essential_files:
                try:
                    os.remove(file_path)
                    cleanup_count += 1
                    print(f"Removed: {filename}")
                except Exception as e:
                    print(f"Warning: Could not remove {filename}: {e}")
        print(f"Cleanup completed. Removed {cleanup_count} temporary files.")
        solution_found = True
    elif result.returncode == 0:
        print("Planner finished but did not find a solution.")
    else:
        print(f"Planner failed with return code {result.returncode}.")
    
    if not solution_found:
        print("No solution found.")

except FileNotFoundError as e:
    print("Error: Command '{}' not found.".format(command[0]))
    print("Details: {}".format(e))
    print("Please build Fast Downward:")
    print("  cd src/downward && ./build.py")
    sys.exit(1)
except subprocess.TimeoutExpired:
    print(f"Error: Planner command timed out after {args.timeout} seconds.")
    sys.exit(1)
except OSError as e:
    if 'os.chdir' in str(e):
        print(f"Error: PDDL directory not found at expected location: {host_pddl_path}")
    else:
        print(f"An OS error occurred: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)
finally:
    os.chdir(original_cwd)

print("\nScript finished.")