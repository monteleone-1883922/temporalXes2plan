import os
import re
import sys
import subprocess
from typing import Dict, List, Optional, Tuple, Any

# Import core utilities
from core_utils import get_logger, sanitize_name

# Import base activity extractor from state builder to keep logic clean
from encoding.pddl_helper import extract_base_activity_name

logger = get_logger(__name__)

# Solvability status constants
SOLVABILITY_SOLVED = "solved"
SOLVABILITY_UNSOLVABLE_STRUCTURAL = "unsolvable_structural"
SOLVABILITY_UNSOLVABLE_RESOURCE = "unsolvable_resource"


def filter_tau_activities_from_plan(plan_content: str) -> Tuple[str, List[str]]:
    """
    Filters out tau (silent transition) activities from a plan output.

    Args:
        plan_content: Raw plan file contents as a string.

    Returns:
        A tuple of (filtered_plan_content_str, list_of_filtered_action_names).
    """
    if not plan_content:
        return "", []
    
    plan_lines = plan_content.strip().split('\n')
    filtered_plan_lines = []
    filtered_actions = []
    
    for line in plan_lines:
        if line.strip() and not line.strip().startswith(';'):
            action_part = line.strip().strip('()')
            if action_part:
                action_name = action_part.split()[0]
                if not action_name.startswith('tau_'):
                    filtered_plan_lines.append(line)
                    filtered_actions.append(action_name)
            else:
                filtered_plan_lines.append(line)
        else:
            filtered_plan_lines.append(line)
    
    return '\n'.join(filtered_plan_lines), filtered_actions


def parse_plan_actions(plan_content: str) -> List[str]:
    """
    Parse action names from a plan content string, filtering out tau activities.

    Args:
        plan_content: String content of a plan file.

    Returns:
        List of action name strings.
    """
    _, filtered_actions = filter_tau_activities_from_plan(plan_content)
    return filtered_actions


def parse_fast_downward_output(stdout: str, stderr: str) -> Dict[str, Optional[float]]:
    """
    Parse Fast Downward standard stdout and stderr output logs to extract search metrics.

    Args:
        stdout: Standard output log of Fast Downward.
        stderr: Standard error log of Fast Downward.

    Returns:
        A dictionary containing parsed floats or integers for search metrics:
        'expanded_nodes', 'space_used', 'solution_length', 'search_time', 'grounding_time', 'total_time'.
    """
    metrics = {
        'expanded_nodes': None,
        'space_used': None,
        'solution_length': None,
        'search_time': None,
        'grounding_time': None,
        'total_time': None
    }
    full_output = stdout + "\n" + stderr
    fd_stdout = ""
    fd_stderr = ""
    stdout_match = re.search(r'STDOUT:\s*\n(.*?)(?=STDERR:|---------------------------|\Z)', full_output, re.DOTALL)
    if stdout_match:
        fd_stdout = stdout_match.group(1)
    else:
        fd_stdout = stdout
        
    stderr_match = re.search(r'STDERR:\s*\n(.*?)(?=---------------------------|\Z)', full_output, re.DOTALL)
    if stderr_match:
        fd_stderr = stderr_match.group(1)
    else:
        fd_stderr = stderr
        
    fd_output = fd_stdout + "\n" + fd_stderr

    expanded_patterns = [
        r'Expanded (\d+) state\(s\)',
        r'Expanded nodes: (\d+)',
        r'(\d+) nodes expanded',
        r'Expansions: (\d+)'
    ]
    for pattern in expanded_patterns:
        match = re.search(pattern, fd_output)
        if match:
            metrics['expanded_nodes'] = int(match.group(1))
            break
    
    space_patterns = [
        r'Peak memory: ([\d.]+)\s*KB',
        r'Peak memory: ([\d.]+)\s*MB',
        r'Memory usage: ([\d.]+)\s*MB',
        r'Memory: ([\d.]+)\s*MB',
        r'(\d+)\s*KB peak memory'
    ]
    
    for pattern in space_patterns:
        match = re.search(pattern, fd_output)
        if match:
            value = float(match.group(1))
            if 'KB' in pattern:
                value = value / 1024.0  # Convert KB to MB
            metrics['space_used'] = value
            break
    
    length_patterns = [
        r'Plan length: (\d+) step\(s\)',
        r'Plan length: (\d+)',
        r'Solution length: (\d+)',
        r'(\d+) steps',
        r'Plan cost: (\d+)'
    ]
    
    for pattern in length_patterns:
        match = re.search(pattern, fd_output)
        if match:
            metrics['solution_length'] = int(match.group(1))
            break
    
    # Search time patterns
    search_time_patterns = [
        r'Search time: ([\d.]+)s',
        r'Actual search time: ([\d.]+)s',
        r'Total search time: ([\d.]+)s'
    ]
    for pattern in search_time_patterns:
        match = re.search(pattern, fd_output)
        if match:
            metrics['search_time'] = float(match.group(1))
            break
    
    # Grounding/translation time patterns
    grounding_time_patterns = [
        r'Done! \[([\d.]+)s CPU, ([\d.]+)s wall-clock\]',  # Translator output format
        r'Translator time: ([\d.]+)s',
        r'Translation time: ([\d.]+)s',
        r'Grounding time: ([\d.]+)s',
        r'Preprocessing time: ([\d.]+)s'
    ]
    for pattern in grounding_time_patterns:
        match = re.search(pattern, fd_output)
        if match:
            if 'wall-clock' in pattern:
                metrics['grounding_time'] = float(match.group(2))
            else:
                metrics['grounding_time'] = float(match.group(1))
            break
    
    if metrics['grounding_time'] is not None and metrics['search_time'] is not None:
        metrics['total_time'] = metrics['grounding_time'] + metrics['search_time']
    elif metrics['grounding_time'] is not None:
        metrics['total_time'] = metrics['grounding_time']
    elif metrics['search_time'] is not None:
        metrics['total_time'] = metrics['search_time']
    
    return metrics


def classify_planner_result(return_code: int, plan_exists: bool, stdout: str = "", stderr: str = "") -> str:
    """
    Classify the planner result into solvability categories based on Fast Downward exit codes.

    Args:
        return_code: Fast Downward process return code.
        plan_exists: Whether a valid plan file was written.
        stdout: Standard output log of Fast Downward.
        stderr: Standard error log of Fast Downward.

    Returns:
        One of: SOLVABILITY_SOLVED, SOLVABILITY_UNSOLVABLE_STRUCTURAL, SOLVABILITY_UNSOLVABLE_RESOURCE
    """
    full_output = stdout + "\n" + stderr
    
    if return_code == 0 and plan_exists:
        return SOLVABILITY_SOLVED
    
    # Exit code 11: SEARCH_UNSOLVABLE - Task is provably unsolvable
    if return_code == 11:
        return SOLVABILITY_UNSOLVABLE_STRUCTURAL
    
    if "Completely explored state space" in full_output:
        return SOLVABILITY_UNSOLVABLE_STRUCTURAL
    
    # Exit codes 20-24 indicate memory/time exhaustion
    if return_code in [20, 21, 22, 23, 24]:
        return SOLVABILITY_UNSOLVABLE_RESOURCE
    
    memory_indicators = ["out of memory", "memory exhausted", "Memory limit exceeded"]
    time_indicators = ["out of time", "timeout", "time limit exceeded"]
    
    for indicator in memory_indicators:
        if indicator.lower() in full_output.lower():
            return SOLVABILITY_UNSOLVABLE_RESOURCE
    
    for indicator in time_indicators:
        if indicator.lower() in full_output.lower():
            return SOLVABILITY_UNSOLVABLE_RESOURCE
    
    # Exit code 12: SEARCH_UNSOLVED_INCOMPLETE - incomplete search
    if return_code == 12:
        return SOLVABILITY_UNSOLVABLE_RESOURCE
    
    return SOLVABILITY_UNSOLVABLE_RESOURCE


def run_planner(
    plan_path: str, 
    search_algorithm: str = "astar_lmcut", 
    timeout: int = 30
) -> Tuple[bool, str, Dict[str, Any], str]:
    """
    Execute call_planner.py with Fast Downward and classify output.

    Args:
        plan_path: Destination path for the plan file.
        search_algorithm: Search algorithm to use.
        timeout: Subprocess execution timeout in seconds.

    Returns:
        A tuple of (success_bool, message_str, metrics_dict, solvability_status_str)
    """
    empty_metrics = {
        'expanded_nodes': None, 
        'space_used': None, 
        'solution_length': None, 
        'search_time': None, 
        'grounding_time': None, 
        'total_time': None
    }
    try:
        if os.path.exists(plan_path):
            os.remove(plan_path)
            
        script_dir = os.path.dirname(__file__)
        call_script_path = os.path.join(script_dir, 'call_planner.py')
        
        result = subprocess.run([
            sys.executable, call_script_path,
            '--search', search_algorithm
        ], capture_output=True, text=True, timeout=timeout)
        
        planning_metrics = parse_fast_downward_output(result.stdout, result.stderr)
        plan_exists = os.path.exists(plan_path)
        solvability = classify_planner_result(result.returncode, plan_exists, result.stdout, result.stderr)
        
        if solvability == SOLVABILITY_SOLVED:
            return True, "Plan found successfully", planning_metrics, solvability
        elif solvability == SOLVABILITY_UNSOLVABLE_STRUCTURAL:
            return False, f"Problem proved unsolvable (return code: {result.returncode})", planning_metrics, solvability
        else:
            return False, f"No solution found - resource limit (return code: {result.returncode})", planning_metrics, solvability
            
    except subprocess.TimeoutExpired:
        return False, "Planning timeout", empty_metrics, SOLVABILITY_UNSOLVABLE_RESOURCE
    except Exception as e:
        return False, f"Planning error: {str(e)}", empty_metrics, SOLVABILITY_UNSOLVABLE_RESOURCE


def parse_plan_file(plan_path: str) -> List[str]:
    """
    Parse PDDL plan file to extract sanitized sequential action names.

    Args:
        plan_path: Path to the generated .plan file.

    Returns:
        List of sanitized activity names (base activity names, excluding tau transitions).
    """
    if not os.path.exists(plan_path):
        return []
    
    actions = []
    try:
        with open(plan_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                
                if line.startswith('(') and line.endswith(')'):
                    parts = line[1:-1].split()
                    if parts:
                        action_name_full = parts[0]
                        
                        if action_name_full.startswith('exec_'):
                            action_name_full = action_name_full[5:]
                        
                        action_name_base = action_name_full.split('_DETDUP')[0]
                        action_name_base = action_name_base.split(' ')[0]
                        action_name_base = re.sub(r'_v\d+$', '', action_name_base)
                        action_name_base = sanitize_name(action_name_base)
                        
                        # Extract base activity name
                        action_name_base = extract_base_activity_name(action_name_base)
                        action_name_base = sanitize_name(action_name_base)
                        
                        if not action_name_base.startswith('tau_'):
                            actions.append(action_name_base)
    except Exception as e:
        logger.error(f"Error parsing plan file: {e}")
        return []
    
    return actions
