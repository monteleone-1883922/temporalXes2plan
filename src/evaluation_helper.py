import os
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pm4py

# Import core utilities
from core_utils import get_logger, sanitize_name

# Import find key helper from state builder
from pddl_helper import find_attribute_key_in_event

logger = get_logger(__name__)


def create_base_argument_parser(description: str = "Run evaluation framework") -> ArgumentParser:
    """
    Create a base ArgumentParser with common arguments for the evaluation framework.

    Args:
        description: A brief description of the parser's purpose.

    Returns:
        The initialized ArgumentParser instance.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument('--log_coverage', type=float, default=0.001, 
                        help='Minimum cumulative coverage percentage for variant filtering (pm4py).')
    parser.add_argument('--xes_name', type=str, default='sepsis.xes', 
                        help='Name of the XES event log to use.')
    parser.add_argument('--pddl_name', type=str, default='xes_log', 
                        help='Name of the PDDL domain to generate.')
    parser.add_argument('--init', type=str, default='../pddl/input/init.pddl', 
                        help='Path to the file containing initial state predicates.')
    parser.add_argument('--goal', type=str, default='../pddl/input/goal_test.pddl', 
                        help='Path to the file containing goal state predicates.')
    parser.add_argument('--search', default="astar_lmcut", type=str,
                        choices=["astar_blind", "astar_hadd", "astar_hff", "astar_lmcut",
                                "eager_greedy_blind", "eager_greedy_hadd", "eager_greedy_hff",
                                "eager_greedy_lmcut", "seq_sat_lama_2011", "seq_opt_bjolp", "lama_first"],
                        help="Search algorithm to use for planning")
    parser.add_argument('--discovery_algorithm', type=str, default='alpha',
                        choices=['alpha', 'inductive', 'heuristics', 'ilp'],
                        help='Discovery algorithm to use for Petri net discovery. '
                             'Options: alpha, inductive, heuristics, ilp')
    parser.add_argument('--use_activity_classifier', action='store_true',
                        help='Use Activity classifier (combine concept:name + lifecycle:transition).')
    parser.add_argument('--max-samples', type=int, default=700,
                        help='Maximum number of generated samples (prefixes, cases) to evaluate overall.')
    parser.add_argument('--max-traces', type=int, default=None,
                        help='Optional cap: maximum number of traces to read from the log.')
    return parser


def extract_numerical_attributes(parser: Any) -> List[str]:
    """
    Extract a list of numerical attribute names from the parser.

    Args:
        parser: The XES parser instance.

    Returns:
        List of numerical attribute names.
    """
    numerical_attrs = []
    if hasattr(parser, 'attribute_categories'):
        for attr, category in parser.attribute_categories.items():
            if category == 'numerical':
                numerical_attrs.append(attr)
    return numerical_attrs


def get_goal_attribute_value(parser: Any, event: Dict[str, Any], target_attr: str) -> Optional[float]:
    """
    Extract the numeric value of a target attribute from a goal event.

    Args:
        parser: The XES parser.
        event: The event dictionary.
        target_attr: The name of the target attribute.

    Returns:
        The numeric value as a float, or None.
    """
    trace_attr_key = find_attribute_key_in_event(event, target_attr)
    
    if trace_attr_key and trace_attr_key in event:
        raw_value = event[trace_attr_key]
        if raw_value is not None and not pd.isna(raw_value):
            try:
                return float(raw_value)
            except (ValueError, TypeError):
                pass
    
    return None


def create_evaluation_directories(script_dir: str) -> str:
    """
    Create the necessary directories for storing evaluation results.

    Args:
        script_dir: The directory containing the current script.

    Returns:
        The path to the created evaluation directory.
    """
    evaluation_dir = os.path.join(script_dir, '..', 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)
    return evaluation_dir


def generate_evaluation_filename(
    pddl_name: str, 
    log_coverage: float, 
    search_algorithm: str, 
    eval_type: str, 
    timestamp: Optional[str] = None
) -> str:
    """
    Generate a standardized filename for evaluation result CSVs.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coverage_str = f"{log_coverage:.6f}".rstrip('0').rstrip('.')
    return f'{eval_type}_eval_{pddl_name}_cov{coverage_str}_{search_algorithm}_{timestamp}.csv'


def calculate_all_prefix_lengths(test_traces: List[List[str]]) -> List[int]:
    """
    Calculate all unique prefix lengths present in a set of test traces.
    """
    all_prefix_lengths = set()
    for trace in test_traces:
        sanitized_trace = [sanitize_name(event) for event in trace]
        if sanitized_trace:
            for i in range(1, len(sanitized_trace)):
                all_prefix_lengths.add(i)
    
    return sorted(list(all_prefix_lengths))


def prepare_test_data(parser: Any, max_traces: Optional[int] = None) -> Tuple[List[List[str]], List[List[Dict[str, Any]]]]:
    """
    Prepare test data from the parsed test split for evaluation.

    Args:
        parser: The XES parser instance.
        max_traces: Optional maximum cap on trace count.

    Returns:
        Tuple of (test_traces, test_traces_with_events)
    """
    test_log = parser.test_df
    test_df = pm4py.convert_to_dataframe(test_log)
    
    test_traces = test_df.groupby('case:concept:name')['concept:name'].apply(list).tolist()
    
    test_traces_with_events = []
    for case_id in test_df['case:concept:name'].unique():
        case_events = test_df[test_df['case:concept:name'] == case_id].to_dict('records')
        test_traces_with_events.append(case_events)

    if max_traces is not None and isinstance(max_traces, int) and max_traces > 0:
        if len(test_traces) > max_traces:
            test_traces = test_traces[:max_traces]
            test_traces_with_events = test_traces_with_events[:max_traces]

    return test_traces, test_traces_with_events
