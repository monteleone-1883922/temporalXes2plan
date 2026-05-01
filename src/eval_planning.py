import os
import time
import csv
from typing import List, Dict, Optional, Any, Tuple, Set, Union

from datetime import datetime
from xes_parser import Parser
from pddl_encoder import Encoder
import utils


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the planning evaluation.

    Returns:
        The parsed arguments as a Namespace object.
    """
    parser = utils.create_base_argument_parser("Evaluate PDDL planner predictions quality")
    return parser.parse_args()


def evaluate_minimality(predicted_suffix: List[str], ground_truth_suffix: List[str]) -> Dict[str, Any]:
    """
    Evaluate the minimality of a predicted suffix compared to the ground truth.

    Args:
        predicted_suffix: The sequence of activities predicted by the planner.
        ground_truth_suffix: The actual remaining sequence of activities from the log.

    Returns:
        A dictionary containing length metrics and a boolean indicating if it's minimal.
    """
    """Evaluate minimality of predicted suffix compared to ground truth."""
    pred_len = len(predicted_suffix)
    gt_len = len(ground_truth_suffix)
    
    if gt_len == 0:
        ratio = float('inf') if pred_len > 0 else 1.0
        is_minimal = pred_len == 0
    else:
        ratio = pred_len / gt_len
        is_minimal = pred_len <= gt_len
    
    return {
        'predicted_length': pred_len,
        'ground_truth_length': gt_len,
        'length_ratio': ratio,
        'length_difference': pred_len - gt_len,
        'is_minimal': is_minimal
    }


def main() -> None:
    """Main entry point for the planning quality evaluation script."""
    print("Initialization of quality evaluation for PDDL planner predictions...")
    args = parse_arguments()
    pddl_name = args.xes_name.split('.')[0]
    print(f"Using search algorithm: {args.search}")
    print(f"Using discovery algorithm: {args.discovery_algorithm}")
    script_dir = os.path.dirname(__file__)
    domain_file_path = os.path.join(script_dir, '..', 'pddl', f'domain.pddl')
    problem_file_path = os.path.join(script_dir, '..', 'pddl', f'problem.pddl')
    parser = Parser(args.xes_name, args.log_coverage, args.discovery_algorithm,
                     use_activity_classifier=args.use_activity_classifier)
    test_traces, test_traces_full = utils.prepare_test_data(parser, args.max_traces)
    
    evaluation_results = []
    overall_stats = {'total_samples': 0,
                     'successful_plans': 0,
                     'reaching_target': 0,
                     'minimal_plans': 0,
                     'solved_count': 0,
                     'unsolvable_structural_count': 0,
                     'unsolvable_resource_count': 0,
                     'total_length_diff': 0,
                     'total_length_ratio': 0.0,
                     'total_planning_time': 0.0,
                     'total_expanded_nodes': 0,
                     'total_space_used': 0.0,
                     'total_solution_length': 0,
                     'total_search_time': 0.0,
                     'total_grounding_time': 0.0,
                     'total_fd_total_time': 0.0,
                     'expanded_nodes_count': 0,
                     'space_used_count': 0,
                     'solution_length_count': 0,
                     'search_time_count': 0,
                     'grounding_time_count': 0,
                     'fd_total_time_count': 0}
    
    prefix_length_stats = {}
    prefix_lengths_to_test = utils.calculate_all_prefix_lengths(test_traces)
    print(f"Testing on {len(test_traces)} traces with all possible prefix lengths: {prefix_lengths_to_test}")
    
    # Generate domain once for this evaluation run (log + algorithm combination)
    # Use a dummy encoder just to generate the domain
    dummy_encoder = Encoder(parser, pddl_name, init=[], goal=[])
    dummy_encoder.generate_domain(output_path=domain_file_path)
    print(f"Domain file generated: {domain_file_path}")
    
    # Set up incremental CSV output with resume support
    evaluation_dir = utils.create_evaluation_directories(script_dir)
    filename = utils.generate_evaluation_filename(pddl_name, args.log_coverage, args.search, 'plan_quality')
    results_file = os.path.join(evaluation_dir, filename)
    
    # Check for existing results to resume from
    processed_samples = set()
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Create unique key from trace_id + prefix_length
                    key = (int(row['trace_id']), int(row['prefix_length']))
                    processed_samples.add(key)
            print(f"Resuming: Found {len(processed_samples)} already processed samples in {results_file}")
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}. Starting fresh.")
            processed_samples = set()
    
    # Open CSV file for appending (or create with header if new)
    csv_file_exists = os.path.exists(results_file) and len(processed_samples) > 0
    csv_file = open(results_file, 'a' if csv_file_exists else 'w', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_file_exists:
        csv_writer.writerow(["trace_id", "prefix_length", "ground_truth_length", "predicted_length", 
                            "length_difference", "length_ratio", "planner_success", "solvability",
                            "reaches_target", "is_minimal", "planning_time", "search_time",
                            "grounding_time", "fd_total_time", "expanded_nodes", "space_used",
                            "solution_length", "search_algorithm", "discovery_algorithm",
                            "full_trace", "predicted_suffix", "valid_completion"])
        csv_file.flush()
    
    stop_processing = False
    for trace_idx, (trace, trace_events) in enumerate(zip(test_traces, test_traces_full)):
        if stop_processing:
            break
        sanitized_trace = [utils.sanitize_name(event) for event in trace]
        if not sanitized_trace:
            continue
            
        print(f"\n--- Processing Trace {trace_idx + 1}/{len(test_traces)} ---")
        print(f"Full trace: {' -> '.join(sanitized_trace)}")
        
        last_event = sanitized_trace[-1]
        for prefix_len in prefix_lengths_to_test:
            if len(sanitized_trace) <= prefix_len:
                continue
            
            # Check if this sample was already processed (for resume)
            sample_key = (trace_idx, prefix_len)
            if sample_key in processed_samples:
                print(f"  Skipping prefix length {prefix_len}: already processed")
                continue
                
            prefix = sanitized_trace[:prefix_len]
            ground_truth_suffix = sanitized_trace[prefix_len:]
            if last_event in prefix:
                print(f"  Skipping prefix length {prefix_len}: target '{last_event}' already completed in prefix")
                continue
            
            print(f"\n  Prefix length {prefix_len}: {' -> '.join(prefix)}")
            print(f"  Ground truth suffix: {' -> '.join(ground_truth_suffix)}")
            print(f"  Target end activity: {last_event}")
            init_condition = utils.compute_initial_state_from_last_activity(parser, prefix, trace_events, prefix_len)
            goal_condition = utils.compute_goal_condition(parser, last_event, trace_events, len(trace_events))
            print(f"  Goal condition: {goal_condition}")
            # Use minimal preconditions in planning evaluation to test suffix-only execution
            encoder = Encoder(parser,
                              pddl_name,
                              init=init_condition,
                              goal=goal_condition,
                              minimal_preconditions=True)
            
            problem_name = f"{pddl_name}_trace{trace_idx}_prefix{prefix_len}"
            encoder.generate_problem(problem_name=problem_name, output_path=problem_file_path)
            
            plan_file_path = os.path.join(script_dir, '..', 'pddl', 'plan_problem.txt')
            
            start_time = time.time()
            planner_success, planner_message, planning_metrics, solvability = utils.run_planner(plan_file_path, args.search)
            planning_time = time.time() - start_time
            
            result = {
                'trace_id': trace_idx,
                'prefix_length': prefix_len,
                'full_trace': sanitized_trace,
                'prefix': prefix,
                'ground_truth_suffix': ground_truth_suffix,
                'target_end_activity': last_event,
                'planning_time': planning_time,
                'planner_success': planner_success,
                'planner_message': planner_message,
                'solvability': solvability,
                'predicted_suffix': [],
                'reaches_target': False,
                'valid_completion': '',
                'minimality_metrics': {},
                'expanded_nodes': planning_metrics.get('expanded_nodes'),
                'space_used': planning_metrics.get('space_used'),
                'solution_length': planning_metrics.get('solution_length'),
                'search_time': planning_metrics.get('search_time'),
                'grounding_time': planning_metrics.get('grounding_time'),
                'fd_total_time': planning_metrics.get('total_time')
            }
            
            overall_stats['total_samples'] += 1
            overall_stats['total_planning_time'] += planning_time

            if planning_metrics.get('expanded_nodes') is not None:
                overall_stats['total_expanded_nodes'] += planning_metrics['expanded_nodes']
                overall_stats['expanded_nodes_count'] += 1
            
            if planning_metrics.get('space_used') is not None:
                overall_stats['total_space_used'] += planning_metrics['space_used']
                overall_stats['space_used_count'] += 1
                
            if planning_metrics.get('solution_length') is not None:
                overall_stats['total_solution_length'] += planning_metrics['solution_length']
                overall_stats['solution_length_count'] += 1
            
            if planning_metrics.get('search_time') is not None:
                overall_stats['total_search_time'] += planning_metrics['search_time']
                overall_stats['search_time_count'] += 1
            
            if planning_metrics.get('grounding_time') is not None:
                overall_stats['total_grounding_time'] += planning_metrics['grounding_time']
                overall_stats['grounding_time_count'] += 1
            
            if planning_metrics.get('total_time') is not None:
                overall_stats['total_fd_total_time'] += planning_metrics['total_time']
                overall_stats['fd_total_time_count'] += 1

            if prefix_len not in prefix_length_stats:
                prefix_length_stats[prefix_len] = {
                    'total_time': 0.0,
                    'count': 0,
                    'successful_count': 0,
                    'reaching_target_count': 0,
                    'solved_count': 0,
                    'unsolvable_structural_count': 0,
                    'unsolvable_resource_count': 0,
                    'total_expanded_nodes': 0,
                    'total_space_used': 0.0,
                    'total_solution_length': 0,
                    'total_search_time': 0.0,
                    'total_grounding_time': 0.0,
                    'total_fd_total_time': 0.0,
                    'expanded_nodes_count': 0,
                    'space_used_count': 0,
                    'solution_length_count': 0,
                    'search_time_count': 0,
                    'grounding_time_count': 0,
                    'fd_total_time_count': 0
                }
            prefix_length_stats[prefix_len]['total_time'] += planning_time
            prefix_length_stats[prefix_len]['count'] += 1
            
            # Track solvability
            if solvability == utils.SOLVABILITY_SOLVED:
                overall_stats['solved_count'] += 1
                prefix_length_stats[prefix_len]['solved_count'] += 1
            elif solvability == utils.SOLVABILITY_UNSOLVABLE_STRUCTURAL:
                overall_stats['unsolvable_structural_count'] += 1
                prefix_length_stats[prefix_len]['unsolvable_structural_count'] += 1
            elif solvability == utils.SOLVABILITY_UNSOLVABLE_RESOURCE:
                overall_stats['unsolvable_resource_count'] += 1
                prefix_length_stats[prefix_len]['unsolvable_resource_count'] += 1
            
            if planning_metrics.get('expanded_nodes') is not None:
                prefix_length_stats[prefix_len]['total_expanded_nodes'] += planning_metrics['expanded_nodes']
                prefix_length_stats[prefix_len]['expanded_nodes_count'] += 1
            
            if planning_metrics.get('space_used') is not None:
                prefix_length_stats[prefix_len]['total_space_used'] += planning_metrics['space_used']
                prefix_length_stats[prefix_len]['space_used_count'] += 1
                
            if planning_metrics.get('solution_length') is not None:
                prefix_length_stats[prefix_len]['total_solution_length'] += planning_metrics['solution_length']
                prefix_length_stats[prefix_len]['solution_length_count'] += 1
            
            if planning_metrics.get('search_time') is not None:
                prefix_length_stats[prefix_len]['total_search_time'] += planning_metrics['search_time']
                prefix_length_stats[prefix_len]['search_time_count'] += 1
            
            if planning_metrics.get('grounding_time') is not None:
                prefix_length_stats[prefix_len]['total_grounding_time'] += planning_metrics['grounding_time']
                prefix_length_stats[prefix_len]['grounding_time_count'] += 1
            
            if planning_metrics.get('total_time') is not None:
                prefix_length_stats[prefix_len]['total_fd_total_time'] += planning_metrics['total_time']
                prefix_length_stats[prefix_len]['fd_total_time_count'] += 1
            
            if planner_success:
                overall_stats['successful_plans'] += 1
                prefix_length_stats[prefix_len]['successful_count'] += 1
                
                # The planner returns only the suffix (actions after the prefix state)
                # so we don't need to extract it - it's already just the suffix
                predicted_actions = utils.parse_plan_file(plan_file_path)
                predicted_suffix = predicted_actions  # The plan IS the suffix
                
                result['predicted_suffix'] = predicted_suffix
                
                print(f"  Predicted suffix: {' -> '.join(predicted_suffix)}")
                
                if predicted_suffix:
                    target_base_name = utils.extract_base_activity_name(last_event)
                    reaches_target = any(utils.extract_base_activity_name(utils.sanitize_name(action)) == target_base_name 
                                       for action in predicted_suffix)
                    
                    result['reaches_target'] = reaches_target
                    result['valid_completion'] = f"{reaches_target}"
                    
                    if reaches_target:
                        overall_stats['reaching_target'] += 1
                        prefix_length_stats[prefix_len]['reaching_target_count'] += 1
                    
                    print(f"  Reaches target: {reaches_target}")
                    
                    minimality = evaluate_minimality(predicted_suffix, ground_truth_suffix)
                    result['minimality_metrics'] = minimality
                    
                    if minimality['is_minimal']:
                        overall_stats['minimal_plans'] += 1
                    
                    overall_stats['total_length_diff'] += minimality['length_difference']
                    overall_stats['total_length_ratio'] += minimality['length_ratio']
                    
                    print(f"  Minimality - Length ratio: {minimality['length_ratio']:.2f}, "
                          f"Difference: {minimality['length_difference']}, "
                          f"Is minimal: {minimality['is_minimal']}")
            else:
                print(f"  Planner failed: {planner_message}")
                if solvability == utils.SOLVABILITY_UNSOLVABLE_STRUCTURAL:
                    try:
                        print("  Diagnostic Info: Structural unsolvable sample details:")
                        print(f"    Init predicates: {init_condition}")
                        enabled_from_last = utils.compute_enabled_activities_after_last_activity(parser, last_event)
                        print(f"    Enabled after last activity (direct/petri): {sorted(list(enabled_from_last))}")
                        reachable = utils.compute_reachable_activities_from_last_activity(parser, last_event, max_depth=5)
                        print(f"    Reachable activities from last activity via direct graph (depth 5): {sorted(list(reachable))}")
                        print(f"    Goal predicates: {goal_condition}")
                    except Exception as e:
                        print(f"    Diagnostic error: {e}")
            
            evaluation_results.append(result)
            
            # Write this result immediately to CSV
            minimality = result.get('minimality_metrics', {})
            csv_writer.writerow([
                result['trace_id'],
                result['prefix_length'],
                minimality.get('ground_truth_length', 0),
                minimality.get('predicted_length', 0),
                minimality.get('length_difference', 0),
                f"{minimality.get('length_ratio', 0):.3f}",
                result['planner_success'],
                result.get('solvability', 'unknown'),
                result['reaches_target'],
                minimality.get('is_minimal', False),
                f"{result['planning_time']:.3f}",
                result.get('search_time', '') if result.get('search_time') is not None else '',
                result.get('grounding_time', '') if result.get('grounding_time') is not None else '',
                result.get('fd_total_time', '') if result.get('fd_total_time') is not None else '',
                result.get('expanded_nodes', ''),
                result.get('space_used', ''),
                result.get('solution_length', ''),
                args.search,
                args.discovery_algorithm,
                ' -> '.join(result['full_trace']),
                ' -> '.join(result['predicted_suffix']),
                result['valid_completion']
            ])
            csv_file.flush()  # Ensure it's written to disk immediately

            # Stop processing if we reached the global sample cap
            if args.max_samples is not None and isinstance(args.max_samples, int) and args.max_samples > 0:
                if overall_stats['total_samples'] >= args.max_samples:
                    stop_processing = True
                    print(f"Reached global sample cap of {args.max_samples} samples. Stopping evaluation.")
                    break
    
    # Close the CSV file
    csv_file.close()
    
    # Re-read all results from CSV for metrics calculation (includes resumed data)
    all_results = []
    try:
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_results.append({
                    'trace_id': int(row['trace_id']),
                    'prefix_length': int(row['prefix_length']),
                    'ground_truth_length': int(row['ground_truth_length']),
                    'predicted_length': int(row['predicted_length']),
                    'length_difference': int(row['length_difference']),
                    'length_ratio': float(row['length_ratio']),
                    'planner_success': row['planner_success'] == 'True',
                    'solvability': row['solvability'],
                    'reaches_target': row['reaches_target'] == 'True',
                    'is_minimal': row['is_minimal'] == 'True',
                    'planning_time': float(row['planning_time']),
                    'search_time': float(row['search_time']) if row['search_time'] else None,
                    'grounding_time': float(row['grounding_time']) if row['grounding_time'] else None,
                    'fd_total_time': float(row['fd_total_time']) if row['fd_total_time'] else None,
                    'expanded_nodes': int(row['expanded_nodes']) if row['expanded_nodes'] else None,
                    'space_used': float(row['space_used']) if row['space_used'] else None,
                    'solution_length': int(row['solution_length']) if row['solution_length'] else None,
                    'full_trace': row['full_trace'],
                    'predicted_suffix': row['predicted_suffix']
                })
        print(f"Loaded {len(all_results)} total results for metrics calculation.")
    except Exception as e:
        print(f"Error reading results file for metrics: {e}")
        all_results = []
    
    # Recalculate overall stats from all_results
    overall_stats = {'total_samples': len(all_results),
                     'successful_plans': sum(1 for r in all_results if r['planner_success']),
                     'reaching_target': sum(1 for r in all_results if r['reaches_target']),
                     'minimal_plans': sum(1 for r in all_results if r['is_minimal']),
                     'solved_count': sum(1 for r in all_results if r['solvability'] == utils.SOLVABILITY_SOLVED),
                     'unsolvable_structural_count': sum(1 for r in all_results if r['solvability'] == utils.SOLVABILITY_UNSOLVABLE_STRUCTURAL),
                     'unsolvable_resource_count': sum(1 for r in all_results if r['solvability'] == utils.SOLVABILITY_UNSOLVABLE_RESOURCE),
                     'total_length_diff': sum(r['length_difference'] for r in all_results),
                     'total_length_ratio': sum(r['length_ratio'] for r in all_results),
                     'total_planning_time': sum(r['planning_time'] for r in all_results),
                     'total_expanded_nodes': sum(r['expanded_nodes'] for r in all_results if r['expanded_nodes'] is not None),
                     'total_space_used': sum(r['space_used'] for r in all_results if r['space_used'] is not None),
                     'total_solution_length': sum(r['solution_length'] for r in all_results if r['solution_length'] is not None),
                     'total_search_time': sum(r['search_time'] for r in all_results if r['search_time'] is not None),
                     'total_grounding_time': sum(r['grounding_time'] for r in all_results if r['grounding_time'] is not None),
                     'total_fd_total_time': sum(r['fd_total_time'] for r in all_results if r['fd_total_time'] is not None),
                     'expanded_nodes_count': sum(1 for r in all_results if r['expanded_nodes'] is not None),
                     'space_used_count': sum(1 for r in all_results if r['space_used'] is not None),
                     'solution_length_count': sum(1 for r in all_results if r['solution_length'] is not None),
                     'search_time_count': sum(1 for r in all_results if r['search_time'] is not None),
                     'grounding_time_count': sum(1 for r in all_results if r['grounding_time'] is not None),
                     'fd_total_time_count': sum(1 for r in all_results if r['fd_total_time'] is not None)}
    
    # Recalculate prefix_length_stats from all_results
    prefix_length_stats = {}
    for r in all_results:
        prefix_len = r['prefix_length']
        if prefix_len not in prefix_length_stats:
            prefix_length_stats[prefix_len] = {
                'total_time': 0.0,
                'count': 0,
                'successful_count': 0,
                'reaching_target_count': 0,
                'solved_count': 0,
                'unsolvable_structural_count': 0,
                'unsolvable_resource_count': 0,
                'total_expanded_nodes': 0,
                'total_space_used': 0.0,
                'total_solution_length': 0,
                'total_search_time': 0.0,
                'total_grounding_time': 0.0,
                'total_fd_total_time': 0.0,
                'expanded_nodes_count': 0,
                'space_used_count': 0,
                'solution_length_count': 0,
                'search_time_count': 0,
                'grounding_time_count': 0,
                'fd_total_time_count': 0
            }
        stats = prefix_length_stats[prefix_len]
        stats['total_time'] += r['planning_time']
        stats['count'] += 1
        if r['planner_success']:
            stats['successful_count'] += 1
        if r['reaches_target']:
            stats['reaching_target_count'] += 1
        if r['solvability'] == utils.SOLVABILITY_SOLVED:
            stats['solved_count'] += 1
        elif r['solvability'] == utils.SOLVABILITY_UNSOLVABLE_STRUCTURAL:
            stats['unsolvable_structural_count'] += 1
        elif r['solvability'] == utils.SOLVABILITY_UNSOLVABLE_RESOURCE:
            stats['unsolvable_resource_count'] += 1
        if r['expanded_nodes'] is not None:
            stats['total_expanded_nodes'] += r['expanded_nodes']
            stats['expanded_nodes_count'] += 1
        if r['space_used'] is not None:
            stats['total_space_used'] += r['space_used']
            stats['space_used_count'] += 1
        if r['solution_length'] is not None:
            stats['total_solution_length'] += r['solution_length']
            stats['solution_length_count'] += 1
        if r['search_time'] is not None:
            stats['total_search_time'] += r['search_time']
            stats['search_time_count'] += 1
        if r['grounding_time'] is not None:
            stats['total_grounding_time'] += r['grounding_time']
            stats['grounding_time_count'] += 1
        if r['fd_total_time'] is not None:
            stats['total_fd_total_time'] += r['fd_total_time']
            stats['fd_total_time_count'] += 1
    
    if overall_stats['total_samples'] > 0:
        success_rate = overall_stats['successful_plans'] / overall_stats['total_samples']
        target_rate = overall_stats['reaching_target'] / overall_stats['total_samples']
        minimality_rate = overall_stats['minimal_plans'] / overall_stats['total_samples']
        avg_length_diff = overall_stats['total_length_diff'] / overall_stats['total_samples']
        avg_length_ratio = overall_stats['total_length_ratio'] / overall_stats['total_samples']
        avg_planning_time = overall_stats['total_planning_time'] / overall_stats['total_samples']
        avg_expanded_nodes = (overall_stats['total_expanded_nodes'] / overall_stats['expanded_nodes_count']) if overall_stats['expanded_nodes_count'] > 0 else 0
        avg_space_used = (overall_stats['total_space_used'] / overall_stats['space_used_count']) if overall_stats['space_used_count'] > 0 else 0
        avg_solution_length = (overall_stats['total_solution_length'] / overall_stats['solution_length_count']) if overall_stats['solution_length_count'] > 0 else 0
        avg_search_time = (overall_stats['total_search_time'] / overall_stats['search_time_count']) if overall_stats['search_time_count'] > 0 else 0
        avg_grounding_time = (overall_stats['total_grounding_time'] / overall_stats['grounding_time_count']) if overall_stats['grounding_time_count'] > 0 else 0
        avg_fd_total_time = (overall_stats['total_fd_total_time'] / overall_stats['fd_total_time_count']) if overall_stats['fd_total_time_count'] > 0 else 0
        
        print(f"\n{'='*60}")
        print("QUALITY EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples evaluated: {overall_stats['total_samples']}")
        print(f"\n--- Solvability Statistics ---")
        solved_pct = (overall_stats['solved_count'] / overall_stats['total_samples'] * 100) if overall_stats['total_samples'] > 0 else 0
        struct_pct = (overall_stats['unsolvable_structural_count'] / overall_stats['total_samples'] * 100) if overall_stats['total_samples'] > 0 else 0
        resource_pct = (overall_stats['unsolvable_resource_count'] / overall_stats['total_samples'] * 100) if overall_stats['total_samples'] > 0 else 0
        print(f"Solved: {overall_stats['solved_count']} ({solved_pct:.2f}%)")
        print(f"Unsolvable (Structural): {overall_stats['unsolvable_structural_count']} ({struct_pct:.2f}%)")
        print(f"Unsolvable (Resource): {overall_stats['unsolvable_resource_count']} ({resource_pct:.2f}%)")
        print(f"\n--- Quality Metrics ---")
        print(f"Planner success rate: {success_rate:.2%} ({overall_stats['successful_plans']}/{overall_stats['total_samples']})")
        print(f"Target reaching rate: {target_rate:.2%} ({overall_stats['reaching_target']}/{overall_stats['total_samples']})")
        print(f"Minimality rate: {minimality_rate:.2%} ({overall_stats['minimal_plans']}/{overall_stats['total_samples']})")
        print(f"Average length difference: {avg_length_diff:.2f}")
        print(f"Average length ratio: {avg_length_ratio:.2f}")
        print(f"Average planning time: {avg_planning_time:.4f} seconds")
        print(f"Average search time: {avg_search_time:.4f} seconds (available in {overall_stats['search_time_count']}/{overall_stats['total_samples']} samples)")
        print(f"Average grounding time: {avg_grounding_time:.4f} seconds (available in {overall_stats['grounding_time_count']}/{overall_stats['total_samples']} samples)")
        print(f"Average FD total time: {avg_fd_total_time:.4f} seconds (available in {overall_stats['fd_total_time_count']}/{overall_stats['total_samples']} samples)")
        print(f"Average expanded nodes: {avg_expanded_nodes:.0f} (available in {overall_stats['expanded_nodes_count']}/{overall_stats['total_samples']} samples)")
        print(f"Average space used: {avg_space_used:.2f} MB (available in {overall_stats['space_used_count']}/{overall_stats['total_samples']} samples)")
        print(f"Average solution length: {avg_solution_length:.1f} (available in {overall_stats['solution_length_count']}/{overall_stats['total_samples']} samples)")
        
        quality_score = (target_rate + minimality_rate) / 2
        print(f"Combined quality score: {quality_score:.2%}")
        
        print(f"\n{'='*60}")
        print("PER-PREFIX LENGTH STATISTICS")
        print(f"{'='*60}")
        for prefix_len in sorted(prefix_length_stats.keys()):
            stats = prefix_length_stats[prefix_len]
            avg_time = stats['total_time'] / stats['count']
            success_rate_prefix = stats['successful_count'] / stats['count'] if stats['count'] > 0 else 0
            target_rate_prefix = stats['reaching_target_count'] / stats['count'] if stats['count'] > 0 else 0
            avg_expanded_nodes_prefix = (stats['total_expanded_nodes'] / stats['expanded_nodes_count']) if stats['expanded_nodes_count'] > 0 else 0
            avg_space_used_prefix = (stats['total_space_used'] / stats['space_used_count']) if stats['space_used_count'] > 0 else 0
            avg_solution_length_prefix = (stats['total_solution_length'] / stats['solution_length_count']) if stats['solution_length_count'] > 0 else 0
            avg_search_time_prefix = (stats['total_search_time'] / stats['search_time_count']) if stats['search_time_count'] > 0 else 0
            avg_grounding_time_prefix = (stats['total_grounding_time'] / stats['grounding_time_count']) if stats['grounding_time_count'] > 0 else 0
            avg_fd_total_time_prefix = (stats['total_fd_total_time'] / stats['fd_total_time_count']) if stats['fd_total_time_count'] > 0 else 0
            
            print(f"Prefix Length {prefix_len:2d}: "
                  f"Samples={stats['count']:3d}, "
                  f"Success={success_rate_prefix:.1%}, "
                  f"Target={target_rate_prefix:.1%}, "
                  f"Avg Time={avg_time:.4f}s")
            print(f"                    "
                  f"Solved={stats['solved_count']}, "
                  f"Structural={stats['unsolvable_structural_count']}, "
                  f"Resource={stats['unsolvable_resource_count']}")
            if stats['expanded_nodes_count'] > 0 or stats['space_used_count'] > 0 or stats['solution_length_count'] > 0:
                print(f"                    "
                      f"Avg Nodes={avg_expanded_nodes_prefix:.0f}, "
                      f"Avg Space={avg_space_used_prefix:.2f}MB, "
                      f"Avg Sol.Len={avg_solution_length_prefix:.1f}")
            if stats['search_time_count'] > 0 or stats['grounding_time_count'] > 0 or stats['fd_total_time_count'] > 0:
                print(f"                    "
                      f"Avg Search={avg_search_time_prefix:.4f}s, "
                      f"Avg Grounding={avg_grounding_time_prefix:.4f}s, "
                      f"Avg FD Total={avg_fd_total_time_prefix:.4f}s")
    
    metrics_filename = utils.generate_evaluation_filename(pddl_name, args.log_coverage, args.search, 'plan_quality_metrics')
    metrics_filename = metrics_filename.replace('.csv', '.txt')
    metrics_file = os.path.join(evaluation_dir, metrics_filename)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Saving aggregate metrics to: {metrics_file}")
    with open(metrics_file, 'w') as f:
        f.write(f"Plan Quality Evaluation - Aggregate Metrics\n")
        f.write(f"{'='*50}\n")
        f.write(f"Dataset: {pddl_name}\n")
        f.write(f"Log Coverage: {args.log_coverage}\n")
        f.write(f"Search Algorithm: {args.search}\n")
        f.write(f"Discovery Algorithm: {args.discovery_algorithm}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {overall_stats['total_samples']}\n")
        f.write(f"\n")
        
        if stop_processing:
            print("Exiting early due to sample cap.")

        if overall_stats['total_samples'] > 0:
            success_rate = overall_stats['successful_plans'] / overall_stats['total_samples']
            target_rate = overall_stats['reaching_target'] / overall_stats['total_samples']
            minimality_rate = overall_stats['minimal_plans'] / overall_stats['total_samples']
            avg_length_diff = overall_stats['total_length_diff'] / overall_stats['total_samples']
            avg_length_ratio = overall_stats['total_length_ratio'] / overall_stats['total_samples']
            avg_planning_time = overall_stats['total_planning_time'] / overall_stats['total_samples']
            quality_score = (target_rate + minimality_rate) / 2
            avg_expanded_nodes = (overall_stats['total_expanded_nodes'] / overall_stats['expanded_nodes_count']) if overall_stats['expanded_nodes_count'] > 0 else 0
            avg_space_used = (overall_stats['total_space_used'] / overall_stats['space_used_count']) if overall_stats['space_used_count'] > 0 else 0
            avg_solution_length = (overall_stats['total_solution_length'] / overall_stats['solution_length_count']) if overall_stats['solution_length_count'] > 0 else 0
            avg_search_time = (overall_stats['total_search_time'] / overall_stats['search_time_count']) if overall_stats['search_time_count'] > 0 else 0
            avg_grounding_time = (overall_stats['total_grounding_time'] / overall_stats['grounding_time_count']) if overall_stats['grounding_time_count'] > 0 else 0
            avg_fd_total_time = (overall_stats['total_fd_total_time'] / overall_stats['fd_total_time_count']) if overall_stats['fd_total_time_count'] > 0 else 0
            
            # Calculate solvability percentages for file output
            solved_pct = (overall_stats['solved_count'] / overall_stats['total_samples'] * 100) if overall_stats['total_samples'] > 0 else 0
            struct_pct = (overall_stats['unsolvable_structural_count'] / overall_stats['total_samples'] * 100) if overall_stats['total_samples'] > 0 else 0
            resource_pct = (overall_stats['unsolvable_resource_count'] / overall_stats['total_samples'] * 100) if overall_stats['total_samples'] > 0 else 0
            
            f.write(f"SOLVABILITY STATISTICS\n")
            f.write(f"{'-'*25}\n")
            f.write(f"Solved: {overall_stats['solved_count']} ({solved_pct:.2f}%)\n")
            f.write(f"Unsolvable (Structural): {overall_stats['unsolvable_structural_count']} ({struct_pct:.2f}%)\n")
            f.write(f"Unsolvable (Resource): {overall_stats['unsolvable_resource_count']} ({resource_pct:.2f}%)\n")
            f.write(f"\n")
            
            f.write(f"OVERALL METRICS\n")
            f.write(f"{'-'*20}\n")
            f.write(f"Planner Success Rate: {success_rate:.2%} ({overall_stats['successful_plans']}/{overall_stats['total_samples']})\n")
            f.write(f"Target Reaching Rate: {target_rate:.2%} ({overall_stats['reaching_target']}/{overall_stats['total_samples']})\n")
            f.write(f"Minimality Rate: {minimality_rate:.2%} ({overall_stats['minimal_plans']}/{overall_stats['total_samples']})\n")
            f.write(f"Average Length Difference: {avg_length_diff:.2f}\n")
            f.write(f"Average Length Ratio: {avg_length_ratio:.2f}\n")
            f.write(f"Average Planning Time: {avg_planning_time:.4f} seconds\n")
            f.write(f"Average Search Time: {avg_search_time:.4f} seconds (available in {overall_stats['search_time_count']}/{overall_stats['total_samples']} samples)\n")
            f.write(f"Average Grounding Time: {avg_grounding_time:.4f} seconds (available in {overall_stats['grounding_time_count']}/{overall_stats['total_samples']} samples)\n")
            f.write(f"Average FD Total Time: {avg_fd_total_time:.4f} seconds (available in {overall_stats['fd_total_time_count']}/{overall_stats['total_samples']} samples)\n")
            f.write(f"Average Expanded Nodes: {avg_expanded_nodes:.0f} (available in {overall_stats['expanded_nodes_count']}/{overall_stats['total_samples']} samples)\n")
            f.write(f"Average Space Used: {avg_space_used:.2f} MB (available in {overall_stats['space_used_count']}/{overall_stats['total_samples']} samples)\n")
            f.write(f"Average Solution Length: {avg_solution_length:.1f} (available in {overall_stats['solution_length_count']}/{overall_stats['total_samples']} samples)\n")
            f.write(f"Combined Quality Score: {quality_score:.2%}\n")
            f.write(f"\n")
            
            f.write(f"PER-PREFIX LENGTH METRICS\n")
            f.write(f"{'-'*25}\n")
            for prefix_len in sorted(prefix_length_stats.keys()):
                stats = prefix_length_stats[prefix_len]
                avg_time = stats['total_time'] / stats['count']
                success_rate_prefix = stats['successful_count'] / stats['count'] if stats['count'] > 0 else 0
                target_rate_prefix = stats['reaching_target_count'] / stats['count'] if stats['count'] > 0 else 0
                avg_expanded_nodes_prefix = (stats['total_expanded_nodes'] / stats['expanded_nodes_count']) if stats['expanded_nodes_count'] > 0 else 0
                avg_space_used_prefix = (stats['total_space_used'] / stats['space_used_count']) if stats['space_used_count'] > 0 else 0
                avg_solution_length_prefix = (stats['total_solution_length'] / stats['solution_length_count']) if stats['solution_length_count'] > 0 else 0
                avg_search_time_prefix = (stats['total_search_time'] / stats['search_time_count']) if stats['search_time_count'] > 0 else 0
                avg_grounding_time_prefix = (stats['total_grounding_time'] / stats['grounding_time_count']) if stats['grounding_time_count'] > 0 else 0
                avg_fd_total_time_prefix = (stats['total_fd_total_time'] / stats['fd_total_time_count']) if stats['fd_total_time_count'] > 0 else 0
                
                f.write(f"Prefix Length {prefix_len:2d}: ")
                f.write(f"Samples={stats['count']:3d}, ")
                f.write(f"Success={success_rate_prefix:.1%}, ")
                f.write(f"Target={target_rate_prefix:.1%}, ")
                f.write(f"Avg Time={avg_time:.4f}s, ")
                f.write(f"Solved={stats['solved_count']}, ")
                f.write(f"Structural={stats['unsolvable_structural_count']}, ")
                f.write(f"Resource={stats['unsolvable_resource_count']}, ")
                f.write(f"Avg Search={avg_search_time_prefix:.4f}s, ")
                f.write(f"Avg Grounding={avg_grounding_time_prefix:.4f}s, ")
                f.write(f"Avg FD Total={avg_fd_total_time_prefix:.4f}s, ")
                f.write(f"Avg Nodes={avg_expanded_nodes_prefix:.0f}, ")
                f.write(f"Avg Space={avg_space_used_prefix:.2f}MB, ")
                f.write(f"Avg Sol.Len={avg_solution_length_prefix:.1f}\n")
        else:
            f.write(f"No samples were evaluated.\n")
    
    print("Plan Quality evaluation completed!")


if __name__ == "__main__":
    main()
