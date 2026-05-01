from datetime import datetime
import os
import time
import csv
from typing import List, Dict, Optional, Any, Tuple, Set, Union

from jellyfish._jellyfish import damerau_levenshtein_distance

import utils
from xes_parser import Parser
from pddl_encoder import Encoder


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the outcome prediction evaluation.

    Returns:
        The parsed arguments as a Namespace object.
    """
    parser = utils.create_base_argument_parser(
        "Run the evaluation of the framework for XES encoding in PDDL for numerical outcome prediction."
    )
    parser.add_argument('--target_attribute', type=str, default=None, 
                        help='Specific numerical attribute to predict. If not specified, all numerical attributes will be evaluated.')
    return parser.parse_args()


def create_possible_goal_conditions(parser: Parser, target_attr: str) -> List[Tuple[List[str], str]]:
    """
    Create goal conditions for all possible discretized values of the target attribute.

    Args:
        parser: The XES parser containing attribute data and intervals.
        target_attr: The name of the numerical attribute to predict.

    Returns:
        A list of tuples, each containing (PDDL goal condition list, discretized value label).
    """
    """Create goal conditions for all possible discretized values of the target attribute."""
    goals = []
    
    if hasattr(parser, 'intervals') and target_attr in parser.intervals:
        thresholds = parser.intervals[target_attr]
        sanitized_attr = target_attr.lower().replace(':', '_').replace(' ', '_')
        
        if len(thresholds) < 2:
            goals.append(([f"({sanitized_attr} unknown)"], "unknown"))
        elif len(thresholds) <= 3:
            goals.append(([f"({sanitized_attr} low)"], "low"))
            goals.append(([f"({sanitized_attr} high)"], "high"))
        elif len(thresholds) <= 5:
            goals.append(([f"({sanitized_attr} low)"], "low"))
            goals.append(([f"({sanitized_attr} medium)"], "medium"))
            goals.append(([f"({sanitized_attr} high)"], "high"))
        else:
            goals.append(([f"({sanitized_attr} very_low)"], "very_low"))
            goals.append(([f"({sanitized_attr} low)"], "low"))
            goals.append(([f"({sanitized_attr} medium)"], "medium"))
            goals.append(([f"({sanitized_attr} high)"], "high"))
    
    return goals


def get_actual_discretized_value(parser: Parser, target_attr: str, actual_value: float) -> Optional[str]:
    """
    Get the discretized string representation of a raw numerical value.

    Args:
        parser: The XES parser containing discretization intervals.
        target_attr: The attribute name.
        actual_value: The raw numerical value.

    Returns:
        The discretized label (e.g., 'low', 'medium', 'high') or None if not available.
    """
    if hasattr(parser, 'intervals') and target_attr in parser.intervals:
        return utils.discretize_value(target_attr, actual_value, parser.intervals)
    return None


def discretized_to_numerical_estimate(parser: Parser, target_attr: str, discretized_value: str) -> Optional[float]:
    """
    Convert a discretized label back to a numerical estimate (e.g., center of interval).

    Args:
        parser: The XES parser containing discretization intervals.
        target_attr: The attribute name.
        discretized_value: The discretized label string.

    Returns:
        A numerical estimate (float) or None if not applicable.
    """
    if not discretized_value or not hasattr(parser, 'intervals') or target_attr not in parser.intervals:
        return None
    
    thresholds = parser.intervals[target_attr]
    if len(thresholds) < 2:
        return thresholds[0] if thresholds else 0
    
    if discretized_value == 'low':
        return (thresholds[0] + thresholds[1]) / 2
    elif discretized_value == 'high':
        if len(thresholds) >= 2:
            return (thresholds[-2] + thresholds[-1]) / 2
    elif discretized_value == 'medium':
        if len(thresholds) >= 3:
            mid_idx = len(thresholds) // 2
            return (thresholds[mid_idx-1] + thresholds[mid_idx]) / 2
    elif discretized_value == 'very_low':
        return thresholds[0]
    
    return (thresholds[0] + thresholds[-1]) / 2   # Default to middle of range


if __name__ == "__main__":
    print("Initialization of the PDDL encoding process for numerical outcome prediction...")
    args = parse_arguments()
    pddl_name = args.xes_name.split('.')[0]

    print(f"Using search algorithm: {args.search}")
    print(f"Using discovery algorithm: {args.discovery_algorithm}")

    script_dir = os.path.dirname(__file__)
    domain_file_path = os.path.join(script_dir, '..', 'pddl', f'domain.pddl')
    problem_file_path = os.path.join(script_dir, '..', 'pddl', f'problem.pddl')
    parser = Parser(args.xes_name, args.log_coverage, args.discovery_algorithm,
                     use_activity_classifier=args.use_activity_classifier)

    numerical_attrs = utils.extract_numerical_attributes(parser)
    print(f"Found numerical attributes: {numerical_attrs}")
    
    if not numerical_attrs:
        print("No numerical attributes found. Cannot perform outcome prediction.")
        exit(1)
    
    # Determine target attribute(s)
    if args.target_attribute:
        # Find matching attribute (case-insensitive)
        target_attrs = []
        target_lower = args.target_attribute.lower()
        for attr in numerical_attrs:
            if attr.lower() == target_lower:
                target_attrs.append(attr)
                break
        if not target_attrs:
            print(f"Target attribute '{args.target_attribute}' not found in numerical attributes. Available: {numerical_attrs}")
            exit(1)
    else:
        target_attrs = numerical_attrs
    
    if not target_attrs:
        print(f"Target attribute(s) not found in numerical attributes. Available: {numerical_attrs}")
        exit(1)
    
    print(f"Evaluating outcome prediction for attributes: {target_attrs}")

    test_traces, test_traces_with_events = utils.prepare_test_data(parser, args.max_traces)
    
    # Set up incremental CSV output with resume support
    evaluation_dir = utils.create_evaluation_directories(script_dir)
    filename = utils.generate_evaluation_filename(pddl_name, args.log_coverage, args.search, 'outcome_prediction')
    evaluation_file_path = os.path.join(evaluation_dir, filename)
    
    # Check for existing results to resume from
    processed_samples = set()
    if os.path.exists(evaluation_file_path):
        try:
            with open(evaluation_file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Create unique key from trace + prefix_len + target_attr
                    key = (row['Full Real Trace'], int(row['Prefix Length']), row['Target Attribute'])
                    processed_samples.add(key)
            print(f"Resuming: Found {len(processed_samples)} already processed samples in {evaluation_file_path}")
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}. Starting fresh.")
            processed_samples = set()
    
    # Open CSV file for appending (or create with header if new)
    csv_file_exists = os.path.exists(evaluation_file_path) and len(processed_samples) > 0
    csv_file = open(evaluation_file_path, 'a' if csv_file_exists else 'w', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_file_exists:
        csv_writer.writerow(["Full Real Trace", "Prefix Length", "Target Attribute", "Actual Value",
                            "Actual Discretized", "Planning Success", "Predicted Discretized",
                            "Real Suffix", "Predicted Suffix", "Distance", "Similarity",
                            "Solvability", "Search Algorithm", "Discovery Algorithm"])
        csv_file.flush()
    
    all_evaluation_samples = []  # Will store results for all target attributes
    
    stop_generation = False
    for target_attr in target_attrs:
        print(f"\n=== Evaluating predictions for attribute: {target_attr} ===")
        
        evaluation_samples = []  # [full_real_trace, prefix_len, target_attr, actual_value, predicted_value, prediction_error, execution_time]
        
        all_prefix_lengths = set()
        for trace in test_traces:
            sanitized_trace = [utils.sanitize_name(event) for event in trace]
            if sanitized_trace and len(sanitized_trace) > 1:
                # Test prefixes from 1 to len(trace)-1
                for i in range(1, len(sanitized_trace)):
                    all_prefix_lengths.add(i)
        
        prefix_lengths_to_test = sorted(list(all_prefix_lengths))
        print(f"Testing on {len(test_traces)} traces with prefix lengths: {prefix_lengths_to_test}")

        for trace_idx, (trace, trace_events) in enumerate(zip(test_traces, test_traces_with_events)):
            sanitized_trace = [utils.sanitize_name(event) for event in trace]
            if not sanitized_trace or len(sanitized_trace) <= 1:
                continue

            # Find the final value of the target attribute from any event in the trace
            actual_final_value = None
            for event in reversed(trace_events):  # Start from the end
                actual_final_value = utils.get_goal_attribute_value(parser, event, target_attr)
                if actual_final_value is not None:
                    break
            
            if actual_final_value is None:
                print(f"Skipping trace {trace_idx} - no value found for {target_attr} in any event")
                continue

            for prefix_len in range(1, len(sanitized_trace)):
                prefix = sanitized_trace[:prefix_len]
                
                # Get the actual discretized target value
                actual_discretized = get_actual_discretized_value(parser, target_attr, actual_final_value)
                if not actual_discretized:
                    continue
                
                # Compute initial state (including current target attribute value) - restrict enabled activities to successors of last prefix activity
                init_condition = utils.compute_initial_state_from_last_activity(parser, prefix, trace_events, prefix_len, target_attr)
                
                # Get the initial attribute value to compare with goal
                sanitized_attr = target_attr.lower().replace(':', '_').replace(' ', '_')
                initial_attr_value = None
                for predicate in init_condition:
                    if predicate.startswith(f"({sanitized_attr} "):
                        initial_attr_value = predicate.split()[-1].rstrip(')')
                        break
                
                # Skip if goal attribute value is the same as initial value
                if initial_attr_value == actual_discretized:
                    continue  # Goal already achieved
                
                goal_condition = [f"({sanitized_attr} {actual_discretized})"]                
                # Honor the global max samples cap across all attributes
                if args.max_samples is not None and isinstance(args.max_samples, int) and args.max_samples > 0:
                    current_total = len(all_evaluation_samples) + len(evaluation_samples)
                    if current_total >= args.max_samples:
                        stop_generation = True
                        break
                real_suffix = sanitized_trace[prefix_len:]  # Real remaining trace after prefix
                evaluation_samples.append([
                    sanitized_trace,      # 0: Full real trace
                    prefix_len,           # 1: Length of the prefix used for init
                    target_attr,          # 2: Target attribute name
                    actual_final_value,   # 3: Actual final value (numerical)
                    actual_discretized,   # 4: Actual discretized value
                    init_condition,       # 5: PDDL init condition
                    goal_condition,       # 6: PDDL goal condition
                    None,                 # 7: Planning success (True/False)
                    None,                 # 8: Predicted discretized value
                    0.0,                  # 9: Execution time
                    trace_events,         # 10: Full trace events for debugging
                    None,                 # 11: Solvability status (solved, unsolvable_structural, unsolvable_resource)
                    real_suffix,          # 12: Real suffix (ground truth)
                    [],                   # 13: Predicted suffix (from planner)
                    0,                    # 14: Damerau-Levenshtein distance
                    0.0                   # 15: Similarity percentage
                ])

        print(f"Generated {len(evaluation_samples)} evaluation samples for attribute {target_attr}")
        if stop_generation:
            print(f"Reached global sample cap of {args.max_samples}. Stopping generation of further samples.")
            # Extend with the generated ones and break out of attribute loop
            all_evaluation_samples.extend(evaluation_samples)
            break

        domain_generated = False
        for i, sample in enumerate(evaluation_samples):
            full_real_trace = sample[0]
            prefix_len = sample[1]
            target_attr_name = sample[2]
            
            # Check if this sample was already processed (for resume)
            sample_key = (' '.join(full_real_trace), prefix_len, target_attr_name)
            if sample_key in processed_samples:
                print(f"\n--- Skipping Sample {i+1}/{len(evaluation_samples)} for {target_attr} (already processed) ---")
                continue
            
            print(f"\n--- Processing Sample {i+1}/{len(evaluation_samples)} for {target_attr} ---")
            start_time = time.time()
            actual_value = sample[3]
            actual_discretized = sample[4]
            init_condition = sample[5]
            goal_condition = sample[6]

            print(f"Prefix: {full_real_trace[:prefix_len]}")
            print(f"Actual {target_attr}: {actual_value} (discretized: {actual_discretized})")
            print(f"Goal condition: {goal_condition}")

            # Use minimal preconditions for outcome prediction as well so that
            # planners can explore possible outcome paths based on enabled markers
            encoder = Encoder(parser,
                              pddl_name,
                              init=init_condition,
                              goal=goal_condition,
                              minimal_preconditions=True)

            if not domain_generated:
                encoder.generate_domain(output_path=domain_file_path)
                print(f"Domain file generated: {domain_file_path}")
                domain_generated = True

            encoder.generate_problem(problem_name=f"{pddl_name}_{target_attr}_prefix{prefix_len}_sample{i}", 
                                   output_path=problem_file_path)

            plan_path = os.path.join(script_dir, '..', 'pddl', 'plan_problem.txt')
            planner_success, planner_message, planning_metrics, solvability = utils.run_planner(plan_path, args.search)
            
            sample[11] = solvability  # Store solvability status
            
            real_suffix = sample[12]
            if planner_success:
                sample[7] = True  # Planning success
                sample[8] = actual_discretized  # Predicted value is the tested goal
                print(f"SUCCESS: Plan found to achieve {actual_discretized}")
                predicted_sequence = utils.parse_plan_file(plan_path)
                sample[13] = predicted_sequence  # Store predicted suffix
                print(f"Predicted sequence: {predicted_sequence}")
            else:
                sample[7] = False
                sample[8] = None
                predicted_sequence = []
                sample[13] = predicted_sequence
                print(f"FAILURE: {planner_message} (solvability: {solvability})")
            
            # Compute Damerau-Levenshtein distance and similarity
            real_suffix_str = ' '.join(real_suffix)
            predicted_suffix_str = ' '.join(predicted_sequence)
            
            if not real_suffix and not predicted_sequence:
                distance = 0
                similarity = 100.0
            elif not real_suffix or not predicted_sequence:
                distance = len(real_suffix) if real_suffix else len(predicted_sequence)
                similarity = 0.0
            else:
                distance = damerau_levenshtein_distance(predicted_suffix_str, real_suffix_str)
                max_len = max(len(predicted_sequence), len(real_suffix))
                if max_len == 0:
                    similarity = 100.0
                else:
                    similarity = (100.0 - (distance / max_len))
            
            sample[14] = distance
            sample[15] = similarity
            print(f"Real suffix: {real_suffix}")
            print(f"Distance (Predicted vs Real Suffix): {distance}")
            print(f"Similarity: {similarity:.2f}%")
            
            end_time = time.time()
            sample[9] = end_time - start_time
            
            # Write this sample's result immediately to CSV
            full_trace_str = ' '.join(full_real_trace)
            planning_success = sample[7] if sample[7] is not None else False
            predicted_discretized = sample[8] if sample[8] is not None else 'None'
            real_suffix_str = ' '.join(sample[12])
            predicted_suffix_str = ' '.join(sample[13])
            solvability = sample[11] if sample[11] else 'unknown'
            csv_writer.writerow([
                full_trace_str,
                prefix_len,
                target_attr,
                actual_value,
                actual_discretized,
                planning_success,
                predicted_discretized,
                real_suffix_str,
                predicted_suffix_str,
                sample[14],  # distance
                f"{sample[15]:.2f}",  # similarity
                solvability,
                args.search,
                args.discovery_algorithm
            ])
            csv_file.flush()  # Ensure it's written to disk immediately

        all_evaluation_samples.extend(evaluation_samples)
        if stop_generation:
            break
    
    # Close the CSV file
    csv_file.close()

    # Close the CSV file
    csv_file.close()

    metrics_filename = utils.generate_evaluation_filename(pddl_name, args.log_coverage, args.search, 'outcome_prediction_metrics')
    metrics_filename = metrics_filename.replace('.csv', '.txt')
    metrics_file_path = os.path.join(evaluation_dir, metrics_filename)

    print(f"\nEvaluation results saved to {evaluation_file_path}")
    
    # Re-read all results from CSV for metrics calculation (includes resumed data)
    all_results = []
    try:
        with open(evaluation_file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_results.append({
                    'full_trace': row['Full Real Trace'],
                    'prefix_len': int(row['Prefix Length']),
                    'target_attr': row['Target Attribute'],
                    'actual_value': float(row['Actual Value']) if row['Actual Value'] else None,
                    'actual_discretized': row['Actual Discretized'],
                    'planning_success': row['Planning Success'] == 'True',
                    'predicted_discretized': row['Predicted Discretized'],
                    'real_suffix': row['Real Suffix'],
                    'predicted_suffix': row['Predicted Suffix'],
                    'distance': float(row['Distance']) if row['Distance'] else 0,
                    'similarity': float(row['Similarity']) if row['Similarity'] else 0.0,
                    'solvability': row['Solvability']
                })
        print(f"Loaded {len(all_results)} total results for metrics calculation.")
    except Exception as e:
        print(f"Error reading results file for metrics: {e}")
        all_results = []

    try:
        print(f"Writing aggregate metrics to {metrics_file_path}...")
        with open(metrics_file_path, 'w') as metrics_file:
            metrics_file.write(f"Outcome Prediction Evaluation - Aggregate Metrics\n")
            metrics_file.write(f"{'='*50}\n")
            metrics_file.write(f"Dataset: {pddl_name}\n")
            metrics_file.write(f"Log Coverage: {args.log_coverage}\n")
            metrics_file.write(f"Search Algorithm: {args.search}\n")
            metrics_file.write(f"Discovery Algorithm: {args.discovery_algorithm}\n")
            metrics_file.write(f"Target Attribute(s): {', '.join(target_attrs)}\n")
            metrics_file.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            metrics_file.write(f"Total Samples: {len(all_results)}\n")
            metrics_file.write(f"\n")

        # Statistics
        for target_attr in target_attrs:
            attr_samples = [s for s in all_results if s['target_attr'] == target_attr]
            
            if attr_samples:
                # Compute solvability statistics
                total_samples = len(attr_samples)
                solved_samples = [s for s in attr_samples if s['solvability'] == utils.SOLVABILITY_SOLVED]
                unsolvable_structural_samples = [s for s in attr_samples if s['solvability'] == utils.SOLVABILITY_UNSOLVABLE_STRUCTURAL]
                unsolvable_resource_samples = [s for s in attr_samples if s['solvability'] == utils.SOLVABILITY_UNSOLVABLE_RESOURCE]
                
                solved_count = len(solved_samples)
                unsolvable_structural_count = len(unsolvable_structural_samples)
                unsolvable_resource_count = len(unsolvable_resource_samples)
                
                solved_pct = (solved_count / total_samples * 100) if total_samples > 0 else 0
                unsolvable_structural_pct = (unsolvable_structural_count / total_samples * 100) if total_samples > 0 else 0
                unsolvable_resource_pct = (unsolvable_resource_count / total_samples * 100) if total_samples > 0 else 0
                
                print(f"\n--- Solvability Statistics for {target_attr} ---")
                print(f"Total Samples: {total_samples}")
                print(f"Solved: {solved_count} ({solved_pct:.2f}%)")
                print(f"Unsolvable (Structural): {unsolvable_structural_count} ({unsolvable_structural_pct:.2f}%)")
                print(f"Unsolvable (Resource): {unsolvable_resource_count} ({unsolvable_resource_pct:.2f}%)")
                
                with open(metrics_file_path, 'a') as metrics_file:
                    metrics_file.write(f"SOLVABILITY STATISTICS FOR {target_attr}\n")
                    metrics_file.write(f"{'-'*35}\n")
                    metrics_file.write(f"Total Samples: {total_samples}\n")
                    metrics_file.write(f"Solved: {solved_count} ({solved_pct:.2f}%)\n")
                    metrics_file.write(f"Unsolvable (Structural): {unsolvable_structural_count} ({unsolvable_structural_pct:.2f}%)\n")
                    metrics_file.write(f"Unsolvable (Resource): {unsolvable_resource_count} ({unsolvable_resource_pct:.2f}%)\n")
                    metrics_file.write(f"\n")
                
                successful_samples = [s for s in attr_samples if s['planning_success'] is True]
                success_rate = len(successful_samples) / total_samples * 100
                print(f"\n--- Statistics for {target_attr} ---")
                print(f"Total samples: {total_samples}")
                print(f"Successful plans: {len(successful_samples)}")
                print(f"Planning Success Rate: {success_rate:.2f}%")
                with open(metrics_file_path, 'a') as metrics_file:
                    metrics_file.write(f"METRICS FOR ATTRIBUTE: {target_attr}\n")
                    metrics_file.write(f"{'-'*30}\n")
                    metrics_file.write(f"Total Samples: {total_samples}\n")
                    metrics_file.write(f"Successful Plans: {len(successful_samples)}\n")
                    metrics_file.write(f"Planning Success Rate: {success_rate:.2f}%\n")
                
                # Discretized value accuracy
                correct_predictions = [s for s in successful_samples if s['predicted_discretized'] == s['actual_discretized']]  # predicted == actual discretized
                if successful_samples:
                    accuracy = len(correct_predictions) / len(successful_samples) * 100
                    print(f"Discretized Value Accuracy (among successful plans): {accuracy:.2f}%")
                    with open(metrics_file_path, 'a') as metrics_file:
                        metrics_file.write(f"Discretized Value Accuracy: {accuracy:.2f}%\n")
                
                # Damerau-Levenshtein similarity metrics (like suffix prediction)
                valid_samples = [s for s in attr_samples if s['solvability'] == utils.SOLVABILITY_SOLVED]
                if valid_samples:
                    total_distance = sum(item['distance'] for item in valid_samples)
                    average_distance = total_distance / len(valid_samples)
                    print(f"\nOverall Average Damerau-Levenshtein Distance across {len(valid_samples)} solved samples: {average_distance:.2f}")

                    total_similarity = sum(item['similarity'] for item in valid_samples)
                    average_similarity = total_similarity / len(valid_samples)
                    print(f"Overall Average Similarity based on Damerau-Levenshtein Distance: {average_similarity:.2f}%")

                    with open(metrics_file_path, 'a') as metrics_file:
                        metrics_file.write(f"\nDAMERAU-LEVENSHTEIN METRICS (Solved Samples Only)\n")
                        metrics_file.write(f"{'-'*45}\n")
                        metrics_file.write(f"Solved Samples: {len(valid_samples)}\n")
                        metrics_file.write(f"Overall Average Distance: {average_distance:.2f}\n")
                        metrics_file.write(f"Overall Average Similarity: {average_similarity:.2f}%\n")
                        metrics_file.write(f"\n")
                else:
                    with open(metrics_file_path, 'a') as metrics_file:
                        metrics_file.write(f"\n")
                
                # Per-prefix statistics
                prefix_stats = {}
                prefix_lengths_to_test = sorted(list(set(s['prefix_len'] for s in attr_samples)))
                print("\n--- Per-Prefix Statistics ---")
                with open(metrics_file_path, 'a') as metrics_file:
                    metrics_file.write(f"PER-PREFIX STATISTICS FOR {target_attr}\n")
                    metrics_file.write(f"{'-'*35}\n")
                
                for prefix_len in prefix_lengths_to_test:
                    prefix_samples = [s for s in attr_samples if s['prefix_len'] == prefix_len]
                    if prefix_samples:
                        prefix_successful = [s for s in prefix_samples if s['planning_success'] is True]
                        prefix_success_rate = len(prefix_successful) / len(prefix_samples) * 100
                        
                        # Solvability breakdown for this prefix
                        prefix_solved = len([s for s in prefix_samples if s['solvability'] == utils.SOLVABILITY_SOLVED])
                        prefix_struct = len([s for s in prefix_samples if s['solvability'] == utils.SOLVABILITY_UNSOLVABLE_STRUCTURAL])
                        prefix_resource = len([s for s in prefix_samples if s['solvability'] == utils.SOLVABILITY_UNSOLVABLE_RESOURCE])

                        print(f"  Prefix Length {prefix_len}: Success Rate = {prefix_success_rate:.2f}% ({len(prefix_successful)}/{len(prefix_samples)} samples)")
                        print(f"    Solvability: Solved={prefix_solved}, Structural={prefix_struct}, Resource={prefix_resource}")
                        prefix_accuracy = 0
                        prefix_avg_distance = None
                        prefix_avg_similarity = None
                        if prefix_successful:
                            prefix_correct = [s for s in prefix_successful if s['predicted_discretized'] == s['actual_discretized']]
                            prefix_accuracy = len(prefix_correct) / len(prefix_successful) * 100
                            print(f"    Discretized Accuracy: {prefix_accuracy:.2f}%")
                        
                        # Damerau-Levenshtein metrics for this prefix
                        prefix_solved_samples = [s for s in prefix_samples if s['solvability'] == utils.SOLVABILITY_SOLVED]
                        if prefix_solved_samples:
                            prefix_avg_distance = sum(s['distance'] for s in prefix_solved_samples) / len(prefix_solved_samples)
                            prefix_avg_similarity = sum(s['similarity'] for s in prefix_solved_samples) / len(prefix_solved_samples)
                            print(f"    Avg Distance: {prefix_avg_distance:.2f}, Avg Similarity: {prefix_avg_similarity:.2f}%")
                        
                        with open(metrics_file_path, 'a') as metrics_file:
                            metrics_file.write(f"Prefix Length {prefix_len}: Success Rate = {prefix_success_rate:.2f}% ({len(prefix_successful)}/{len(prefix_samples)} samples)\n")
                            metrics_file.write(f"  Solvability: Solved={prefix_solved}, Structural={prefix_struct}, Resource={prefix_resource}\n")
                            if prefix_successful:
                                metrics_file.write(f"  Discretized Accuracy: {prefix_accuracy:.2f}%\n")
                            if prefix_avg_distance is not None:
                                metrics_file.write(f"  Avg Distance: {prefix_avg_distance:.2f}, Avg Similarity: {prefix_avg_similarity:.2f}%\n")
                        
                        prefix_stats[prefix_len] = {
                            'success_rate': prefix_success_rate,
                            'accuracy': prefix_accuracy,
                            'samples': len(prefix_samples),
                            'solved': prefix_solved,
                            'structural': prefix_struct,
                            'resource': prefix_resource,
                            'avg_distance': prefix_avg_distance,
                            'avg_similarity': prefix_avg_similarity
                        }
                
                if prefix_stats:   # Overall metrics across all prefix lengths
                    avg_success_rate = sum(stats['success_rate'] for stats in prefix_stats.values()) / len(prefix_stats)
                    weighted_success_rate = sum(stats['success_rate'] * stats['samples'] for stats in prefix_stats.values()) / sum(stats['samples'] for stats in prefix_stats.values())
                    
                    # Normalized averages for distance and similarity
                    valid_prefix_lengths_count = sum(1 for stats in prefix_stats.values() if stats['avg_distance'] is not None)
                    if valid_prefix_lengths_count > 0:
                        normalized_avg_distance = sum(d for d in [stats['avg_distance'] for stats in prefix_stats.values()] if d is not None) / valid_prefix_lengths_count
                        normalized_avg_similarity = sum(s for s in [stats['avg_similarity'] for stats in prefix_stats.values()] if s is not None) / valid_prefix_lengths_count
                    else:
                        normalized_avg_distance = None
                        normalized_avg_similarity = None
                    
                    print(f"\n--- Overall Metrics for {target_attr} ---")
                    print(f"Average Success Rate across prefix lengths: {avg_success_rate:.2f}%")
                    print(f"Weighted Success Rate (by sample count): {weighted_success_rate:.2f}%")
                    if normalized_avg_distance is not None:
                        print(f"Normalized Average Distance (avg of per-prefix averages): {normalized_avg_distance:.2f}")
                        print(f"Normalized Average Similarity (avg of per-prefix averages): {normalized_avg_similarity:.2f}%")
                    
                    with open(metrics_file_path, 'a') as metrics_file:
                        metrics_file.write(f"\nOVERALL METRICS FOR {target_attr}\n")
                        metrics_file.write(f"{'-'*25}\n")
                        metrics_file.write(f"Average Success Rate across prefix lengths: {avg_success_rate:.2f}%\n")
                        metrics_file.write(f"Weighted Success Rate (by sample count): {weighted_success_rate:.2f}%\n")
                        if normalized_avg_distance is not None:
                            metrics_file.write(f"Normalized Avg Distance: {normalized_avg_distance:.2f}\n")
                            metrics_file.write(f"Normalized Avg Similarity: {normalized_avg_similarity:.2f}%\n")
                        metrics_file.write(f"\n")
            else:
                print(f"\n--- No samples found for {target_attr} ---")
                with open(metrics_file_path, 'a') as metrics_file:
                    metrics_file.write(f"METRICS FOR ATTRIBUTE: {target_attr}\n")
                    metrics_file.write(f"{'-'*30}\n")
                    metrics_file.write(f"No samples found for {target_attr}\n")
                    metrics_file.write(f"\n")

        print(f"\nAggregate metrics saved to: {metrics_file_path}")

    except IOError as e:
        print(f"Error writing evaluation file: {e}")

    print("\nOutcome prediction evaluation completed.")
