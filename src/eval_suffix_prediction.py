from datetime import datetime
from jellyfish._jellyfish import damerau_levenshtein_distance
import os
import time
import csv
from typing import List, Dict, Optional, Any, Tuple, Set, Union

from pddl_encoder import Encoder
from xes_parser import Parser

import evaluation_helper as eval_helper
import pddl_helper as pddl_builder
import argparse
import planner_helper as planner_utils
import core_utils as utils


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the suffix prediction evaluation.

    Returns:
        The parsed arguments as a Namespace object.
    """
    parser = eval_helper.create_base_argument_parser(
        "Run the evaluation of the framework for XES encoding in PDDL and generation of predictions through FOND planning."
    )
    return parser.parse_args()


def compute_goal_condition_suffix(
    parser: Parser, 
    target_activity: str, 
    trace_events: Optional[List[Any]] = None, 
    full_trace_length: int = 0, 
    suffix: Optional[List[str]] = None
) -> List[str]:
    """
    Compute a proper PDDL goal condition including attribute values from the final state.

    Args:
        parser: XES parser instance.
        target_activity: The final activity to reach.
        trace_events: Full trace data with attribute values.
        full_trace_length: Length of the full trace to get final attribute values.
        suffix: Optional list of activities in the real suffix.

    Returns:
        List of PDDL predicates for the goal state (deduplicated).
    """
    """
    Compute a proper PDDL goal condition including attribute values from the final state.
    
    Args:
        parser: XES parser instance
        target_activity: The final activity to reach
        trace_events: Full trace data with attribute values
        full_trace_length: Length of the full trace to get final attribute values
        
    Returns:
        list: PDDL predicates for the goal state (deduplicated)
    """
    # If a full suffix is provided, require all suffix activities to be completed
    if suffix and isinstance(suffix, (list, tuple)) and suffix:
        goal_predicates = [f"(completed {act})" for act in suffix]
    else:
        goal_predicates = [f"(completed {target_activity})"]
    
    if trace_events and full_trace_length > 0:
        last_trace_event = trace_events[-1]
        goal_predicates.extend(pddl_builder.extract_attribute_predicates(parser, last_trace_event))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_predicates = []
    for pred in goal_predicates:
        if pred not in seen:
            seen.add(pred)
            unique_predicates.append(pred)
    
    return unique_predicates


if __name__ == "__main__":
    print("Initialization of the PDDL encoding process for the XES event log...")
    args = parse_arguments()
    pddl_name = args.xes_name.split('.')[0]

    print(f"Using search algorithm: {args.search}")
    print(f"Using discovery algorithm: {args.discovery_algorithm}")

    script_dir = os.path.dirname(__file__)
    domain_file_path = os.path.join(script_dir, '..', 'pddl', f'domain.pddl')
    problem_file_path = os.path.join(script_dir, '..', 'pddl', f'problem.pddl')
    parser = Parser(args.xes_name, args.log_coverage, args.discovery_algorithm,
                     use_activity_classifier=args.use_activity_classifier)
    test_traces, test_traces_with_events = eval_helper.prepare_test_data(parser, args.max_traces)
    
    evaluation_samples = []  # [full_real_trace, prefix_len, real_suffix, last_event, init_cond, goal_cond, predicted_suffix, distance, similarity, execution_time, trace_events, solvability, full_suffix_match]
    
    prefix_lengths_to_test = eval_helper.calculate_all_prefix_lengths(test_traces)
    print(f"Testing on {len(test_traces)} traces with all possible prefix lengths: {prefix_lengths_to_test}")

    stop_generation = False
    for trace_idx, (trace, trace_events) in enumerate(zip(test_traces, test_traces_with_events)):
        sanitized_trace = [utils.sanitize_name(event) for event in trace]
        if not sanitized_trace:
            continue
        last_event = sanitized_trace[-1]

        for prefix_len in range(1, len(sanitized_trace)):
            if len(sanitized_trace) > prefix_len:
                prefix = sanitized_trace[:prefix_len]
                real_suffix = sanitized_trace[prefix_len:]
                # Use initial state that enables successors of the last prefix activity only
                init_condition = pddl_builder.compute_initial_state_from_last_activity(parser, prefix, trace_events, prefix_len)
                # Use goal condition requiring all activities in the real suffix to be completed
                goal_condition = compute_goal_condition_suffix(parser, last_event, trace_events, len(trace_events), suffix=real_suffix)
                evaluation_samples.append([
                    sanitized_trace,      # 0: Full real trace
                    prefix_len,           # 1: Length of the prefix used for init
                    real_suffix,          # 2: The actual remaining trace events (ground truth for prediction)
                    last_event,           # 3: The final event (goal)
                    init_condition,       # 4: PDDL init condition with attributes
                    goal_condition,       # 5: PDDL goal condition with attributes
                    [],                   # 6: Predicted suffix (from planner)
                    0,                    # 7: Distance
                    0.0,                  # 8: Similarity
                    0.0,                  # 9: Execution time for this sample
                    trace_events,         # 10: Full trace events for debugging
                    None,                 # 11: Solvability status (solved, unsolvable_structural, unsolvable_resource)
                    False                 # 12: Full suffix match (bool)
                ])

                # Stop generating samples if we reached the global cap
                if args.max_samples is not None and isinstance(args.max_samples, int) and args.max_samples > 0:
                    if len(evaluation_samples) >= args.max_samples:
                        stop_generation = True
                        break
        if stop_generation:
            break
    print(f"Generated {len(evaluation_samples)} evaluation samples for all possible prefix lengths.")

    # Set up incremental CSV output with resume support
    evaluation_dir = eval_helper.create_evaluation_directories(script_dir)
    filename = eval_helper.generate_evaluation_filename(pddl_name, args.log_coverage, args.search, 'suffix_prediction')
    evaluation_file_path = os.path.join(evaluation_dir, filename)
    
    # Check for existing results to resume from
    processed_samples = set()
    if os.path.exists(evaluation_file_path):
        try:
            with open(evaluation_file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Create unique key from trace + prefix_len
                    key = (row['Full Real Trace'], int(row['Prefix Length']))
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
        csv_writer.writerow(["Full Real Trace", "Prefix Length", "Real Suffix", "Predicted Suffix", 
                            "Distance", "Similarity", "Full Suffix Match", "Solvability", 
                            "Search Algorithm", "Discovery Algorithm"])
        csv_file.flush()

    for i, sample in enumerate(evaluation_samples):
        full_real_trace = sample[0]
        prefix_len = sample[1]
        
        # Check if this sample was already processed (for resume)
        sample_key = (' '.join(full_real_trace), prefix_len)
        if sample_key in processed_samples:
            print(f"\n--- Skipping Sample {i+1}/{len(evaluation_samples)} (already processed) ---")
            continue
        
        print(f"\n--- Processing Sample {i+1}/{len(evaluation_samples)} (Prefix Length: {prefix_len}) ---")
        start_time = time.time() # Start timer for this sample
        real_suffix = sample[2]
        last_event = sample[3]
        init_condition = sample[4]
        goal_condition = sample[5]

        print(f"Real Trace Prefix: {full_real_trace[:prefix_len]}")
        print(f"Real Trace Suffix (Target): {real_suffix}")
        print(f"Initial Condition: {init_condition}")
        print(f"Goal Condition: {goal_condition}")

        # Use minimal preconditions for suffix prediction to allow the planner
        # to generate full suffix sequences based only on the initial 'enabled'
        # markers (derived from the prefix's last activity) and attribute conditions.
        encoder = Encoder(parser,
                  pddl_name,
                  init=init_condition,
                  goal=goal_condition,
                  minimal_preconditions=True)

        if i == 0:
             encoder.generate_domain(output_path=domain_file_path)
             print(f"Domain file generated: {domain_file_path}")

        encoder.generate_problem(problem_name=f"{pddl_name}_prefix{prefix_len}_sample{i}", output_path=problem_file_path)
        print(f"Problem file generated: {problem_file_path}")

        plan_path = os.path.join(script_dir, '..', 'pddl', 'plan_problem.txt')
        print(f"Running planner...")
        
        planner_success, planner_message, planning_metrics, solvability = planner_utils.run_planner(plan_path, args.search)
        sample[11] = solvability  # Store solvability status
        
        if planner_success:
            print("Planning executed successfully.")
            predicted_suffix = planner_utils.parse_plan_file(plan_path)
            sample[6] = predicted_suffix
        else:
            print(f"Planning failed: {planner_message} (solvability: {solvability})")
            predicted_suffix = []  # Empty predicted suffix on failure
            sample[6] = predicted_suffix

            # Diagnostic: show why structural failures happen
            if solvability == planner_utils.SOLVABILITY_UNSOLVABLE_STRUCTURAL:
                try:
                    print("  Diagnostic Info: Structural unsolvable sample details:")
                    print(f"    Init predicates: {init_condition}")
                    enabled_from_last = pddl_builder.compute_enabled_activities_after_last_activity(parser, last_event)
                    print(f"    Enabled after last activity (direct/petri): {sorted(list(enabled_from_last))}")
                    reachable = pddl_builder.compute_reachable_activities_from_last_activity(parser, last_event, max_depth=5)
                    print(f"    Reachable activities from last activity via direct graph (depth 5): {sorted(list(reachable))}")
                    print(f"    Goal predicates: {goal_condition}")
                except Exception as e:
                    print(f"    Diagnostic error: {e}")
        
        real_suffix_str = ' '.join(real_suffix)
        predicted_suffix_str = ' '.join(predicted_suffix)

        # Check whether the predicted suffix matches the full real suffix
        full_suffix_match = (predicted_suffix == real_suffix)
        sample[12] = full_suffix_match

        if not real_suffix and not predicted_suffix:
            distance = 0
            similarity = 100.0
        elif not real_suffix or not predicted_suffix:
            distance = len(real_suffix) if real_suffix else len(predicted_suffix)
            similarity = 0.0
        else:
            distance = damerau_levenshtein_distance(predicted_suffix_str, real_suffix_str)
            max_len = max(len(predicted_suffix), len(real_suffix))
            if max_len == 0:
                similarity = 100.0
            else:
                similarity = (100.0 - (distance / max_len))
        sample[7] = distance
        sample[8] = similarity

        print(f"Predicted suffix: {predicted_suffix}")
        print(f"Distance (Predicted Suffix vs Real Suffix): {distance}")
        print(f"Similarity: {similarity:.2f}%")
        
        end_time = time.time()
        sample[9] = end_time - start_time

        # Write this sample's result immediately to CSV
        full_trace_str = ' '.join(full_real_trace)
        real_suffix_str = ' '.join(real_suffix)
        predicted_suffix_str = ' '.join(sample[6])
        solvability = sample[11] if sample[11] else 'unknown'
        csv_writer.writerow([
            full_trace_str,
            prefix_len,
            real_suffix_str,
            predicted_suffix_str,
            sample[7],  # distance
            f"{sample[8]:.2f}",  # similarity
            sample[12],  # full_suffix_match
            solvability,
            args.search,
            args.discovery_algorithm
        ])
        csv_file.flush()  # Ensure it's written to disk immediately

        print("PDDL encoding and planning process finished for this sample.")

    # Close the CSV file
    csv_file.close()
    
    metrics_filename = eval_helper.generate_evaluation_filename(pddl_name, args.log_coverage, args.search, 'suffix_prediction_metrics')
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
                    'real_suffix': row['Real Suffix'],
                    'predicted_suffix': row['Predicted Suffix'],
                    'distance': float(row['Distance']),
                    'similarity': float(row['Similarity']),
                    'full_suffix_match': row['Full Suffix Match'] == 'True',
                    'solvability': row['Solvability']
                })
        print(f"Loaded {len(all_results)} total results for metrics calculation.")

        # Write aggregate metrics to file
        print(f"Writing aggregate metrics to {metrics_file_path}...")
        with open(metrics_file_path, 'w') as metrics_file:
            metrics_file.write(f"Suffix Prediction Evaluation - Aggregate Metrics\n")
            metrics_file.write(f"={'='*50}\n")
            metrics_file.write(f"Dataset: {pddl_name}\n")
            metrics_file.write(f"Log Coverage: {args.log_coverage}\n")
            metrics_file.write(f"Search Algorithm: {args.search}\n")
            metrics_file.write(f"Discovery Algorithm: {args.discovery_algorithm}\n")
            metrics_file.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            metrics_file.write(f"Total Samples: {len(all_results)}\n")
            metrics_file.write(f"\n")

        # Overall and per-prefix averages
        if all_results:
            # Compute solvability statistics
            total_samples = len(all_results)
            solved_samples = [s for s in all_results if s['solvability'] == planner_utils.SOLVABILITY_SOLVED]
            unsolvable_structural_samples = [s for s in all_results if s['solvability'] == planner_utils.SOLVABILITY_UNSOLVABLE_STRUCTURAL]
            unsolvable_resource_samples = [s for s in all_results if s['solvability'] == planner_utils.SOLVABILITY_UNSOLVABLE_RESOURCE]
            
            solved_count = len(solved_samples)
            unsolvable_structural_count = len(unsolvable_structural_samples)
            unsolvable_resource_count = len(unsolvable_resource_samples)
            
            solved_pct = (solved_count / total_samples * 100) if total_samples > 0 else 0
            unsolvable_structural_pct = (unsolvable_structural_count / total_samples * 100) if total_samples > 0 else 0
            unsolvable_resource_pct = (unsolvable_resource_count / total_samples * 100) if total_samples > 0 else 0
            
            print(f"\n--- Solvability Statistics ---")
            print(f"Total Samples: {total_samples}")
            print(f"Solved: {solved_count} ({solved_pct:.2f}%)")
            print(f"Unsolvable (Structural): {unsolvable_structural_count} ({unsolvable_structural_pct:.2f}%)")
            print(f"Unsolvable (Resource): {unsolvable_resource_count} ({unsolvable_resource_pct:.2f}%)")
            
            # Write solvability metrics to file
            with open(metrics_file_path, 'a') as metrics_file:
                metrics_file.write(f"SOLVABILITY STATISTICS\n")
                metrics_file.write(f"{'-'*25}\n")
                metrics_file.write(f"Total Samples: {total_samples}\n")
                metrics_file.write(f"Solved: {solved_count} ({solved_pct:.2f}%)\n")
                metrics_file.write(f"Unsolvable (Structural): {unsolvable_structural_count} ({unsolvable_structural_pct:.2f}%)\n")
                metrics_file.write(f"Unsolvable (Resource): {unsolvable_resource_count} ({unsolvable_resource_pct:.2f}%)\n")
                metrics_file.write(f"\n")
            
            valid_samples = [s for s in all_results if s['solvability'] == planner_utils.SOLVABILITY_SOLVED]

            if valid_samples:
                total_distance = sum(item['distance'] for item in valid_samples)
                average_distance = total_distance / len(valid_samples)
                print(f"\nOverall Average Damerau-Levenshtein Distance across {len(valid_samples)} solved samples: {average_distance:.2f}")

                total_similarity = sum(item['similarity'] for item in valid_samples)
                average_similarity = total_similarity / len(valid_samples)
                print(f"Overall Average Similarity based on Damerau-Levenshtein Distance: {average_similarity:.2f}%")

                # Write overall metrics to file
                with open(metrics_file_path, 'a') as metrics_file:
                    metrics_file.write(f"OVERALL METRICS (Solved Samples Only)\n")
                    metrics_file.write(f"{'-'*35}\n")
                    metrics_file.write(f"Solved Samples: {len(valid_samples)}\n")
                    metrics_file.write(f"Overall Average Distance: {average_distance:.2f}\n")
                    metrics_file.write(f"Overall Average Similarity: {average_similarity:.2f}%\n")
                    metrics_file.write(f"\n")

                # Check how many valid samples have the full suffix correctly predicted
                full_suffix_matches = [s for s in valid_samples if s['full_suffix_match']]
                full_suffix_match_count = len(full_suffix_matches)
                full_suffix_match_pct = (full_suffix_match_count / len(valid_samples) * 100) if len(valid_samples) > 0 else 0
                print(f"Overall Full Suffix Exact Match: {full_suffix_match_count}/{len(valid_samples)} ({full_suffix_match_pct:.2f}%)")
                with open(metrics_file_path, 'a') as metrics_file:
                    metrics_file.write(f"OVERALL FULL SUFFIX MATCH (Solved Samples Only)\n")
                    metrics_file.write(f"{'-'*40}\n")
                    metrics_file.write(f"Full Suffix Exact Matches: {full_suffix_match_count} ({full_suffix_match_pct:.2f}%)\n")
                    metrics_file.write(f"\n")

                # Per-prefix averages and store them for normalization
                prefix_avg_distances = {}
                prefix_avg_similarities = {}
                valid_prefix_lengths_count = 0
                prefix_solvability_stats = {}  # Store solvability stats per prefix
                
                # Get all prefix lengths from results
                all_prefix_lengths = sorted(set(s['prefix_len'] for s in all_results))

                print("\n--- Per-Prefix Averages ---")
                
                # Write per-prefix header to file
                with open(metrics_file_path, 'a') as metrics_file:
                    metrics_file.write(f"PER-PREFIX METRICS\n")
                    metrics_file.write(f"{'-'*20}\n")
                
                for prefix_len in all_prefix_lengths:
                    prefix_samples = [s for s in valid_samples if s['prefix_len'] == prefix_len]
                    all_prefix_samples = [s for s in all_results if s['prefix_len'] == prefix_len]
                    
                    # Compute solvability stats for this prefix
                    prefix_total = len(all_prefix_samples)
                    prefix_solved = len([s for s in all_prefix_samples if s['solvability'] == planner_utils.SOLVABILITY_SOLVED])
                    prefix_struct = len([s for s in all_prefix_samples if s['solvability'] == planner_utils.SOLVABILITY_UNSOLVABLE_STRUCTURAL])
                    prefix_resource = len([s for s in all_prefix_samples if s['solvability'] == planner_utils.SOLVABILITY_UNSOLVABLE_RESOURCE])
                    
                    prefix_solvability_stats[prefix_len] = {
                        'total': prefix_total,
                        'solved': prefix_solved,
                        'structural': prefix_struct,
                        'resource': prefix_resource
                    }
                    
                    if prefix_samples:
                        avg_dist_prefix = sum(s['distance'] for s in prefix_samples) / len(prefix_samples)
                        avg_sim_prefix = sum(s['similarity'] for s in prefix_samples) / len(prefix_samples)
                        prefix_avg_distances[prefix_len] = avg_dist_prefix
                        prefix_avg_similarities[prefix_len] = avg_sim_prefix
                        valid_prefix_lengths_count += 1
                        print(f"  Prefix Length {prefix_len}: Avg Distance = {avg_dist_prefix:.2f}, Avg Similarity = {avg_sim_prefix:.2f}% ({len(prefix_samples)} solved samples)")
                        print(f"    Solvability: Solved={prefix_solved}/{prefix_total}, Structural={prefix_struct}, Resource={prefix_resource}")
                        
                        # Write to file
                        with open(metrics_file_path, 'a') as metrics_file:
                            metrics_file.write(f"Prefix Length {prefix_len}: Avg Distance = {avg_dist_prefix:.2f}, Avg Similarity = {avg_sim_prefix:.2f}% ({len(prefix_samples)} solved samples)\n")
                            metrics_file.write(f"  Solvability: Solved={prefix_solved}/{prefix_total}, Structural={prefix_struct}, Resource={prefix_resource}\n")
                            # Full suffix match for this prefix
                            prefix_full_matches = len([s for s in prefix_samples if s['full_suffix_match']])
                            prefix_full_match_pct = (prefix_full_matches / len(prefix_samples) * 100) if len(prefix_samples) > 0 else 0
                            metrics_file.write(f"  Full Suffix Exact Matches: {prefix_full_matches} ({prefix_full_match_pct:.2f}%)\n")
                    else:
                        print(f"  Prefix Length {prefix_len}: No solved samples for distance/similarity.")
                        print(f"    Solvability: Solved={prefix_solved}/{prefix_total}, Structural={prefix_struct}, Resource={prefix_resource}")
                        prefix_avg_distances[prefix_len] = None 
                        prefix_avg_similarities[prefix_len] = None
                        
                        # Write to file
                        with open(metrics_file_path, 'a') as metrics_file:
                            metrics_file.write(f"Prefix Length {prefix_len}: No solved samples for distance/similarity.\n")
                            metrics_file.write(f"  Solvability: Solved={prefix_solved}/{prefix_total}, Structural={prefix_struct}, Resource={prefix_resource}\n")

                # Normalized Overall Average (average of per-prefix averages)
                if valid_prefix_lengths_count > 0:
                    normalized_avg_distance = sum(d for d in prefix_avg_distances.values() if d is not None) / valid_prefix_lengths_count
                    normalized_avg_similarity = sum(s for s in prefix_avg_similarities.values() if s is not None) / valid_prefix_lengths_count
                    print("\n--- Normalized Overall Averages (Equal weight per prefix length) ---")
                    print(f"Normalized Overall Average Distance: {normalized_avg_distance:.2f}")
                    print(f"Normalized Overall Average Similarity: {normalized_avg_similarity:.2f}%")
                    
                    # Write normalized metrics to file
                    with open(metrics_file_path, 'a') as metrics_file:
                        metrics_file.write(f"\nNORMALIZED OVERALL METRICS (Equal weight per prefix length)\n")
                        metrics_file.write(f"{'-'*50}\n")
                        metrics_file.write(f"Normalized Overall Average Distance: {normalized_avg_distance:.2f}\n")
                        metrics_file.write(f"Normalized Overall Average Similarity: {normalized_avg_similarity:.2f}%\n")
                        metrics_file.write(f"\n")
                else:
                     print("\nCould not calculate normalized averages as no prefix lengths had successful samples.")
                     
                     # Write to file
                     with open(metrics_file_path, 'a') as metrics_file:
                         metrics_file.write(f"\nCould not calculate normalized averages as no prefix lengths had successful samples.\n\n")
                
                print(f"\nAggregate metrics saved to: {metrics_file_path}")
            else:
                print("\nNo successful samples were evaluated.")
                
                # Write to file
                with open(metrics_file_path, 'a') as metrics_file:
                    metrics_file.write(f"No successful samples were evaluated.\n")
        else:
            print("\nNo evaluation samples were found in results file.")
            
            # Write to file
            with open(metrics_file_path, 'w') as metrics_file:
                metrics_file.write(f"No evaluation samples were found.\n")

    except IOError as e:
        print(f"Error processing evaluation: {e}")
