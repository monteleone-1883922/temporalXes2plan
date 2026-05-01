#!/usr/bin/env python3
"""
Summarize evaluation metrics from txt files into CSV summaries.

This script processes the evaluation folder and creates summary CSV files
for each type of evaluation (plan_quality, suffix_prediction, outcome_prediction).
It also generates visualizations for plan quality metrics.
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict
import argparse
from typing import List, Dict, Optional, Any, Tuple, Set, Union

import matplotlib.pyplot as plt
import numpy as np


def parse_plan_quality_metrics(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse plan quality metrics from a text file.

    Args:
        filepath: Path to the .txt file containing plan quality metrics.

    Returns:
        A dictionary of parsed metrics.
    """
    """Parse plan quality metrics from txt file."""
    metrics = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract header info
    if match := re.search(r'Dataset:\s*(\S+)', content):
        metrics['dataset'] = match.group(1)
    if match := re.search(r'Log Coverage:\s*(\S+)', content):
        metrics['log_coverage'] = match.group(1)
    if match := re.search(r'Search Algorithm:\s*(\S+)', content):
        metrics['search_algorithm'] = match.group(1)
    if match := re.search(r'Discovery Algorithm:\s*(\S+)', content):
        metrics['discovery_algorithm'] = match.group(1)
    if match := re.search(r'Total Samples:\s*(\d+)', content):
        metrics['total_samples'] = int(match.group(1))
    
    # Solvability statistics
    if match := re.search(r'Solved:\s*(\d+)\s*\(([\d.]+)%\)', content):
        metrics['solved'] = int(match.group(1))
        metrics['solved_pct'] = float(match.group(2))
    if match := re.search(r'Unsolvable \(Structural\):\s*(\d+)\s*\(([\d.]+)%\)', content):
        metrics['unsolvable_structural'] = int(match.group(1))
        metrics['unsolvable_structural_pct'] = float(match.group(2))
    if match := re.search(r'Unsolvable \(Resource\):\s*(\d+)\s*\(([\d.]+)%\)', content):
        metrics['unsolvable_resource'] = int(match.group(1))
        metrics['unsolvable_resource_pct'] = float(match.group(2))
    
    # Overall metrics
    if match := re.search(r'Planner Success Rate:\s*([\d.]+)%', content):
        metrics['planner_success_rate'] = float(match.group(1))
    if match := re.search(r'Target Reaching Rate:\s*([\d.]+)%', content):
        metrics['target_reaching_rate'] = float(match.group(1))
    if match := re.search(r'Minimality Rate:\s*([\d.]+)%', content):
        metrics['minimality_rate'] = float(match.group(1))
    if match := re.search(r'Average Length Difference:\s*([-\d.]+)', content):
        metrics['avg_length_difference'] = float(match.group(1))
    if match := re.search(r'Average Planning Time:\s*([\d.]+)\s*seconds', content):
        metrics['avg_planning_time'] = float(match.group(1))
    if match := re.search(r'Average Search Time:\s*([\d.]+)\s*seconds', content):
        metrics['avg_search_time'] = float(match.group(1))
    if match := re.search(r'Average Grounding Time:\s*([\d.]+)\s*seconds', content):
        metrics['avg_grounding_time'] = float(match.group(1))
    if match := re.search(r'Average Expanded Nodes:\s*([\d.]+)', content):
        metrics['avg_expanded_nodes'] = float(match.group(1))
    if match := re.search(r'Average Space Used:\s*([\d.]+)\s*MB', content):
        metrics['avg_space_used_mb'] = float(match.group(1))
    if match := re.search(r'Average Solution Length:\s*([\d.]+)', content):
        metrics['avg_solution_length'] = float(match.group(1))
    
    # Parse per-prefix length metrics
    prefix_metrics = parse_plan_quality_prefix_metrics(content)
    if prefix_metrics:
        metrics['prefix_metrics'] = prefix_metrics
    
    return metrics


def parse_plan_quality_prefix_metrics(content: str) -> Dict[int, Dict[str, Any]]:
    """
    Parse per-prefix length metrics from the content of a plan quality text file.

    Args:
        content: The string content of the metrics file.

    Returns:
        A dictionary mapping prefix lengths to their respective metrics.
    """
    """Parse per-prefix length metrics from plan quality txt file content."""
    prefix_metrics = {}
    
    # Pattern to match per-prefix lines
    # Example: Prefix Length  1: Samples= 53, Success=100.0%, Target=100.0%, Avg Time=0.1817s, Solved=53, Structural=0, Resource=0, Avg Search=0.0004s, Avg Grounding=0.0294s, Avg FD Total=0.0298s, Avg Nodes=5, Avg Space=10.09MB, Avg Sol.Len=4.2
    pattern = r'Prefix Length\s+(\d+):\s*Samples=\s*(\d+).*?Avg FD Total=([\d.]+)s'
    
    matches = re.findall(pattern, content)
    for match in matches:
        prefix_len = int(match[0])
        samples = int(match[1])
        fd_total_time = float(match[2])
        prefix_metrics[prefix_len] = {
            'samples': samples,
            'avg_fd_total_time': fd_total_time
        }
    
    return prefix_metrics


def parse_suffix_prediction_metrics(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse suffix prediction metrics from a text file.

    Args:
        filepath: Path to the .txt file containing suffix prediction metrics.

    Returns:
        A dictionary of parsed metrics.
    """
    """Parse suffix prediction metrics from txt file."""
    metrics = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract header info
    if match := re.search(r'Dataset:\s*(\S+)', content):
        metrics['dataset'] = match.group(1)
    if match := re.search(r'Log Coverage:\s*(\S+)', content):
        metrics['log_coverage'] = match.group(1)
    if match := re.search(r'Search Algorithm:\s*(\S+)', content):
        metrics['search_algorithm'] = match.group(1)
    if match := re.search(r'Discovery Algorithm:\s*(\S+)', content):
        metrics['discovery_algorithm'] = match.group(1)
    if match := re.search(r'Total Samples:\s*(\d+)', content):
        metrics['total_samples'] = int(match.group(1))
    
    # Solvability statistics
    if match := re.search(r'Solved:\s*(\d+)\s*\(([\d.]+)%\)', content):
        metrics['solved'] = int(match.group(1))
        metrics['solved_pct'] = float(match.group(2))
    if match := re.search(r'Unsolvable \(Structural\):\s*(\d+)\s*\(([\d.]+)%\)', content):
        metrics['unsolvable_structural'] = int(match.group(1))
        metrics['unsolvable_structural_pct'] = float(match.group(2))
    if match := re.search(r'Unsolvable \(Resource\):\s*(\d+)\s*\(([\d.]+)%\)', content):
        metrics['unsolvable_resource'] = int(match.group(1))
        metrics['unsolvable_resource_pct'] = float(match.group(2))
    
    # Overall metrics (Solved Samples Only)
    if match := re.search(r'Solved Samples:\s*(\d+)', content):
        metrics['solved_samples'] = int(match.group(1))
    if match := re.search(r'Overall Average Distance:\s*([\d.]+)', content):
        metrics['overall_avg_distance'] = float(match.group(1))
    if match := re.search(r'Overall Average Similarity:\s*([\d.]+)%', content):
        metrics['overall_avg_similarity'] = float(match.group(1))
    
    # Full suffix match
    if match := re.search(r'Full Suffix Exact Matches:\s*(\d+)\s*\(([\d.]+)%\)', content):
        metrics['full_suffix_exact_matches'] = int(match.group(1))
        metrics['full_suffix_exact_match_pct'] = float(match.group(2))
    
    # Normalized metrics
    if match := re.search(r'Normalized Overall Average Distance:\s*([\d.]+)', content):
        metrics['normalized_avg_distance'] = float(match.group(1))
    if match := re.search(r'Normalized Overall Average Similarity:\s*([\d.]+)%', content):
        metrics['normalized_avg_similarity'] = float(match.group(1))
    
    return metrics


def parse_outcome_prediction_metrics(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse outcome prediction metrics from a text file.

    Args:
        filepath: Path to the .txt file containing outcome prediction metrics.

    Returns:
        A dictionary of parsed metrics.
    """
    """Parse outcome prediction metrics from txt file."""
    metrics = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract header info
    if match := re.search(r'Dataset:\s*(\S+)', content):
        metrics['dataset'] = match.group(1)
    if match := re.search(r'Log Coverage:\s*(\S+)', content):
        metrics['log_coverage'] = match.group(1)
    if match := re.search(r'Search Algorithm:\s*(\S+)', content):
        metrics['search_algorithm'] = match.group(1)
    if match := re.search(r'Discovery Algorithm:\s*(\S+)', content):
        metrics['discovery_algorithm'] = match.group(1)
    if match := re.search(r'Target Attribute\(s\):\s*(.+)', content):
        metrics['target_attributes'] = match.group(1).strip()
    if match := re.search(r'Total Samples:\s*(\d+)', content):
        metrics['total_samples'] = int(match.group(1))
    
    # Parse per-attribute metrics
    # Find all attributes with solvability statistics
    attr_blocks = re.findall(
        r'SOLVABILITY STATISTICS FOR (\w+)\s*[-]+\s*'
        r'Total Samples:\s*(\d+)\s*'
        r'Solved:\s*(\d+)\s*\(([\d.]+)%\)\s*'
        r'Unsolvable \(Structural\):\s*(\d+)\s*\(([\d.]+)%\)\s*'
        r'Unsolvable \(Resource\):\s*(\d+)\s*\(([\d.]+)%\)',
        content
    )
    
    # Aggregate solvability across all attributes
    total_solved = 0
    total_structural = 0
    total_resource = 0
    total_attr_samples = 0
    attr_metrics = []
    
    for attr_match in attr_blocks:
        attr_name = attr_match[0]
        attr_samples = int(attr_match[1])
        solved = int(attr_match[2])
        solved_pct = float(attr_match[3])
        structural = int(attr_match[4])
        resource = int(attr_match[6])
        
        total_attr_samples += attr_samples
        total_solved += solved
        total_structural += structural
        total_resource += resource
        
        # Find accuracy for this attribute
        accuracy_pattern = rf'METRICS FOR ATTRIBUTE: {attr_name}\s*[-]+.*?Discretized Value Accuracy:\s*([\d.]+)%'
        if acc_match := re.search(accuracy_pattern, content, re.DOTALL):
            accuracy = float(acc_match.group(1))
        else:
            accuracy = None
        
        # Find Damerau-Levenshtein metrics for this attribute
        dl_pattern = rf'DAMERAU-LEVENSHTEIN METRICS \(Solved Samples Only\)\s*[-]+\s*Solved Samples:\s*\d+\s*Overall Average Distance:\s*([\d.]+)\s*Overall Average Similarity:\s*([\d.]+)%'
        overall_avg_distance = None
        overall_avg_similarity = None
        # Look for the DL section that comes after the METRICS FOR ATTRIBUTE section
        attr_section_pattern = rf'METRICS FOR ATTRIBUTE: {attr_name}.*?(?=SOLVABILITY STATISTICS FOR|$)'
        if attr_section_match := re.search(attr_section_pattern, content, re.DOTALL):
            attr_section = attr_section_match.group(0)
            if dl_match := re.search(r'Overall Average Distance:\s*([\d.]+)', attr_section):
                overall_avg_distance = float(dl_match.group(1))
            if dl_match := re.search(r'Overall Average Similarity:\s*([\d.]+)%', attr_section):
                overall_avg_similarity = float(dl_match.group(1))
        
        # Find normalized metrics for this attribute
        normalized_distance = None
        normalized_similarity = None
        overall_pattern = rf'OVERALL METRICS FOR {attr_name}\s*[-]+.*?Normalized Avg Distance:\s*([\d.]+)\s*Normalized Avg Similarity:\s*([\d.]+)%'
        if norm_match := re.search(overall_pattern, content, re.DOTALL):
            normalized_distance = float(norm_match.group(1))
            normalized_similarity = float(norm_match.group(2))
        
        attr_metrics.append({
            'attribute': attr_name,
            'samples': attr_samples,
            'solved': solved,
            'solved_pct': solved_pct,
            'accuracy': accuracy,
            'overall_avg_distance': overall_avg_distance,
            'overall_avg_similarity': overall_avg_similarity,
            'normalized_avg_distance': normalized_distance,
            'normalized_avg_similarity': normalized_similarity
        })
    
    if total_attr_samples > 0:
        metrics['total_attr_samples'] = total_attr_samples
        metrics['solved'] = total_solved
        metrics['solved_pct'] = (total_solved / total_attr_samples) * 100 if total_attr_samples > 0 else 0
        metrics['unsolvable_structural'] = total_structural
        metrics['unsolvable_structural_pct'] = (total_structural / total_attr_samples) * 100 if total_attr_samples > 0 else 0
        metrics['unsolvable_resource'] = total_resource
        metrics['unsolvable_resource_pct'] = (total_resource / total_attr_samples) * 100 if total_attr_samples > 0 else 0
    
    # Calculate average accuracy across attributes with data
    accuracies = [a['accuracy'] for a in attr_metrics if a['accuracy'] is not None]
    if accuracies:
        metrics['avg_discretized_accuracy'] = sum(accuracies) / len(accuracies)
    
    # Calculate average Damerau-Levenshtein metrics across attributes
    overall_distances = [a['overall_avg_distance'] for a in attr_metrics if a['overall_avg_distance'] is not None]
    if overall_distances:
        metrics['overall_avg_distance'] = sum(overall_distances) / len(overall_distances)
    
    overall_similarities = [a['overall_avg_similarity'] for a in attr_metrics if a['overall_avg_similarity'] is not None]
    if overall_similarities:
        metrics['overall_avg_similarity'] = sum(overall_similarities) / len(overall_similarities)
    
    normalized_distances = [a['normalized_avg_distance'] for a in attr_metrics if a['normalized_avg_distance'] is not None]
    if normalized_distances:
        metrics['normalized_avg_distance'] = sum(normalized_distances) / len(normalized_distances)
    
    normalized_similarities = [a['normalized_avg_similarity'] for a in attr_metrics if a['normalized_avg_similarity'] is not None]
    if normalized_similarities:
        metrics['normalized_avg_similarity'] = sum(normalized_similarities) / len(normalized_similarities)
    
    # Store individual attribute info
    metrics['attributes_detail'] = attr_metrics
    
    return metrics


def summarize_plan_quality(eval_dir: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Aggregate plan quality metrics into a single CSV summary.

    Args:
        eval_dir: Directory containing individual metrics files.
        output_path: Path to the output summary CSV.
    """
    """Create summary CSV for plan quality evaluations."""
    eval_dir = Path(eval_dir)
    metrics_files = sorted(eval_dir.glob('plan_quality_metrics_*.txt'))
    
    if not metrics_files:
        print("No plan quality metrics files found.")
        return
    
    all_metrics = []
    for filepath in metrics_files:
        try:
            metrics = parse_plan_quality_metrics(filepath)
            metrics['source_file'] = filepath.name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    if not all_metrics:
        print("No valid plan quality metrics parsed.")
        return
    
    # Define columns (excluding FD total time, avg length ratio, combined quality score)
    columns = [
        'dataset', 'log_coverage', 'search_algorithm', 'discovery_algorithm',
        'total_samples', 'solved', 'solved_pct', 
        'unsolvable_structural', 'unsolvable_structural_pct',
        'unsolvable_resource', 'unsolvable_resource_pct',
        'planner_success_rate', 'target_reaching_rate', 'minimality_rate',
        'avg_length_difference', 'avg_planning_time', 'avg_search_time',
        'avg_grounding_time', 'avg_expanded_nodes', 'avg_space_used_mb',
        'avg_solution_length', 'source_file'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)
    
    print(f"Plan quality summary written to: {output_path}")
    print(f"  - {len(all_metrics)} evaluations summarized")


def summarize_suffix_prediction(eval_dir: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Aggregate suffix prediction metrics into a single CSV summary.

    Args:
        eval_dir: Directory containing individual metrics files.
        output_path: Path to the output summary CSV.
    """
    """Create summary CSV for suffix prediction evaluations."""
    eval_dir = Path(eval_dir)
    metrics_files = sorted(eval_dir.glob('suffix_prediction_metrics_*.txt'))
    
    if not metrics_files:
        print("No suffix prediction metrics files found.")
        return
    
    all_metrics = []
    for filepath in metrics_files:
        try:
            metrics = parse_suffix_prediction_metrics(filepath)
            metrics['source_file'] = filepath.name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    if not all_metrics:
        print("No valid suffix prediction metrics parsed.")
        return
    
    columns = [
        'dataset', 'log_coverage', 'search_algorithm', 'discovery_algorithm',
        'total_samples', 'solved', 'solved_pct',
        'unsolvable_structural', 'unsolvable_structural_pct',
        'unsolvable_resource', 'unsolvable_resource_pct',
        'solved_samples', 'overall_avg_distance', 'overall_avg_similarity',
        'full_suffix_exact_matches', 'full_suffix_exact_match_pct',
        'normalized_avg_distance', 'normalized_avg_similarity', 'source_file'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)
    
    print(f"Suffix prediction summary written to: {output_path}")
    print(f"  - {len(all_metrics)} evaluations summarized")


def summarize_outcome_prediction(eval_dir: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Aggregate outcome prediction metrics into a single CSV summary.

    Args:
        eval_dir: Directory containing individual metrics files.
        output_path: Path to the output summary CSV.
    """
    """Create summary CSV for outcome prediction evaluations."""
    eval_dir = Path(eval_dir)
    metrics_files = sorted(eval_dir.glob('outcome_prediction_metrics_*.txt'))
    
    if not metrics_files:
        print("No outcome prediction metrics files found.")
        return
    
    all_metrics = []
    for filepath in metrics_files:
        try:
            metrics = parse_outcome_prediction_metrics(filepath)
            metrics['source_file'] = filepath.name
            # Remove complex nested data for CSV
            metrics.pop('attributes_detail', None)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    if not all_metrics:
        print("No valid outcome prediction metrics parsed.")
        return
    
    columns = [
        'dataset', 'log_coverage', 'search_algorithm', 'discovery_algorithm',
        'target_attributes', 'total_samples', 'total_attr_samples',
        'solved', 'solved_pct',
        'unsolvable_structural', 'unsolvable_structural_pct',
        'unsolvable_resource', 'unsolvable_resource_pct',
        'avg_discretized_accuracy',
        'overall_avg_distance', 'overall_avg_similarity',
        'normalized_avg_distance', 'normalized_avg_similarity',
        'source_file'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)
    
    print(f"Outcome prediction summary written to: {output_path}")
    print(f"  - {len(all_metrics)} evaluations summarized")


def generate_plan_quality_fd_time_chart(
    eval_dir: Union[str, Path], 
    output_path: Union[str, Path], 
    figures_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Generate a line chart showing prediction time by prefix length.

    Args:
        eval_dir: Directory containing metrics files.
        output_path: Reference path (used to derive default figures directory).
        figures_dir: Optional custom directory to save the chart.
    """
    """
    Generate a line chart showing Avg FD Total Time across prefix lengths 
    for astar_blind search algorithm and inductive discovery algorithm
    across all evaluated logs.
    """
    eval_dir = Path(eval_dir)
    metrics_files = sorted(eval_dir.glob('plan_quality_metrics_*.txt'))
    
    if not metrics_files:
        print("No plan quality metrics files found for chart generation.")
        return
    
    # Collect data for astar_blind + inductive evaluations
    dataset_prefix_data = defaultdict(dict)  # {dataset: {prefix_len: fd_total_time}}
    
    for filepath in metrics_files:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Extract search algorithm
            search_match = re.search(r'Search Algorithm:\s*(\S+)', content)
            if not search_match or search_match.group(1) != 'astar_blind':
                continue
            
            # Extract discovery algorithm
            discovery_match = re.search(r'Discovery Algorithm:\s*(\S+)', content)
            if not discovery_match or discovery_match.group(1) != 'inductive':
                continue
            
            # Extract dataset name
            dataset_match = re.search(r'Dataset:\s*(\S+)', content)
            if not dataset_match:
                continue
            dataset = dataset_match.group(1)
            
            # Parse per-prefix metrics
            prefix_metrics = parse_plan_quality_prefix_metrics(content)
            if prefix_metrics:
                for prefix_len, metrics in prefix_metrics.items():
                    dataset_prefix_data[dataset][prefix_len] = metrics['avg_fd_total_time']
                    
        except Exception as e:
            print(f"Error parsing {filepath} for chart: {e}")
    
    if not dataset_prefix_data:
        print("No astar_blind + inductive plan quality data found for chart generation.")
        return
    
    # Create the line chart
    plt.figure(figsize=(12, 7))
    
    # Define colors and markers for different datasets
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_prefix_data)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*', '+', 'x']
    
    for idx, (dataset, prefix_data) in enumerate(sorted(dataset_prefix_data.items())):
        prefix_lengths = sorted(prefix_data.keys())
        fd_times = [prefix_data[p] for p in prefix_lengths]
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.plot(prefix_lengths, fd_times, 
                 label=dataset, 
                 marker=marker,
                 color=color,
                 linewidth=2,
                 markersize=6)
    
    plt.xlabel('Prefix Length (#Activities)', fontsize=16)
    plt.ylabel('Avg Prediction Time (s)', fontsize=16)
    #plt.title('Plan Quality: Avg FD Total Time by Prefix Length\n(astar_blind, inductive)', fontsize=14)
    plt.legend(title='Event Log', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the chart
    if figures_dir:
        figures_path = Path(figures_dir)
    else:
        figures_path = Path(output_path).parent.parent / 'figures'
    
    figures_path.mkdir(parents=True, exist_ok=True)
    chart_path = figures_path / 'prediction_total_time_by_prefix.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FD Total Time chart saved to: {chart_path}")
    print(f"  - {len(dataset_prefix_data)} datasets plotted")


def main() -> None:
    """Main entry point for the evaluation summarization script."""
    parser = argparse.ArgumentParser(
        description='Summarize evaluation metrics from txt files into CSV summaries.'
    )
    parser.add_argument(
        '--eval-dir',
        type=str,
        default='evaluation',
        help='Path to evaluation directory (default: evaluation)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for summary CSVs (default: same as eval-dir)'
    )
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: Evaluation directory '{eval_dir}' does not exist.")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing evaluation files from: {eval_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate summaries for each evaluation type
    summarize_plan_quality(eval_dir, output_dir / 'summary_plan_quality.csv')
    print()
    summarize_suffix_prediction(eval_dir, output_dir / 'summary_suffix_prediction.csv')
    print()
    summarize_outcome_prediction(eval_dir, output_dir / 'summary_outcome_prediction.csv')
    print()
    
    # Generate visualizations
    generate_plan_quality_fd_time_chart(eval_dir, output_dir / 'summary_plan_quality.csv')
    
    print()
    print("Done!")


if __name__ == '__main__':
    main()
