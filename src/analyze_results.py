import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Any, Tuple, Set, Union


def load_batch_results(results_dir: Union[str, Path]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Load batch evaluation results from a directory.

    Args:
        results_dir: Directory containing 'batch_summary.csv' and 'complete_results.json'.

    Returns:
        A tuple of (summary DataFrame, detailed results dictionary).
    """
    results_dir = Path(results_dir)
    summary_file = results_dir / "batch_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    df = pd.read_csv(summary_file)
    detailed_file = results_dir / "complete_results.json"
    detailed_results = None
    if detailed_file.exists():
        with open(detailed_file, 'r') as f:
            detailed_results = json.load(f)
    
    return df, detailed_results


def analyze_success_rates(df: pd.DataFrame) -> None:
    """
    Print a detailed analysis of success rates by category.

    Args:
        df: DataFrame containing evaluation results.
    """
    print("SUCCESS RATE ANALYSIS")
    print("=" * 50)
    
    total = len(df)
    successful = df['success'].sum()
    print(f"Overall success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    print("\nBy Evaluation Type:")
    eval_success = df.groupby('evaluation_type')['success'].agg(['count', 'sum', 'mean'])
    eval_success['percentage'] = eval_success['mean'] * 100
    print(eval_success[['count', 'sum', 'percentage']])
    
    print("\nBy Search Algorithm:")
    search_success = df.groupby('search_algorithm')['success'].agg(['count', 'sum', 'mean'])
    search_success['percentage'] = search_success['mean'] * 100
    print(search_success[['count', 'sum', 'percentage']])
    
    print("\nBy Event Log:")
    log_success = df.groupby('xes_name')['success'].agg(['count', 'sum', 'mean'])
    log_success['percentage'] = log_success['mean'] * 100
    print(log_success[['count', 'sum', 'percentage']])
    
    print("\nBy Log Coverage:")
    coverage_success = df.groupby('log_coverage')['success'].agg(['count', 'sum', 'mean'])
    coverage_success['percentage'] = coverage_success['mean'] * 100
    print(coverage_success[['count', 'sum', 'percentage']])


def analyze_performance(df: pd.DataFrame) -> None:
    """
    Analyze performance metrics (duration) for successful runs.

    Args:
        df: DataFrame containing evaluation results.
    """
    """Analyze performance metrics."""
    print("\n\nPERFORMANCE ANALYSIS")
    print("=" * 50)
    
    successful_df = df[df['success'] == True]
    
    if len(successful_df) == 0:
        print("No successful runs to analyze.")
        return
    
    print(f"Analyzing {len(successful_df)} successful runs...")
    
    print(f"\nDuration Statistics (seconds):")
    duration_stats = successful_df['duration'].describe()
    print(duration_stats)
    
    print(f"\nAverage Duration by Search Algorithm:")
    search_duration = successful_df.groupby('search_algorithm')['duration'].agg(['mean', 'std', 'count'])
    print(search_duration.sort_values('mean'))
    
    print(f"\nAverage Duration by Evaluation Type:")
    eval_duration = successful_df.groupby('evaluation_type')['duration'].agg(['mean', 'std', 'count'])
    print(eval_duration.sort_values('mean'))
    
    print(f"\nAverage Duration by Log Coverage:")
    coverage_duration = successful_df.groupby('log_coverage')['duration'].agg(['mean', 'std', 'count'])
    print(coverage_duration.sort_values('log_coverage'))


def create_visualizations(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """
    Create visualization plots and save them to the specified directory.

    Args:
        df: DataFrame containing evaluation results.
        output_dir: Directory where plots will be saved.
    """
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plt.style.use('default')
    
    plt.figure(figsize=(12, 6))
    search_success = df.groupby('search_algorithm')['success'].mean() * 100
    search_success.plot(kind='bar')
    plt.title('Success Rate by Search Algorithm')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'success_by_algorithm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    log_success = df.groupby('xes_name')['success'].mean() * 100
    log_success.plot(kind='bar')
    plt.title('Success Rate by Event Log')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'success_by_log.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    successful_df = df[df['success'] == True]
    if len(successful_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(successful_df['duration'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Execution Duration (Successful Runs)')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_dir / 'duration_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        successful_df.boxplot(column='duration', by='search_algorithm', figsize=(12, 8))
        plt.title('Duration by Search Algorithm (Successful Runs)')
        plt.suptitle('')  # Remove default title
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'duration_by_algorithm.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    if len(df) > 0:
        pivot_data = df.pivot_table(
            values='success', 
            index='search_algorithm', 
            columns='log_coverage', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', fmt='.2f', 
                   cbar_kws={'label': 'Success Rate'})
        plt.title('Success Rate Heatmap: Algorithm vs Log Coverage')
        plt.tight_layout()
        plt.savefig(output_dir / 'success_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualization plots saved to: {output_dir}")


def generate_report(
    df: pd.DataFrame, 
    detailed_results: Optional[Dict[str, Any]], 
    output_file: Union[str, Path]
) -> None:
    """
    Generate a comprehensive text report of the evaluation results.

    Args:
        df: DataFrame containing evaluation results.
        detailed_results: Optional dictionary with detailed JSON data.
        output_file: Path to the output text file.
    """
    """Generate a comprehensive report."""
    with open(output_file, 'w') as f:
        f.write("XES2PDDL Batch Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        total = len(df)
        successful = df['success'].sum()
        f.write(f"Total evaluations: {total}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {total - successful}\n")
        f.write(f"Success rate: {successful/total*100:.1f}%\n\n")
        
        f.write("Configuration Summary:\n")
        f.write(f"Event logs tested: {df['xes_name'].nunique()}\n")
        f.write(f"Search algorithms tested: {df['search_algorithm'].nunique()}\n")
        f.write(f"Log coverage levels tested: {df['log_coverage'].nunique()}\n")
        f.write(f"Evaluation types: {', '.join(df['evaluation_type'].unique())}\n\n")
        
        f.write("Success Rate by Category:\n")
        f.write("-" * 30 + "\n")
        
        for category in ['evaluation_type', 'search_algorithm', 'xes_name', 'log_coverage']:
            f.write(f"\nBy {category.replace('_', ' ').title()}:\n")
            breakdown = df.groupby(category)['success'].agg(['count', 'sum', 'mean'])
            breakdown['percentage'] = breakdown['mean'] * 100
            f.write(breakdown.to_string())
            f.write("\n")
        
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0:
            f.write(f"\nPerformance Analysis (Successful Runs):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average duration: {successful_df['duration'].mean():.2f}s\n")
            f.write(f"Median duration: {successful_df['duration'].median():.2f}s\n")
            f.write(f"Max duration: {successful_df['duration'].max():.2f}s\n")
            f.write(f"Min duration: {successful_df['duration'].min():.2f}s\n")
    
    print(f"Report saved to: {output_file}")


def main() -> Optional[int]:
    """
    Main entry point for the analysis script.

    Returns:
        Exit code (1 for error, None for success).
    """
    parser = argparse.ArgumentParser(description="Analyze XES2PDDL batch evaluation results")
    parser.add_argument("results_dir", help="Directory containing batch results")
    parser.add_argument("--output", "-o", default="analysis", help="Output directory for analysis results")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--report-only", action="store_true", help="Generate only text report")
    
    args = parser.parse_args()
    
    try:
        df, detailed_results = load_batch_results(args.results_dir)
        
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        if not args.report_only:
            analyze_success_rates(df)
            analyze_performance(df)
        
        if not args.no_plots and not args.report_only:
            try:
                create_visualizations(df, output_dir / "plots")
            except ImportError:
                print("\nWarning: matplotlib/seaborn not available. Skipping plots.")
            except Exception as e:
                print(f"\nWarning: Error creating plots: {e}")
        
        report_file = output_dir / "evaluation_report.txt"
        generate_report(df, detailed_results, report_file)
        
        print(f"\nAnalysis complete. Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
