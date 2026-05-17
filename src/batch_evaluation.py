import argparse
import csv
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import signal
import sys
import subprocess
import time
from typing import List, Dict, Optional, Any, Tuple, Set, Union, Callable


class BatchEvaluator:
    """
    Evaluator for running batches of process mining and planning tasks.

    Attributes:
        config_name (Optional[str]): Name of the predefined configuration to use.
        output_dir (Path): Directory where results and logs will be stored.
        resume (bool): Whether to resume from a previous checkpoint.
        checkpoint_file (Path): Path to the checkpoint JSON file.
        interrupted (bool): Flag to track if the process was interrupted.
        predefined_configs (Dict[str, Dict[str, Any]]): Mapping of preset names to configurations.
        default_config (Dict[str, Any]): The default configuration used if none specified.
        config (Dict[str, Any]): The active configuration for the current run.
    """

    config_name: Optional[str]  # Name of the predefined configuration to use (e.g., 'planning', 'full')
    output_dir: Path  # Directory where evaluation results, logs, and summaries will be stored
    resume: bool  # Whether to attempt resuming from a previously saved checkpoint
    checkpoint_file: Path  # Path to the JSON file storing the state of the current batch execution
    interrupted: bool  # Flag indicating if the user has requested to stop the evaluation (SIGINT/SIGTERM)
    predefined_configs: Dict[str, Dict[str, Any]]  # Mapping of preset names to their configuration parameters
    default_config: Dict[str, Any]  # The default configuration parameters used if no preset is selected
    config: Dict[str, Any]  # The actual configuration being used for the current run
    def __init__(
        self, 
        config_name: Optional[str] = None, 
        output_dir: Union[str, Path] = "../evaluation/batch_results", 
        checkpoint_file: Optional[Union[str, Path]] = None, 
        resume: bool = False
    ) -> None:
        self.config_name = config_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.checkpoint_file = checkpoint_file or (self.output_dir / "checkpoint.json")
        self.interrupted = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.predefined_configs = self.get_predefined_configs()
        self.default_config = self.predefined_configs["single_log"].copy()
        self.config = self.load_config()
    
    def get_predefined_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Define and return the set of predefined batch configurations.

        Returns:
            Dictionary mapping config names (e.g., 'planning', 'full') to their parameters.
        """
        # Define which logs need as activity names (concept:name + lifecycle:transition)
        activity_classifier_logs = {
            "bpic_2013_cp.xes": True,
            "bpic_2013_i.xes": True,
            "bpic_2012.xes": True,
            "bpic_2012_a.xes": True,
            "bpic_2012_o.xes": True,
            "bpic_2012_w.xes": True,
            "bpic_2012_wc.xes": True,
            "env_permit.xes": True
        }
        
        return {
            "planning": {
                "event_logs": [
                    "sepsis.xes", "bpic_2013_cp.xes", "bpic_2013_i.xes", "bpic_2012_a.xes", 
                    "bpic_2012_o.xes", "bpic_2012_w.xes", "bpic_2012_wc.xes"
                ],
                "search_algorithms": ["astar_blind", "astar_hff", "astar_lmcut",
                                      "eager_greedy_hff", "seq_sat_lama_2011"],
                "discovery_algorithms": ["inductive"],
                "coverage_levels": [0.001],
                "evaluations": ["planning"],
                "use_activity_classifier": activity_classifier_logs,
                "timeout": None,
                "max_samples": 500,
                "max_traces": None
            },
            "suffix": {
                "event_logs": [
                    "sepsis.xes", "bpic_2013_cp.xes", "bpic_2013_i.xes", "bpic_2012_a.xes", 
                    "bpic_2012_o.xes", "bpic_2012_w.xes", "bpic_2012_wc.xes"
                ],
                "search_algorithms": ["astar_blind"],
                "discovery_algorithms": ["alpha", "inductive", "heuristics", "ilp"],
                "coverage_levels": [0.001],
                "evaluations": ["suffix_prediction"],
                "use_activity_classifier": activity_classifier_logs,
                "timeout": None,
                "max_samples": 500,
                "max_traces": None
            },
            "outcome": {
                "event_logs": [
                    "sepsis.xes", "bpic_2013_cp.xes", "bpic_2013_i.xes", "bpic_2012_a.xes", 
                    "bpic_2012_o.xes", "bpic_2012_w.xes", "bpic_2012_wc.xes"
                ],
                "search_algorithms": ["astar_blind"],
                "discovery_algorithms": ["alpha", "inductive", "heuristics", "ilp"],
                "coverage_levels": [0.001],
                "evaluations": ["outcome_prediction"],
                "use_activity_classifier": activity_classifier_logs,
                "timeout": None,
                "max_samples": 500,
                "max_traces": None
            },
            "single_log": {
                "event_logs": ["sepsis.xes"],
                "search_algorithms": ["astar_blind", "astar_hadd", "astar_hff", "astar_lmcut",
                                      "eager_greedy_blind", "eager_greedy_hadd", "eager_greedy_hff", "eager_greedy_lmcut",
                                      "seq_sat_lama_2011"],
                "discovery_algorithms": ["alpha", "inductive", "heuristics", "ilp"],
                "coverage_levels": [0.5, 0.8, 0.99],
                "evaluations": ["planning", "suffix_prediction", "outcome_prediction"],
                "use_activity_classifier": activity_classifier_logs,
                "timeout": None,
                "max_samples": 700,
                "max_traces": None
            },
            "full": {
                "event_logs": [
                    "sepsis.xes", "helpdesk.xes", "env_permit.xes",
                    "bpic_2012.xes", "bpic_2013_cp.xes", "bpic_2017.xes",
                    "bpic_2012_a.xes", "bpic_2012_o.xes", "bpic_2012_w.xes",
                    "bpic_2012_wc.xes", "bpic_2013_i.xes"
                ],
                "search_algorithms": ["astar_blind", "astar_hadd", "astar_hff", "astar_lmcut",
                                      "eager_greedy_blind", "eager_greedy_hadd", "eager_greedy_hff", "eager_greedy_lmcut",
                                      "seq_sat_lama_2011"],
                "discovery_algorithms": ["alpha", "inductive", "heuristics", "ilp"],
                "coverage_levels": [0.001],
                "evaluations": ["planning", "suffix_prediction", "outcome_prediction"],
                "use_activity_classifier": activity_classifier_logs,
                "timeout": None,
                "max_samples": 700,
                "max_traces": None
            }
        }
            
    def _signal_handler(self, signum: int, frame: Optional[Any]) -> None:
        """Handle interrupt signals to ensure progress is saved."""
        print(f"\nReceived signal {signum}. Saving checkpoint and exiting...")
        self.interrupted = True

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load the checkpoint file if it exists.

        Returns:
            A dictionary containing completed configurations and batch metadata.
        """
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                print(f"Loaded checkpoint from {self.checkpoint_file}")
                return checkpoint
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Could not load checkpoint from {self.checkpoint_file}, starting fresh")
                return {"completed_configs": [], "batch_id": None, "config_hash": None}
        return {"completed_configs": [], "batch_id": None, "config_hash": None}
    
    def save_checkpoint(self, completed_configs: List[str], batch_id: str, config_hash: str) -> None:
        """
        Save the current progress to a checkpoint file.

        Args:
            completed_configs: List of successfully completed run IDs.
            batch_id: Identifier for the current batch.
            config_hash: Hash representing the configuration state.
        """
        checkpoint = {
            "completed_configs": completed_configs,
            "batch_id": batch_id,
            "config_hash": config_hash,
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def get_config_hash(self) -> str:
        """Generate an MD5 hash of the current configuration."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_run_id(self, config: Dict[str, Any]) -> str:
        """Generate a unique run identifier for a specific configuration."""
        coverage_str = f"{config['log_coverage']:.6f}".rstrip('0').rstrip('.')
        return f"{config['evaluation_type']}_{config['xes_name'].replace('.xes', '')}_{config['search_algorithm']}_{config['discovery_algorithm']}_cov{coverage_str}"
        
    def load_config(self) -> Dict[str, Any]:
        """Load the active configuration based on the provided config name."""
        if self.config_name and self.config_name in self.predefined_configs:
            return self.predefined_configs[self.config_name].copy()
        else:
            return self.default_config.copy()
    
    def save_config_template(self, filename: str = "batch_config_template.json") -> None:
        """Save a template JSON configuration to a file."""
        template = {
            "description": "Template configuration for XES2PDDL batch evaluation",
            "event_logs": [
                "sepsis.xes",
                "helpdesk.xes", 
                "env_permit.xes",
                "bpic_2013_cp.xes"
            ],
            "search_algorithms": [
                "astar_lmcut",
                "lazy_greedy_ff",
                "ehc_ff"
            ],
            "discovery_algorithms": ["alpha"],
            "coverage_levels": [0.5, 0.8, 0.99],
            "evaluations": ["planning", "suffix_prediction", "outcome_prediction"],
            "use_activity_classifier": {
                "bpic_2013_cp.xes": True,
                "bpic_2013_i.xes": True
            },
            "timeout": 300
            ,"max_traces": None
        }
        with open(filename, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"Configuration template saved to {filename}")
    
    def list_predefined_configs(self) -> None:
        """Print a summary of all predefined configurations to stdout."""
        print("Available predefined configurations:")
        print("=" * 50)
        for name, config in self.predefined_configs.items():
            logs_count = len(config["event_logs"])
            algos_count = len(config["search_algorithms"])
            eval_count = len(config["evaluations"])
            coverage_levels = config.get("coverage_levels", [0.8])
            coverage_count = len(coverage_levels)
            
            estimated_total = logs_count * algos_count * coverage_count * eval_count
            
            print(f"{name:20} - {logs_count} logs × {algos_count} algorithms × {coverage_count} coverage levels × {eval_count} evaluations ≈ {estimated_total} runs")
            print(f"{'':20}   Coverage levels: {[int(c*100) for c in coverage_levels]}%")
            print(f"{'':20}   Timeout: {config['timeout']}s")
            print(f"{'':20}   Logs: {', '.join(config['event_logs'][:3])}{'...' if logs_count > 3 else ''}")
            print()
    
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """
        Generate all individual configurations (runs) from the active batch config.

        Returns:
            A list of configuration dictionaries.
        """
        configurations = []
        discovery_algorithms = self.config.get("discovery_algorithms", ["alpha"])
        coverage_levels = self.config.get("coverage_levels", [0.8])
        
        print(f"Generating configurations with coverage levels: {coverage_levels}")
        
        for eval_type in self.config["evaluations"]:
            for log in self.config["event_logs"]:
                for search, discovery, coverage in product(
                    self.config["search_algorithms"],
                    discovery_algorithms,
                    coverage_levels
                ):
                    config = {
                        "evaluation_type": eval_type,
                        "xes_name": log,
                        "search_algorithm": search,
                        "discovery_algorithm": discovery,
                        "log_coverage": coverage,
                        "use_activity_classifier": self.config.get("use_activity_classifier", {}).get(log, False),
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    configurations.append(config)
        
        return configurations
    
    def run_evaluation(self, config: Dict[str, Any], batch_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a single evaluation task based on the provided configuration.

        Args:
            config: The configuration for this specific run.
            batch_id: Optional batch identifier for file naming.

        Returns:
            A dictionary containing the run results (success, duration, etc.).
        """
        eval_type = config["evaluation_type"]
        if eval_type == "planning":
            script = "eval_planning.py"
        elif eval_type == "suffix_prediction":
            script = "eval_suffix_prediction.py"
        elif eval_type == "outcome_prediction":
            script = "eval_outcome_prediction.py"
        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")
        
        cmd = [
            sys.executable, script,
            "--xes_name", config["xes_name"],
            "--search", config["search_algorithm"],
            "--discovery_algorithm", config["discovery_algorithm"],
            "--log_coverage", str(config["log_coverage"])
        ]
        
        # Add activity classifier flag if enabled for this log
        if config.get("use_activity_classifier", False):
            cmd.append("--use_activity_classifier")
        # Add sample cap parameter
        cmd.extend(["--max-samples", str(self.config.get("max_samples", 700))])
        # Pass optional max traces to the evaluation script to cap traces read
        if self.config.get("max_traces") is not None:
            cmd.extend(["--max-traces", str(self.config.get("max_traces"))])
        
        timestamp = batch_id or config['timestamp']
        coverage_str = f"{config['log_coverage']:.6f}".rstrip('0').rstrip('.')
        run_id = f"{eval_type}_{config['xes_name'].replace('.xes', '')}_{config['search_algorithm']}_{config['discovery_algorithm']}_cov{coverage_str}_{timestamp}"
        run_output_dir = self.output_dir / run_id
        run_output_dir.mkdir(exist_ok=True)
        stdout_file = run_output_dir / "stdout.txt"
        stderr_file = run_output_dir / "stderr.txt"
        config_file = run_output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        try:
            start_time = time.time()
            
            with open(stdout_file, 'w') as stdout, open(stderr_file, 'w') as stderr:
                process = subprocess.run(
                    cmd,
                    cwd=".",
                    stdout=stdout,
                    stderr=stderr,
                    timeout=self.config.get("timeout", 300),
                    text=True
                )
            
            end_time = time.time()
            duration = end_time - start_time
            result = {
                "config": config,
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "duration": duration,
                "output_dir": str(run_output_dir),
                "timestamp": datetime.now().isoformat()
            }
            
            result_file = run_output_dir / "result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"  {status} in {duration:.2f}s")
            
            return result
            
        except subprocess.TimeoutExpired:
            duration = self.config.get("timeout", 300)
            result = {
                "config": config,
                "success": False,
                "return_code": -1,
                "duration": duration,
                "error": "Timeout",
                "output_dir": str(run_output_dir),
                "timestamp": datetime.now().isoformat()
            }
            
            result_file = run_output_dir / "result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"  TIMEOUT after {duration}s")
            return result
            
        except Exception as e:
            result = {
                "config": config,
                "success": False,
                "return_code": -2,
                "duration": 0,
                "error": str(e),
                "output_dir": str(run_output_dir),
                "timestamp": datetime.now().isoformat()
            }
            
            result_file = run_output_dir / "result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"  ERROR: {e}")
            return result
    
    def run_batch(self) -> List[Dict[str, Any]]:
        """
        Execute the entire batch of evaluations.

        Returns:
            A list of result dictionaries for all runs in the batch.
        """
        configurations = self.generate_configurations()
        
        checkpoint = self.load_checkpoint() if self.resume else {"completed_configs": [], "batch_id": None, "config_hash": None}
        current_config_hash = self.get_config_hash()
        
        batch_id = checkpoint.get("batch_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.resume and checkpoint.get("config_hash") != current_config_hash:
            print("Warning: Configuration has changed since last checkpoint. Starting fresh.")
            checkpoint = {"completed_configs": [], "batch_id": batch_id, "config_hash": current_config_hash}
        
        completed_run_ids = set(checkpoint["completed_configs"])
        
        if self.resume and completed_run_ids:
            remaining_configs = []
            for config in configurations:
                run_id = self.get_run_id(config)
                if run_id not in completed_run_ids:
                    remaining_configs.append(config)
            configurations = remaining_configs
            print(f"Resuming batch: {len(completed_run_ids)} successfully completed, {len(configurations)} remaining")
        else:
            print(f"Starting new batch evaluation with {len(configurations)} configurations...")
        
        if not configurations:
            print("All configurations already completed successfully!")
            return []
        
        print(f"Output directory: {self.output_dir}")
        print(f"Checkpoint file: {self.checkpoint_file}")
        
        results = []
        completed_configs = list(completed_run_ids)
        
        summary_file = self.output_dir / f"batch_summary_{batch_id}.csv"
        append_mode = self.resume and summary_file.exists()
        
        with open(summary_file, 'a' if append_mode else 'w', newline='') as csvfile:
            fieldnames = [
                'evaluation_type', 'xes_name', 'search_algorithm', 'discovery_algorithm', 'log_coverage',
                'success', 'duration', 'return_code', 'timestamp', 'output_dir'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not append_mode:
                writer.writeheader()
            
            total_configs = len(configurations) + len(completed_run_ids)
            start_index = len(completed_run_ids)
            
            for i, config in enumerate(configurations, 1):
                if self.interrupted:
                    print("\nInterrupted. Saving checkpoint...")
                    break
                    
                current_index = start_index + i
                run_id = self.get_run_id(config)
                print(f"\n[{current_index}/{total_configs}] Running {run_id}...", end="")
                
                result = self.run_evaluation(config, batch_id)
                results.append(result)
                
                if result['success']:
                    run_id = self.get_run_id(config)
                    completed_configs.append(run_id)
                    self.save_checkpoint(completed_configs, batch_id, current_config_hash)
                
                csv_row = {
                    'evaluation_type': result['config']['evaluation_type'],
                    'xes_name': result['config']['xes_name'],
                    'search_algorithm': result['config']['search_algorithm'],
                    'discovery_algorithm': result['config']['discovery_algorithm'],
                    'log_coverage': result['config']['log_coverage'],
                    'success': result['success'],
                    'duration': result['duration'],
                    'return_code': result['return_code'],
                    'timestamp': result['timestamp'],
                    'output_dir': result['output_dir']
                }
                writer.writerow(csv_row)
                csvfile.flush()
        
        if results:
            session_results_file = self.output_dir / f"session_results_{batch_id}_{datetime.now().strftime('%H%M%S')}.json"
            with open(session_results_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        if not self.interrupted:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                print(f"Batch completed successfully. Checkpoint file removed.")
        
        self.print_summary(results)
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print a summary of the batch evaluation results to stdout."""
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total - successful
        avg_duration = sum(r['duration'] for r in results) / total if total > 0 else 0 
        print(f"\n{'='*60}")
        print("BATCH EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total evaluations: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/total*100:.1f}%")
        print(f"Average duration: {avg_duration:.2f}s")
        
        if failed > 0:
            print(f"\nFailed evaluations:")
            for result in results:
                if not result['success']:
                    config = result['config']
                    error = result.get('error', f"Return code: {result['return_code']}")
                    coverage_str = f"{config['log_coverage']:.6f}".rstrip('0').rstrip('.')
                    print(f"  - {config['evaluation_type']} | {config['xes_name']} | {config['search_algorithm']} | cov={coverage_str} | Error: {error}")
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Summary CSV: {self.output_dir}/batch_summary.csv")


def main() -> None:
    """Main entry point for the batch evaluation script."""
    parser = argparse.ArgumentParser(description="Batch evaluation for XES2PDDL framework")
    parser.add_argument("--preset", type=str, choices=[
        "planning", "suffix", "single_log", "outcome", "full"], help="Use predefined configuration")
    parser.add_argument("--output", type=str, default="../evaluation/batch_results", help="Output directory")
    parser.add_argument("--log", type=str, help="Event log to use for single log evaluation (overrides preset log)")
    parser.add_argument("--save-template", action="store_true", help="Save configuration template and exit")
    parser.add_argument("--preview", action="store_true", help="Preview configurations without running")
    parser.add_argument("--list-presets", action="store_true", help="List available predefined configurations")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoint")
    parser.add_argument("--checkpoint-file", type=str, help="Custom checkpoint file path (default: output_dir/checkpoint.json)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of generated samples (prefixes) to evaluate overall per evaluation. Overrides preset default.")
    parser.add_argument("--max-traces", type=int, help="Optional: maximum number of traces to read from the log. Overrides preset default.")
    
    args = parser.parse_args()
    
    if args.list_presets:
        evaluator = BatchEvaluator(output_dir=args.output)
        evaluator.list_predefined_configs()
        return
        
    if args.save_template:
        evaluator = BatchEvaluator(output_dir=args.output)
        evaluator.save_config_template()
        return
        
    config_name = args.preset
    evaluator = BatchEvaluator(
        config_name=config_name,
        output_dir=args.output,
        checkpoint_file=args.checkpoint_file,
        resume=args.resume
    )
    
    if args.preset:
        print(f"Using predefined configuration: {args.preset}")
    else:
        print("No configuration specified, using default configuration (single_log)")
    
    if args.log:
        evaluator.config["event_logs"] = [args.log]
        print(f"Overriding event log with: {args.log}")
    if args.max_samples:
        evaluator.config['max_samples'] = args.max_samples
        print(f"Overriding global max samples (prefixes) per evaluation: {args.max_samples}")
    if args.max_traces:
        evaluator.config['max_traces'] = args.max_traces
        print(f"Overriding max traces per log: {args.max_traces}")
            
    if args.resume:
        print("Resume mode enabled - will skip already completed configurations")
        
    if args.preview:
        configs = evaluator.generate_configurations()
        print(f"Generated {len(configs)} configurations:")
        for i, config in enumerate(configs, 1):
            coverage_str = f"{config['log_coverage']:.6f}".rstrip('0').rstrip('.')
            print(f"{i:3d}. {config['evaluation_type']:15} | {config['xes_name']:20} | {config['search_algorithm']:15} | {config['discovery_algorithm']:10} | cov={coverage_str}")
        return
        
    evaluator.run_batch()


if __name__ == "__main__":
    main()
