"""Orchestrator for the evaluation harness.

Reads CLI arguments, downloads logs, runs the evaluation pipeline for each log,
and writes per-log results plus a rolling cross-log summary.

Usage::

    conda run -n temporalXes2Plan --cwd src python -m evaluation.run_evaluation \\
        --output-dir results/ --n-test-cases 50 --planner-timeout 60
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.eval_api import EvalAPI
from evaluation.test_case_selector import split as _split_log
from evaluation.trace_sampler import sample_prefix as _sample_prefix
from evaluation.query_builder import build_q1, build_q2, build_q3, is_q3_reachable, QuerySpec
from evaluation.metrics_collector import q1_metrics, q2_metrics, q3_metrics
from evaluation.plan_validator import validate_plan
from evaluation.planner_runner_with_retry import RetryConfig, run_with_retry as _run_with_retry
from evaluation.log_downloader import download_if_needed, get_log_selection
from evaluation.report_generator import (
    LogResult, QueryResult, write_log_result, write_cross_log_summary,
)
from web.serializer import serialize_parse_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """All tunable parameters for the evaluation harness."""

    log_ids: Optional[List[str]]
    test_pct: float
    min_test_cases: int
    max_test_cases: int
    min_prefix_pct: float
    max_prefix_pct: float
    seed: int
    algorithm: str
    coverage: float
    planner_timeout: int
    planner_memory_mb: int
    max_retries: int
    retry_delay: float
    cache_dir: Path
    output_dir: Path
    force_download: bool
    resume: bool
    cost_weight: float
    csv_mapping: Optional[Dict[str, Optional[str]]]


# ---------------------------------------------------------------------------
# Top-level evaluation per log
# ---------------------------------------------------------------------------

def evaluate_log(
    log_id: str,
    log_name: str,
    log_path: Path,
    log_fmt: str,
    output_dir: Path,
    api: EvalAPI,
    cfg: EvalConfig,
) -> LogResult:
    """Run the full evaluation pipeline for a single event log.

    Args:
        log_id: Event Log ID from the metadata CSV.
        log_name: Human-readable log name.
        log_path: Path to the cached log file.
        log_fmt: "xes" or "csv".
        output_dir: Root results directory.
        api: EvalAPI instance (stateless; shared across logs).
        cfg: Evaluation configuration.

    Returns:
        LogResult capturing all per-query outcomes and metadata.
    """
    log_out_dir = output_dir / log_id
    failures_dir = log_out_dir / "failures"

    # 1. Validate timestamps
    try:
        vr = api.validate_log(str(log_path), mapping=cfg.csv_mapping)
    except Exception as exc:
        logger.error("[%s] validate_log failed: %s", log_id, exc)
        return LogResult(
            log_id=log_id, log_name=log_name, log_fmt=log_fmt,
            n_train_cases=0, n_test_cases=0, n_activities=0,
            pipeline_ok=False, pipeline_error=str(exc),
        )

    if vr.missing_timestamp:
        logger.error("[%s] Log has missing timestamps — skipping.", log_id)
        return LogResult(
            log_id=log_id, log_name=log_name, log_fmt=log_fmt,
            n_train_cases=0, n_test_cases=0, n_activities=0,
            pipeline_ok=False, pipeline_error="missing_timestamps",
        )

    # 2. Train/test split
    tts = _split_log(
        str(log_path), log_fmt,
        test_pct=cfg.test_pct,
        min_test_cases=cfg.min_test_cases,
        max_test_cases=cfg.max_test_cases,
        seed=cfg.seed,
        mapping=cfg.csv_mapping,
    )

    with tts:
        # 3. Parse on training data
        try:
            parse_result = api.parse(
                str(tts.train_path),
                coverage_percentage=cfg.coverage,
                discovery_algorithm=cfg.algorithm,
            )
        except Exception as exc:
            logger.error("[%s] parse failed: %s", log_id, exc)
            return LogResult(
                log_id=log_id, log_name=log_name, log_fmt=log_fmt,
                n_train_cases=tts.n_train, n_test_cases=tts.n_test, n_activities=0,
                pipeline_ok=False, pipeline_error=str(exc),
            )

        n_activities = len(parse_result.petri_net_model.activities)

        # 4. Build domain (durative + costs required for weighted metric)
        try:
            domain_text, variant_effects = api.build_domain_with_variant_effects(
                parse_result, use_durative=True, use_costs=True
            )
        except Exception as exc:
            logger.error("[%s] build_domain failed: %s", log_id, exc)
            return LogResult(
                log_id=log_id, log_name=log_name, log_fmt=log_fmt,
                n_train_cases=tts.n_train, n_test_cases=tts.n_test,
                n_activities=n_activities,
                pipeline_ok=False, pipeline_error=str(exc),
            )

        # 5. Serialize Petri net for downstream modules
        serialized = serialize_parse_result(parse_result)

        log_out_dir.mkdir(parents=True, exist_ok=True)
        (log_out_dir / "domain.pddl").write_text(domain_text, encoding="utf-8")
        logger.info(
            "[%s] Domain built (%d activities, durative=True).", log_id, n_activities
        )

        # 6-7. Sample prefix and run Q1/Q2/Q3 for each test trace
        query_results: List[QueryResult] = []
        for trace in tts.test_cases:
            trace_id = str(trace.attributes.get("concept:name", "unknown"))
            prefix = _sample_prefix(
                trace, serialized, api,
                min_prefix_pct=cfg.min_prefix_pct,
                max_prefix_pct=cfg.max_prefix_pct,
                seed=cfg.seed,
            )
            if prefix is None:
                logger.warning("[%s] %s: prefix sampling failed — skipping trace.", log_id, trace_id)
                continue

            # Q1 — process completion, no deadline
            q1_spec = build_q1(prefix, cfg.cost_weight)
            query_results.append(
                _run_query(log_id, trace_id, q1_spec, domain_text, api, cfg, serialized, failures_dir, prefix, variant_effects)
            )

            # Q2 — completion within remaining time budget
            q2_spec = build_q2(prefix, cfg.cost_weight)
            if q2_spec is not None:
                query_results.append(
                    _run_query(log_id, trace_id, q2_spec, domain_text, api, cfg, serialized, failures_dir, prefix, variant_effects)
                )

            # Q3 — completion within budget + attribute constraints
            q3_spec = build_q3(prefix, serialized, cfg.cost_weight)
            if q3_spec is not None:
                query_results.append(
                    _run_query(log_id, trace_id, q3_spec, domain_text, api, cfg, serialized, failures_dir, prefix, variant_effects)
                )
            else:
                query_id = f"{log_id}_{trace_id}_Q3"
                logger.warning("[%s] %s Q3 skipped — no discretized attributes.", log_id, trace_id)
                query_results.append(QueryResult(
                    query_id=query_id,
                    query_type="Q3",
                    trace_id=trace_id,
                    prefix_ratio=prefix.prefix_ratio,
                    n_prefix_events=len(prefix.prefix_events),
                    attempts=0,
                    solvability="skipped_no_attributes",
                    planner_duration_s=None,
                    metrics={},
                    validation=None,
                ))

        logger.info("[%s] Done — %d queries.", log_id, len(query_results))
        return LogResult(
            log_id=log_id,
            log_name=log_name,
            log_fmt=log_fmt,
            n_train_cases=tts.n_train,
            n_test_cases=tts.n_test,
            n_activities=n_activities,
            pipeline_ok=True,
            pipeline_error=None,
            queries=query_results,
        )


# ---------------------------------------------------------------------------
# Internal: run one Q1/Q2/Q3 query
# ---------------------------------------------------------------------------

def _run_query(
    log_id: str,
    trace_id: str,
    spec: QuerySpec,
    domain_text: str,
    api: EvalAPI,
    cfg: EvalConfig,
    serialized: Dict[str, Any],
    failures_dir: Path,
    prefix: Any,
    variant_effects: Optional[Dict[str, Dict[str, Any]]] = None,
) -> QueryResult:
    query_id = f"{log_id}_{trace_id}_{spec.query_type}"

    problem_text = api.build_problem(
        serialized=serialized,
        init_places=spec.init_places,
        goal_sop=spec.goal_sop,
        init_effects=spec.init_effects,
        metric=spec.metric,
        cost_weight=spec.cost_weight,
        require_completion=spec.require_completion,
        deadline=spec.deadline,
    )

    retry_cfg = RetryConfig(
        max_attempts=cfg.max_retries,
        base_delay_s=cfg.retry_delay,
        timeout_s=cfg.planner_timeout,
        memory_mb=cfg.planner_memory_mb,
    )
    result, attempts = _run_with_retry(domain_text, problem_text, api, retry_cfg, failures_dir, query_id)

    if spec.query_type == "Q1":
        metrics = q1_metrics(result, cfg.cost_weight)
    elif spec.query_type == "Q2":
        metrics = q2_metrics(result, prefix, cfg.cost_weight)
    else:
        ground_truth_reachable = is_q3_reachable(spec.goal_sop, serialized)
        metrics = q3_metrics(result, prefix, spec.goal_sop, ground_truth_reachable, cfg.cost_weight)

    validation = None
    if result.success and result.plan_steps:
        report = validate_plan(
            result.plan_steps,
            spec.init_places,
            serialized,
            init_attributes=spec.init_effects,
            variant_effects=variant_effects,
        )
        validation = {
            "valid": report.valid,
            "attribute_checked": report.attribute_checked,
            "error_step": report.error_step,
            "error_action": report.error_action,
            "error_reason": report.error_reason,
            "steps_executed": report.steps_executed,
        }

    return QueryResult(
        query_id=query_id,
        query_type=spec.query_type,
        trace_id=trace_id,
        prefix_ratio=prefix.prefix_ratio,
        n_prefix_events=len(prefix.prefix_events),
        attempts=attempts,
        solvability=result.solvability,
        planner_duration_s=result.duration_s,
        metrics=metrics,
        validation=validation,
    )


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_log_result(data: Dict[str, Any]) -> LogResult:
    queries = [
        QueryResult(
            query_id=q["query_id"],
            query_type=q["query_type"],
            trace_id=q["trace_id"],
            prefix_ratio=q["prefix_ratio"],
            n_prefix_events=q["n_prefix_events"],
            attempts=q["attempts"],
            solvability=q["solvability"],
            planner_duration_s=q["planner_duration_s"],
            metrics=q["metrics"],
            validation=q["validation"],
        )
        for q in data.get("queries", [])
    ]
    return LogResult(
        log_id=data["log_id"],
        log_name=data["log_name"],
        log_fmt=data["log_fmt"],
        n_train_cases=data["n_train_cases"],
        n_test_cases=data["n_test_cases"],
        n_activities=data["n_activities"],
        pipeline_ok=data["pipeline_ok"],
        pipeline_error=data["pipeline_error"],
        queries=queries,
    )


# ---------------------------------------------------------------------------
# CSV mapping interactive prompt
# ---------------------------------------------------------------------------

def _prompt_csv_mapping(log_path: Path) -> Dict[str, Optional[str]]:
    import csv as _csv
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = _csv.reader(f)
        headers = next(reader, [])
    logger.info("CSV log detected. Available columns: %s", headers)
    mapping: Dict[str, Optional[str]] = {}
    for key in ("case_id", "activity", "timestamp", "lifecycle"):
        val = input(f"  Column for '{key}' (leave blank to skip): ").strip()
        mapping[key] = val if val else None
    return mapping


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluation CLI."""
    p = argparse.ArgumentParser(
        prog="python -m evaluation.run_evaluation",
        description="Evaluate the XES→PDDL pipeline against a set of event logs.",
    )
    p.add_argument("--metadata", type=Path, default=Path("evaluation/data/Metadata.csv"),
                   help="Path to Metadata.csv (downloaded from Zenodo if absent).")
    p.add_argument("--cache-dir", dest="cache_dir", type=Path, default=Path("evaluation/data/cache"),
                   help="Directory for downloaded log files.")
    p.add_argument("--output-dir", dest="output_dir", type=Path, default=Path("evaluation/results"),
                   help="Directory for results.")
    p.add_argument("--log-ids", dest="log_ids", type=int,  nargs="+", default=None,
                    help="Event Log IDs to include, e.g. --log-ids LOG_001 LOG_005")
    p.add_argument("--cost-weight", dest="cost_weight", type=float, default=0.001,
                   help="α in (total-time + α * total-cost) metric.")
    p.add_argument("--test-pct", dest="test_pct", type=float, default=0.2,
                   help="Fraction of traces used as test set (e.g. 0.2 = 20%%).")
    p.add_argument("--min-test-cases", dest="min_test_cases", type=int, default=10,
                   help="Minimum number of test traces regardless of percentage.")
    p.add_argument("--max-test-cases", dest="max_test_cases", type=int, default=200,
                   help="Maximum number of test traces regardless of percentage.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/test split and prefix sampling.")
    p.add_argument("--min-prefix-pct", dest="min_prefix_pct", type=float, default=0.2,
                   help="Minimum prefix fraction.")
    p.add_argument("--max-prefix-pct", dest="max_prefix_pct", type=float, default=0.8,
                   help="Maximum prefix fraction.")
    p.add_argument("--algorithm", type=str, default="inductive",
                   help="Process discovery algorithm.")
    p.add_argument("--coverage", type=float, default=0.001,
                   help="Variant coverage threshold.")
    p.add_argument("--planner-timeout", dest="planner_timeout", type=int, default=60,
                   help="Per-query OPTIC timeout in seconds.")
    p.add_argument("--planner-memory-mb", dest="planner_memory_mb", type=int, default=4000,
                   help="Memory cap for OPTIC in MB.")
    p.add_argument("--max-retries", dest="max_retries", type=int, default=3,
                   help="Max retry attempts per query.")
    p.add_argument("--retry-delay", dest="retry_delay", type=float, default=2.0,
                   help="Base delay in seconds for linear backoff.")
    p.add_argument("--force-download", dest="force_download", action="store_true",
                   help="Re-download log files even if cached.")
    p.add_argument("--resume", action="store_true",
                   help="Skip logs whose result.json already exists.")
    p.add_argument("--csv-mapping", dest="csv_mapping", type=str, default=None,
                   help="JSON string mapping CSV columns, e.g. '{\"case_id\": \"col_a\"}'.")
    p.add_argument("--log-level", dest="log_level", type=str, default="INFO",
                   help="Python logging level.")
    return p


def main(args: argparse.Namespace) -> None:
    """Run the evaluation harness.

    Args:
        args: Parsed CLI arguments (from build_parser().parse_args()).
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / "eval.log", args.log_level)

    csv_mapping: Optional[Dict[str, Optional[str]]] = None
    if args.csv_mapping:
        csv_mapping = json.loads(args.csv_mapping)

    cfg = EvalConfig(
        log_ids=args.log_ids,
        test_pct=args.test_pct,
        min_test_cases=args.min_test_cases,
        max_test_cases=args.max_test_cases,
        min_prefix_pct=args.min_prefix_pct,
        max_prefix_pct=args.max_prefix_pct,
        seed=args.seed,
        algorithm=args.algorithm,
        coverage=args.coverage,
        planner_timeout=args.planner_timeout,
        planner_memory_mb=args.planner_memory_mb,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        cache_dir=Path(args.cache_dir),
        output_dir=output_dir,
        force_download=args.force_download,
        resume=args.resume,
        cost_weight=args.cost_weight,
        csv_mapping=csv_mapping,
    )

    selection = get_log_selection(Path(args.metadata), log_ids=cfg.log_ids)
    api = EvalAPI()
    all_results: List[LogResult] = []

    logger.info("Starting — %d logs, planner=optic", len(selection))

    for _, row in selection.iterrows():
        log_id = str(row.get("Event Log ID", row.get("log_id", "unknown")))
        log_name = str(row.get("Event Log Name", row.get("log_name", log_id)))

        result_path = output_dir / log_id / "result.json"
        if cfg.resume and result_path.exists():
            logger.info("[%s] Resuming — result.json already exists, skipping.", log_id)
            existing = _load_log_result(json.loads(result_path.read_text(encoding="utf-8")))
            all_results.append(existing)
            write_cross_log_summary(all_results, output_dir, cfg.cost_weight)
            continue

        log_path, log_fmt = download_if_needed(row, cfg.cache_dir, force=cfg.force_download)

        if log_fmt == "csv" and cfg.csv_mapping is None:
            cfg.csv_mapping = _prompt_csv_mapping(log_path)

        log_result = evaluate_log(log_id, log_name, log_path, log_fmt, output_dir, api, cfg)

        write_log_result(log_id, log_result, output_dir)
        all_results.append(log_result)
        write_cross_log_summary(all_results, output_dir, cfg.cost_weight)
        logger.info(
            "[%s] %s",
            log_id,
            "OK" if log_result.pipeline_ok else f"FAILED: {log_result.pipeline_error}",
        )

    ok_count = sum(1 for r in all_results if r.pipeline_ok)
    logger.info(
        "Done — %d/%d logs OK, summary at %s/summary.json", ok_count, len(all_results), output_dir,
    )


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Path, level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)-8s %(name)-30s %(message)s"
    logging.basicConfig(level=numeric, format=fmt, force=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(numeric)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)


if __name__ == "__main__":
    main(build_parser().parse_args())
