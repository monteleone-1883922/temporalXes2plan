"""Orchestrator for the evaluation harness.

Reads CLI arguments, downloads a single log, runs the evaluation pipeline
for it, and writes its per-log result plus summary.

Usage (from project root)::

    conda run -n temporalXes2Plan python -m src.evaluation.run_evaluation \\
        --log-id 55 --planner-timeout 60
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.eval_api import EvalAPI
from models import AnalysisConfig
from evaluation.test_case_selector import split as _split_log
from evaluation.trace_sampler import sample_prefix as _sample_prefix
from evaluation.query_builder import build_q1, build_q2, build_q3, is_q3_reachable, QuerySpec
from evaluation.metrics_collector import q1_metrics, q2_metrics, q3_metrics, sequence_alignment_score
from evaluation.planner_runner_with_retry import RetryConfig, run_with_retry as _run_with_retry
from evaluation.log_downloader import download_if_needed, get_log_selection
from evaluation.report_generator import (
    LogResult, QueryResult, write_log_result, write_log_summary,
)
from network_search.scoring import ScoreWeights
from replay.trace_replayer import TraceReplayer
from replay.plan_replayer import replay_plan
from encoding.prepared_input import PreparedDomainInput
from encoding.prepared_graph_utils import build_petrinet_model_from_prepared
from encoding.domain_builder import build_domain_with_variant_map
from core_utils import save_original_and_current, sanitize_name

logger = logging.getLogger(__name__)

# Same directory web.app.DEFAULT_DATA_DIR resolves to (PROJECT_ROOT/data) —
# publishing evaluation results here makes them visible in the Petri net editor.
WEB_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """All tunable parameters for the evaluation harness."""

    log_id: int
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
    force_rebuild: bool
    # Discretizer (Stage 1–3) params — mapped to AnalysisConfig
    kmeans_max_k: int = 5
    dominance_threshold: float = 0.30
    min_residual_points: int = 50
    min_gvf_threshold: float = 0.70
    gvf_target: float = 0.90
    min_gvf_improvement: float = 0.01
    jenks_sample_size: int = 20000
    # network_search optimizer — used by default to pick the AnalysisConfig;
    # the discretizer params above still apply as the base_config for the
    # fields the optimizer doesn't tune (see network_search/search_space.py).
    use_optimizer: bool = True
    search_n_trials: int = 30
    w_det_xor: float = 1.0
    w_det_eff: float = 1.0
    w_fb_xor: float = 0.5
    w_fb_eff: float = 0.5
    w_prune_xor: float = 1.0
    w_xor: float = 1.0
    w_eff: float = 1.0
    w_dup: float = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_to_dict(config: AnalysisConfig) -> Dict[str, Any]:
    """AnalysisConfig -> JSON-serializable dict (ignored_attributes is the
    only field that isn't already JSON-native — a Set[str])."""
    data = dataclasses.asdict(config)
    data["ignored_attributes"] = sorted(data["ignored_attributes"])
    return data


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

    log_out_dir.mkdir(parents=True, exist_ok=True)
    _log_handler = logging.FileHandler(log_out_dir / "eval.log", encoding="utf-8")
    _log_handler.setLevel(logging.DEBUG)
    _log_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)-30s %(message)s"
    ))
    logging.getLogger().addHandler(_log_handler)
    try:
        # 1. Validate timestamps
        logger.debug("[%s] Validating log: path=%s fmt=%s", log_id, log_path, log_fmt)
        try:
            vr = api.validate_log(str(log_path), mapping=cfg.csv_mapping)
        except Exception as exc:
            logger.error(
                "[%s] validate_log failed (path=%s, fmt=%s): %s",
                log_id, log_path, log_fmt, exc, exc_info=True,
            )
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
        logger.debug("[%s] Splitting log (test_pct=%.2f, max_test=%d)", log_id, cfg.test_pct, cfg.max_test_cases)
        tts = _split_log(
            str(log_path), log_fmt,
            test_pct=cfg.test_pct,
            min_test_cases=cfg.min_test_cases,
            max_test_cases=cfg.max_test_cases,
            seed=cfg.seed,
            mapping=cfg.csv_mapping,
        )

        with tts:
            logger.info("[%s] Split done — train=%d test=%d", log_id, tts.n_train, tts.n_test)

            # 3-4. Build the domain once for the whole log. If data/<log_id>/
            # current.json already exists, reuse it as the cache: skip
            # parse+encode+optimizer entirely and reconstruct prepared/
            # variant_map/domain straight from the saved JSON — exactly the
            # same reconstruction web/api.py already does for its own
            # current.json-driven endpoints (PreparedDomainInput.from_dict +
            # build_petrinet_model_from_prepared + build_domain_with_variant_map),
            # so no XES re-parse or optimizer re-run happens on repeat
            # evaluations of the same log. --force-rebuild bypasses this and
            # always rebuilds from scratch (log_path is tts.train_path, never
            # the original log, so the optimizer's own internal scoring split
            # and its final rebuild "on the complete log" both stay entirely
            # within evaluation's own training data, never touching the
            # held-out test traces in tts.test_cases).
            web_config_dir = WEB_DATA_DIR / log_id
            current_path = web_config_dir / "current.json"
            use_cache = current_path.exists() and not cfg.force_rebuild

            try:
                if use_cache:
                    logger.info(
                        "[%s] Reusing cached domain from %s (skip parse/encode/optimizer). "
                        "Use --force-rebuild to rebuild.", log_id, current_path,
                    )
                    serialized = json.loads(current_path.read_text(encoding="utf-8"))
                    prepared = PreparedDomainInput.from_dict(serialized)
                    meta = serialized.get("metadata", {})
                    petri_net_model = build_petrinet_model_from_prepared(
                        prepared, meta.get("start_place"), meta.get("end_place"),
                    )
                    analysis_cfg = AnalysisConfig(
                        kmeans_max_k=cfg.kmeans_max_k,
                        dominance_threshold=cfg.dominance_threshold,
                        min_residual_points=cfg.min_residual_points,
                        min_gvf_threshold=cfg.min_gvf_threshold,
                        gvf_target=cfg.gvf_target,
                        min_gvf_improvement=cfg.min_gvf_improvement,
                        jenks_sample_size=cfg.jenks_sample_size,
                    )
                    domain, variant_map = build_domain_with_variant_map(
                        prepared, config=analysis_cfg,
                        use_durative=True, use_costs=True,
                    )
                    n_activities = len(petri_net_model.activities)
                    used_optimizer = False
                    search_summary = None
                else:
                    logger.debug(
                        "[%s] Building network (algorithm=%s, coverage=%.4f, optimizer=%s)",
                        log_id, cfg.algorithm, cfg.coverage, cfg.use_optimizer,
                    )
                    discretizer_cache = log_out_dir / "discretizer_cache.json"
                    analysis_cfg = AnalysisConfig(
                        kmeans_max_k=cfg.kmeans_max_k,
                        dominance_threshold=cfg.dominance_threshold,
                        min_residual_points=cfg.min_residual_points,
                        min_gvf_threshold=cfg.min_gvf_threshold,
                        gvf_target=cfg.gvf_target,
                        min_gvf_improvement=cfg.min_gvf_improvement,
                        jenks_sample_size=cfg.jenks_sample_size,
                    )
                    search_weights = ScoreWeights(
                        w_det_xor=cfg.w_det_xor, w_det_eff=cfg.w_det_eff,
                        w_fb_xor=cfg.w_fb_xor, w_fb_eff=cfg.w_fb_eff,
                        w_prune_xor=cfg.w_prune_xor, w_xor=cfg.w_xor, w_eff=cfg.w_eff,
                        w_dup=cfg.w_dup,
                    ) if cfg.use_optimizer else None

                    build_result = api.build_network(
                        str(tts.train_path),
                        discovery_algorithm=cfg.algorithm,
                        coverage_percentage=cfg.coverage,
                        use_durative=True,
                        use_costs=True,
                        config=analysis_cfg,
                        search=cfg.use_optimizer,
                        search_n_trials=cfg.search_n_trials,
                        search_seed=cfg.seed,
                        search_weights=search_weights,
                        snapshot_dir=str(log_out_dir / "analysis_debug"),
                        discretizer_cache_path=discretizer_cache,
                        force_rediscretize=cfg.force_rebuild,
                    )
                    prepared = build_result.prepared
                    variant_map = build_result.variant_map
                    analysis_cfg = build_result.config
                    domain = build_result.domain
                    petri_net_model = build_result.parse_result.petri_net_model
                    n_activities = len(petri_net_model.activities)
                    used_optimizer = cfg.use_optimizer

                    search_summary = None
                    if build_result.trial_records is not None:
                        best = max(
                            build_result.trial_records,
                            key=lambda r: r.score if r.score is not None else float("-inf"),
                        )
                        search_summary = {
                            "n_trials": cfg.search_n_trials,
                            "best_score": best.score,
                            "best_trial_number": best.trial_number,
                        }
                        (log_out_dir / "search_config.json").write_text(
                            json.dumps(_config_to_dict(analysis_cfg), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )

                    serialized = api.serialize(prepared, build_result.parse_result)

                domain_text = str(domain)
            except Exception as exc:
                logger.error(
                    "[%s] build failed (algorithm=%s, train_path=%s, use_cache=%s): %s",
                    log_id, cfg.algorithm, tts.train_path, use_cache, exc, exc_info=True,
                )
                return LogResult(
                    log_id=log_id, log_name=log_name, log_fmt=log_fmt,
                    n_train_cases=tts.n_train, n_test_cases=tts.n_test, n_activities=0,
                    pipeline_ok=False, pipeline_error=str(exc),
                )

            logger.info(
                "[%s] Domain built (%d activities, durative=True).", log_id, n_activities
            )

            # 5. Publish into the web UI's data directory — the single place
            # both the GUI import flow and the evaluation harness write the
            # domain to (web.app reads DATA_DIR/<config_name>/), so no
            # separate copy is kept under output_dir/<log_id>/.
            orig_written, curr_written = save_original_and_current(
                str(web_config_dir), serialized
            )
            logger.info(
                "[%s] Web UI data published at %s (original=%s, current=%s).",
                log_id, web_config_dir, orig_written, curr_written,
            )

            pddl_path = web_config_dir / "pddl" / "domain.pddl"
            if pddl_path.exists():
                logger.info("[%s] %s already exists — skipped", log_id, pddl_path)
            else:
                domain.write(pddl_path)
                logger.info("[%s] domain.pddl written to %s", log_id, pddl_path)

            # 6-7. Sample prefix and run Q1/Q2/Q3 for each test trace
            replayer = TraceReplayer(prepared, petri_net_model, config=analysis_cfg)
            query_results: List[QueryResult] = []
            for trace in tts.test_cases:
                trace_id = str(trace.attributes.get("concept:name", "unknown")).replace(" ", "_")
                prefix = _sample_prefix(
                    trace, replayer,
                    min_prefix_pct=cfg.min_prefix_pct,
                    max_prefix_pct=cfg.max_prefix_pct,
                    seed=cfg.seed,
                )
                if prefix is None:
                    logger.warning("[%s] %s: prefix sampling failed — skipping trace.", log_id, trace_id)
                    continue

                pddl_dir = log_out_dir / "pddl"

                # Q1 — process completion, no deadline
                q1_spec = build_q1(prefix, cfg.cost_weight)
                query_results.append(
                    _run_query(log_id, trace_id, q1_spec, domain_text, api, cfg, serialized, failures_dir, prefix, prepared, variant_map, pddl_dir, is_replayable=prefix.is_replayable)
                )

                # Q2 — completion within remaining time budget
                q2_spec = build_q2(prefix, cfg.cost_weight)
                if q2_spec is not None:
                    query_results.append(
                        _run_query(log_id, trace_id, q2_spec, domain_text, api, cfg, serialized, failures_dir, prefix, prepared, variant_map, pddl_dir, is_replayable=prefix.is_replayable)
                    )

                # Q3 — completion within budget + attribute constraints
                q3_spec = build_q3(prefix, serialized, cfg.cost_weight)
                if q3_spec is not None:
                    query_results.append(
                        _run_query(log_id, trace_id, q3_spec, domain_text, api, cfg, serialized, failures_dir, prefix, prepared, variant_map, pddl_dir, is_replayable=prefix.is_replayable)
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
                        is_replayable=prefix.is_replayable,
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
                used_optimizer=used_optimizer,
                search_summary=search_summary,
            )
    finally:
        logging.getLogger().removeHandler(_log_handler)
        _log_handler.close()


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
    prepared: Any,
    variant_map: Dict[str, Any],
    pddl_dir: Optional[Path] = None,
    is_replayable: bool = True,
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
        temporal=True
    )

    retry_cfg = RetryConfig(
        max_attempts=cfg.max_retries,
        base_delay_s=cfg.retry_delay,
        timeout_s=cfg.planner_timeout,
        memory_mb=cfg.planner_memory_mb,
    )
    result, attempts = _run_with_retry(domain_text, problem_text, api, retry_cfg, failures_dir, query_id, pddl_dir)

    if spec.query_type == "Q1":
        metrics = q1_metrics(result, prefix, cfg.cost_weight)
    elif spec.query_type == "Q2":
        metrics = q2_metrics(result, prefix, cfg.cost_weight)
    else:
        ground_truth_reachable = is_q3_reachable(spec.goal_sop, serialized)
        metrics = q3_metrics(result, prefix, spec.goal_sop, ground_truth_reachable, cfg.cost_weight)

    validation = None
    if result.success and result.plan_steps:
        init_attrs = {e["attribute"]: e["value"] for e in (spec.init_effects or [])}
        outcome = replay_plan(
            result.plan_steps,
            set(spec.init_places),
            init_attrs,
            prepared,
            variant_map,
        )
        validation = {
            "valid": outcome.is_replayable,
            "error_step": outcome.error_step,
            "error_reason": outcome.error_reason,
            "steps_executed": len(outcome.steps),
        }

        suffix_activities = [
            sanitize_name(str(e.get("concept:name")))
            for e in prefix.suffix_events
        ]
        metrics["alignment_score"] = sequence_alignment_score(result.plan_steps, suffix_activities)
    else:
        metrics["alignment_score"] = None

    return QueryResult(
        query_id=query_id,
        query_type=spec.query_type,
        trace_id=trace_id,
        prefix_ratio=prefix.prefix_ratio,
        n_prefix_events=len(prefix.prefix_events),
        attempts=attempts,
        solvability=result.solvability,
        planner_duration_s=result.search_time_s,
        metrics=metrics,
        validation=validation,
        is_replayable=is_replayable,
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
            is_replayable=q.get("is_replayable", True),
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
        used_optimizer=data.get("used_optimizer", False),
        search_summary=data.get("search_summary"),
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
    _here = Path(__file__).parent
    p = argparse.ArgumentParser(
        prog="python -m src.evaluation.run_evaluation",
        description="Evaluate the XES→PDDL pipeline against a set of event logs.",
    )
    p.add_argument("--metadata", type=Path, default=_here / "data" / "Metadata.csv",
                   help="Path to Metadata.csv (downloaded from Zenodo if absent).")
    p.add_argument("--cache-dir", dest="cache_dir", type=Path, default=_here / "data" / "cache",
                   help="Directory for downloaded log files.")
    p.add_argument("--output-dir", dest="output_dir", type=Path, default=_here / "results",
                   help="Directory for results.")
    p.add_argument("--log-id", dest="log_id", type=int, required=True,
                    help="Event Log ID to evaluate — one log per run, e.g. --log-id 55")
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
    p.add_argument("--force-rebuild", dest="force_rebuild", action="store_true",
                   help="Ignora la cache esistente per il log (current.json in "
                        "data/<log_id>/ e il discretizer cache) e ricostruisce "
                        "dominio/rete da zero, rieseguendo l'optimizer se attivo.")
    # Discretizer params
    p.add_argument("--kmeans-max-k", dest="kmeans_max_k", type=int, default=5,
                   help="Stage 1 threshold (few unique values) and max-k cap for Stage 3 Jenks.")
    p.add_argument("--dominance-threshold", dest="dominance_threshold", type=float, default=0.30,
                   help="Stage 2: minimum mass fraction for a KDE region to be dominant.")
    p.add_argument("--min-residual-points", dest="min_residual_points", type=int, default=50,
                   help="Stage 3: minimum residual points required to attempt Jenks (else single bin).")
    p.add_argument("--min-gvf-threshold", dest="min_gvf_threshold", type=float, default=0.70,
                   help="Stage 3: GVF at k=2 below this forces single bin (no structure).")
    p.add_argument("--gvf-target", dest="gvf_target", type=float, default=0.90,
                   help="Stage 3: GVF early-stop target — accept k once GVF exceeds this.")
    p.add_argument("--min-gvf-improvement", dest="min_gvf_improvement", type=float, default=0.01,
                   help="Stage 3: stop incrementing k when marginal GVF gain falls below this.")
    p.add_argument("--jenks-sample-size", dest="jenks_sample_size", type=int, default=20000,
                   help="Stage 3: max points for Jenks DP; larger arrays are sampled.")
    # network_search optimizer
    p.add_argument("--no-optimizer", dest="no_optimizer", action="store_true",
                   help="Disable the Optuna optimizer and use the fixed discretizer "
                        "params above (plus AnalysisConfig defaults) directly. "
                        "The optimizer is used by default.")
    p.add_argument("--search-n-trials", dest="search_n_trials", type=int, default=30,
                   help="Number of candidate AnalysisConfigs the optimizer evaluates per log.")
    p.add_argument("--w-det-xor", dest="w_det_xor", type=float, default=1.0,
                   help="ScoreWeights.w_det_xor (must be in [0,1]) for the optimizer.")
    p.add_argument("--w-det-eff", dest="w_det_eff", type=float, default=1.0,
                   help="ScoreWeights.w_det_eff (must be in [0,1]) for the optimizer.")
    p.add_argument("--w-fb-xor", dest="w_fb_xor", type=float, default=0.5,
                   help="ScoreWeights.w_fb_xor (must be in [0,1]) for the optimizer.")
    p.add_argument("--w-fb-eff", dest="w_fb_eff", type=float, default=0.5,
                   help="ScoreWeights.w_fb_eff (must be in [0,1]) for the optimizer.")
    p.add_argument("--w-prune-xor", dest="w_prune_xor", type=float, default=1.0,
                   help="ScoreWeights.w_prune_xor (must be in [0,1]) for the optimizer.")
    p.add_argument("--w-xor", dest="w_xor", type=float, default=1.0,
                   help="ScoreWeights.w_xor — weight of the XOR-axis score in the final score.")
    p.add_argument("--w-eff", dest="w_eff", type=float, default=1.0,
                   help="ScoreWeights.w_eff — weight of the effect-axis score in the final score.")
    p.add_argument("--w-dup", dest="w_dup", type=float, default=0.2,
                   help="ScoreWeights.w_dup for the optimizer.")
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
        log_id=args.log_id,
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
        force_rebuild=args.force_rebuild,
        kmeans_max_k=args.kmeans_max_k,
        dominance_threshold=args.dominance_threshold,
        min_residual_points=args.min_residual_points,
        min_gvf_threshold=args.min_gvf_threshold,
        gvf_target=args.gvf_target,
        min_gvf_improvement=args.min_gvf_improvement,
        jenks_sample_size=args.jenks_sample_size,
        use_optimizer=not args.no_optimizer,
        search_n_trials=args.search_n_trials,
        w_det_xor=args.w_det_xor,
        w_det_eff=args.w_det_eff,
        w_fb_xor=args.w_fb_xor,
        w_fb_eff=args.w_fb_eff,
        w_prune_xor=args.w_prune_xor,
        w_xor=args.w_xor,
        w_eff=args.w_eff,
        w_dup=args.w_dup,
    )

    selection = get_log_selection(Path(args.metadata), log_ids=[str(cfg.log_id)])
    if len(selection) != 1:
        raise ValueError(
            f"Expected exactly one log matching --log-id {cfg.log_id} in "
            f"{args.metadata}, found {len(selection)}."
        )
    row = selection.iloc[0]
    api = EvalAPI()

    log_id = str(row.get("Event Log ID", row.get("log_id", "unknown")))
    log_name = str(row.get("Event Log Name", row.get("log_name", log_id)))

    logger.info("Starting — log %s, planner=optic", log_id)

    result_path = output_dir / log_id / "result.json"
    if cfg.resume and result_path.exists():
        logger.info("[%s] Resuming — result.json already exists, skipping.", log_id)
        log_result = _load_log_result(json.loads(result_path.read_text(encoding="utf-8")))
    else:
        log_path, log_fmt = download_if_needed(row, cfg.cache_dir, force=cfg.force_download)

        if log_fmt == "csv" and cfg.csv_mapping is None:
            cfg.csv_mapping = _prompt_csv_mapping(log_path)

        log_result = evaluate_log(log_id, log_name, log_path, log_fmt, output_dir, api, cfg)
        write_log_result(log_id, log_result, output_dir)
        logger.info(
            "[%s] %s",
            log_id,
            "OK" if log_result.pipeline_ok else f"FAILED: {log_result.pipeline_error}",
        )

    summary_path = write_log_summary(log_result, output_dir, cfg.cost_weight)
    logger.info(
        "Done — log %s %s, summary at %s", log_id,
        "OK" if log_result.pipeline_ok else "FAILED", summary_path,
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

#conda run -n temporalXes2Plan --cwd src python -m evaluation.run_evaluation  \
# --log-ids 55 --min-prefix-pct 0.5 --algorithm powl --planner-timeout 1800 \
# --planner-memory-mb 16000 --cost-weight 0.05 --log-level DEBUG
