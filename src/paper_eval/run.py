"""CLI of the paper evaluation data layer.

    PYTHONPATH=src python -m paper_eval.run --log sepsis --phase all
    PYTHONPATH=src python -m paper_eval.run --log road_fines --phase volume

Phases (each skipped when its output already exists, unless --force):
  prepare   data/paper_eval/<log>.xes (+ .stats.json)
  split     results/paper_eval/<log>/split.json
  model     results/paper_eval/<log>/<split_hash>/current.json, domain.pddl, …
  prefixes  results/paper_eval/<log>/<split_hash>/{prefixes,instances}.jsonl, prefix_stats.json
  volume    results/paper_eval/<log>/volume.json — prefix counts per bucket
            without building the model (a quick size estimate)
  all       prepare + split + model + prefixes

Phases that are NOT part of "all" (they run OPTIC or read its results):
  plan      OPTIC on every instance -> <split_hash>/plan_results.jsonl, plan_meta.json
            (resumable; one OPTIC at a time; --goal, --limit, --timeout, --memory-mb;
            --force discards the previous results)
  stats     descriptive statistics of the model -> <split_hash>/model_stats.json
  report    RQ1 table from plan_results.jsonl -> <split_hash>/rq1_summary.{json,md}
  query     --sql "SELECT ..." on the per-case views (see paper_eval/query.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List

from core_utils import _load_log
from paper_eval import prefixes as px
from paper_eval.build_model import build_model, load_model
from paper_eval.config import LOGS, PaperEvalConfig, bucket_label
from paper_eval.model_stats import run_stats
from paper_eval.plan import GOAL_CHOICES, PREFIXES_META, prefixes_deadline_variant_only, run_plan
from paper_eval.prepare import prepare_log
from paper_eval.report import run_report
from paper_eval.split import (
    case_id, chronological_case_ids, control_flow_view, make_split, read_split, variant_mean_durations,
    write_split,
)
from replay.trace_replayer import TraceReplayer

logger = logging.getLogger("paper_eval")

PHASES = ("prepare", "split", "model", "prefixes", "volume", "plan", "stats", "report", "query", "all")


class Paths:
    """Where each phase reads/writes for one log. The model and prefixes live
    under a folder named by split_hash, so a new split gets a new folder."""

    def __init__(self, log_name: str, cfg: PaperEvalConfig):
        self.log_xes = Path(cfg.data_dir) / f"{log_name}.xes"
        self.log_dir = Path(cfg.results_dir) / log_name
        self.split_json = self.log_dir / "split.json"
        self.volume_json = self.log_dir / "volume.json"

    def model_dir(self, split_hash: str) -> Path:
        return self.log_dir / split_hash


def run_split(log_name: str, cfg: PaperEvalConfig, paths: Paths, force: bool) -> None:
    if paths.split_json.exists() and not force:
        logger.info("[%s] split already at %s — skipped", log_name, paths.split_json)
        return
    log = _load_log(str(paths.log_xes), "xes", None)
    split = make_split(log, paths.log_xes, cfg.test_pct, cfg.val_pct, cfg.deadline_quantiles)
    write_split(split, paths.split_json)
    logger.info(
        "[%s] split %s: train_fit=%d val=%d test=%d, T_q (days)=%s",
        log_name, split.split_hash, len(split.train_fit_ids), len(split.val_ids),
        len(split.test_ids), {k: round(v / px.DAY_S, 2) for k, v in split.deadline_thresholds_s.items()},
    )


def _test_traces(log, test_ids: List[str]):
    """TEST traces in chronological order, so runs are deterministic."""
    wanted = set(test_ids)
    traces = [t for t in log if case_id(t) in wanted]
    order = {cid: i for i, cid in enumerate(chronological_case_ids(traces))}
    return sorted(traces, key=lambda t: order[case_id(t)])


def run_prefixes(log_name: str, cfg: PaperEvalConfig, paths: Paths, force: bool) -> None:
    split = read_split(paths.split_json)
    model_dir = paths.model_dir(split.split_hash)
    meta_path = model_dir / PREFIXES_META
    if (model_dir / "instances.jsonl").exists() and not force:
        # Instances made with another deadline rule must not be silently reused:
        # the plan results (keyed by group_id) would not match them.
        existing = prefixes_deadline_variant_only(model_dir)
        if existing != cfg.deadline_variant_only:
            raise ValueError(
                f"{model_dir / 'instances.jsonl'} was made with deadline_variant_only={existing}, "
                f"but the current value is {cfg.deadline_variant_only}. Use --force to regenerate."
            )
        logger.info("[%s] prefixes already in %s — skipped", log_name, model_dir)
        return
    # Replay runs on the model built from TRAIN, over TEST traces only.
    model = load_model(model_dir)
    replayer = TraceReplayer(model.prepared, model.petri_net_model, config=model.config)
    log = _load_log(str(paths.log_xes), "xes", None)
    # Mean duration of each variant over the TRAIN cases only: the first term
    # of the deadline goal. The test traces never contribute to it.
    train_ids = set(split.train_ids)
    variant_means = variant_mean_durations(t for t in log if case_id(t) in train_ids)
    records, instances = px.generate(
        _test_traces(log, split.test_ids),
        replayer,
        variant_means,
        cfg.buckets_for(log_name),
        cfg.k_min,
        model.domain.duration_scale_factor,
        variant_only=cfg.deadline_variant_only,
    )
    meta_path.write_text(json.dumps({"deadline_variant_only": cfg.deadline_variant_only}), encoding="utf-8")
    # prefixes.jsonl = one row per (prefix, goal); instances.jsonl = the same
    # after deduplication, each with the ground truth of the cases it groups.
    px.write_jsonl(records, model_dir / "prefixes.jsonl")
    px.write_jsonl(instances, model_dir / "instances.jsonl")
    stats = px.prefix_stats(records, instances)
    (model_dir / "prefix_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("[%s] prefixes: %s", log_name, stats["overall"])


def run_volume(log_name: str, cfg: PaperEvalConfig, paths: Paths) -> None:
    """Prefix counts of the TEST part, per bucket, and the number of
    distinct activity prefixes — an upper bound on the planner instances
    per goal, available before any model is built."""
    split = read_split(paths.split_json)
    log = _load_log(str(paths.log_xes), "xes", None)
    buckets = cfg.buckets_for(log_name)
    # No replay here: counts only need activity sequences, which is why
    # this phase is fast and needs no model.
    per_bucket: Counter = Counter()
    distinct = set()
    for trace in _test_traces(log, split.test_ids):
        acts = [e["concept:name"] for e in control_flow_view(trace)]
        for k in range(cfg.k_min, len(acts)):
            per_bucket[bucket_label(k, buckets)] += 1
            distinct.add(tuple(acts[:k]))
    volume = {
        "n_test_traces": len(split.test_ids),
        "n_prefixes": sum(per_bucket.values()),
        "n_distinct_activity_prefixes": len(distinct),
        "n_prefixes_by_bucket": {str(b): n for b, n in per_bucket.items()},
        "n_goals_per_prefix": 2,  # completion and deadline
    }
    paths.volume_json.write_text(json.dumps(volume, indent=2), encoding="utf-8")
    logger.info("[%s] volume: %s", log_name, volume)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, choices=sorted(LOGS))
    parser.add_argument("--phase", default="all", choices=PHASES)
    parser.add_argument("--force", action="store_true", help="recompute the selected phase(s)")
    parser.add_argument("--search-n-trials", type=int, default=None)
    # Switches for the two automatic attribute filters (defaults come from PaperEvalConfig).
    parser.add_argument("--noise-filter", action=argparse.BooleanOptionalAction, default=None,
                        help="drop noise attributes (constant, long text, counter-like)")
    parser.add_argument("--unused-filter", action=argparse.BooleanOptionalAction, default=None,
                        help="drop attributes no guard reads, merging the groups that become equal")
    parser.add_argument("--deadline-variant-only", action=argparse.BooleanOptionalAction, default=None,
                        help="prefixes: deadline = variant mean - elapsed always (default: min with real remaining time)")
    # Options of the "plan" phase.
    parser.add_argument("--goal", default="all", choices=GOAL_CHOICES,
                        help="plan: which instances to run (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="plan: handle at most this many instances not done yet (to try the run)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="plan: OPTIC wall-clock limit in seconds (default: per-log value of the config)")
    parser.add_argument("--memory-mb", type=int, default=None,
                        help="plan: OPTIC virtual memory limit in MB (default: per-log value of the config)")
    parser.add_argument("--sql", default=None, help="query: SELECT to run on the views prefixes, plan_results, cases, instances_view")
    args = parser.parse_args(argv)

    cfg = PaperEvalConfig()
    if args.search_n_trials is not None:
        cfg.search_n_trials = args.search_n_trials
    if args.noise_filter is not None:
        cfg.noise_filter = args.noise_filter
    if args.unused_filter is not None:
        cfg.unused_attribute_filter = args.unused_filter
    if args.deadline_variant_only is not None:
        cfg.deadline_variant_only = args.deadline_variant_only
    paths = Paths(args.log, cfg)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(paths.log_dir / "run.log", encoding="utf-8")],
        force=True,
    )

    # "all" runs the four phases in dependency order; "volume" is separate.
    spec = LOGS[args.log]
    phases = ("prepare", "split", "model", "prefixes") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "prepare":
            prepare_log(spec, cfg.data_dir, cfg.min_trace_len, force=args.force)
        elif phase == "split":
            run_split(args.log, cfg, paths, args.force)
        elif phase == "model":
            split = read_split(paths.split_json)
            build_model(
                paths.log_xes, split, cfg, paths.model_dir(split.split_hash),
                force=args.force, log_spec=spec,
            )
        elif phase == "prefixes":
            run_prefixes(args.log, cfg, paths, args.force)
        elif phase == "volume":
            run_volume(args.log, cfg, paths)
        elif phase == "plan":
            counts = run_plan(
                args.log, cfg, paths, goal=args.goal, limit=args.limit,
                timeout_s=args.timeout, memory_mb=args.memory_mb, force=args.force,
            )
            logger.info("[%s] plan: handled %d instances: %s", args.log, sum(counts.values()), dict(counts))
        elif phase == "stats":
            split = read_split(paths.split_json)
            run_stats(paths.model_dir(split.split_hash), cfg.data_dir / f"{args.log}.stats.json")
        elif phase == "report":
            split = read_split(paths.split_json)
            run_report(paths.model_dir(split.split_hash))
        elif phase == "query":
            if not args.sql:
                parser.error("--phase query needs --sql")
            from paper_eval.query import run_query  # duckdb is needed by this phase only
            split = read_split(paths.split_json)
            print(run_query(paths.model_dir(split.split_hash), args.sql))


def _ensure_hash_seed(seed: int = 42) -> None:
    """Re-execute with PYTHONHASHSEED fixed: str hashing (and so set order)
    is only seeded at interpreter start, and it would otherwise change
    from run to run."""
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.execv(sys.executable, [sys.executable, "-m", "paper_eval.run", *sys.argv[1:]])


if __name__ == "__main__":
    _ensure_hash_seed()
    main()
