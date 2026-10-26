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
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import List

from core_utils import _load_log
from paper_eval import prefixes as px
from paper_eval.build_model import build_model, load_model
from paper_eval.config import LOGS, PaperEvalConfig, bucket_label
from paper_eval.prepare import prepare_log
from paper_eval.split import (
    case_id, chronological_case_ids, control_flow_view, make_split, read_split, write_split,
)
from replay.trace_replayer import TraceReplayer

logger = logging.getLogger("paper_eval")

PHASES = ("prepare", "split", "model", "prefixes", "volume", "all")


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
    if (model_dir / "instances.jsonl").exists() and not force:
        logger.info("[%s] prefixes already in %s — skipped", log_name, model_dir)
        return
    # Replay runs on the model built from TRAIN, over TEST traces only.
    model = load_model(model_dir)
    replayer = TraceReplayer(model.prepared, model.petri_net_model, config=model.config)
    log = _load_log(str(paths.log_xes), "xes", None)
    records, instances = px.generate(
        _test_traces(log, split.test_ids),
        replayer,
        split.deadline_thresholds_s,
        cfg.buckets_for(log_name),
        cfg.k_min,
        model.domain.duration_scale_factor,
    )
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
        "n_goals_per_prefix": 1 + len(cfg.deadline_quantiles),
    }
    paths.volume_json.write_text(json.dumps(volume, indent=2), encoding="utf-8")
    logger.info("[%s] volume: %s", log_name, volume)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, choices=sorted(LOGS))
    parser.add_argument("--phase", default="all", choices=PHASES)
    parser.add_argument("--force", action="store_true", help="recompute the selected phase(s)")
    parser.add_argument("--search-n-trials", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = PaperEvalConfig()
    if args.search_n_trials is not None:
        cfg.search_n_trials = args.search_n_trials
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
            build_model(paths.log_xes, split, cfg, paths.model_dir(split.split_hash), force=args.force)
        elif phase == "prefixes":
            run_prefixes(args.log, cfg, paths, args.force)
        elif phase == "volume":
            run_volume(args.log, cfg, paths)


if __name__ == "__main__":
    main()
