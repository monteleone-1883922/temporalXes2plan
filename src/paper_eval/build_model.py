"""Phase "model": TPE search + final network on the TRAIN part of a split.

The search scores candidates on the split's temporal validation set
(Pipeline.build_network(search_split=...)); the final network is built on
the whole TRAIN set (TRAIN_fit + VAL) with the winning config. Nothing from
TEST is read here.

Outputs go to <model_dir> (one per split, keyed by split_hash):
  current.json        serialized domain (same shape as the web UI / thesis runner)
  domain.pddl         PDDL domain
  analysis_config.json the AnalysisConfig actually used
  search_trials.json  score of every TPE trial
"""

from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from core_utils import _load_log, save_original_and_current, write_traces_xes
from encoding.domain_builder import build_domain_with_variant_map
from encoding.prepared_graph_utils import build_petrinet_model_from_prepared
from encoding.prepared_input import PreparedDomainInput
from models import AnalysisConfig
from paper_eval.config import PaperEvalConfig
from paper_eval.split import SplitResult, case_id
from pipeline import Pipeline

logger = logging.getLogger(__name__)

DOMAIN_NAME = "paper_eval"


@dataclass
class LoadedModel:
    """Everything later phases need, rebuilt from current.json so a fresh
    build and a cached one are used identically."""

    serialized: Dict[str, Any]
    prepared: PreparedDomainInput
    petri_net_model: Any
    domain: Any
    variant_map: Dict[str, Any]
    config: AnalysisConfig


def config_to_dict(config: AnalysisConfig) -> Dict[str, Any]:
    # ignored_attributes is a set, which JSON cannot store: sorted list
    # (sorted so the file is deterministic); config_from_dict reverses it.
    data = dataclasses.asdict(config)
    data["ignored_attributes"] = sorted(data["ignored_attributes"])
    return data


def config_from_dict(data: Dict[str, Any]) -> AnalysisConfig:
    data = dict(data)
    data["ignored_attributes"] = set(data.get("ignored_attributes", []))
    return AnalysisConfig(**data)


def build_model(
    log_xes: Path, split: SplitResult, cfg: PaperEvalConfig, model_dir: Path, force: bool = False,
) -> Path:
    """Run the search + final build on TRAIN and persist the result in
    model_dir. Skipped when model_dir/current.json exists, unless force."""
    model_dir = Path(model_dir)
    # Cache: current.json is the last file written, so its presence means
    # a previous build completed.
    if (model_dir / "current.json").exists() and not force:
        logger.info("model already at %s — skipped", model_dir)
        return model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    if force:
        # save_original_and_current never overwrites existing files, so they
        # must be removed for the new model to be persisted.
        for name in ("original.json", "current.json"):
            (model_dir / name).unlink(missing_ok=True)

    # Keep only the TRAIN traces (TRAIN_fit + VAL); TEST is never touched.
    log = _load_log(str(log_xes), "xes", None)
    train_ids = set(split.train_ids)
    train_traces = [t for t in log if case_id(t) in train_ids]
    # Guards against a split computed on a different log than log_xes.
    if len(train_traces) != len(train_ids):
        raise ValueError(
            f"split lists {len(train_ids)} training cases, {len(train_traces)} found in {log_xes}"
        )

    # The Parser reads from file, so the TRAIN subset goes to a temp XES.
    with tempfile.TemporaryDirectory() as tmp:
        train_xes = write_traces_xes(log, train_traces, Path(tmp) / "train.xes")
        # search=True: TPE picks the AnalysisConfig, scoring each candidate
        # on search_split (= VAL, the temporally last TRAIN cases) after
        # fitting on the rest. Then the final network is built on the whole
        # TRAIN file with the winning config. use_durative/use_costs also
        # imply has_deadline=True in the generated domain.
        result = Pipeline().build_network(
            str(train_xes),
            domain_name=DOMAIN_NAME,
            discovery_algorithm=cfg.discovery_algorithm,
            coverage_percentage=cfg.coverage_percentage,
            use_durative=True,
            use_costs=True,
            config=AnalysisConfig(),
            search=True,
            search_n_trials=cfg.search_n_trials,
            search_seed=cfg.seed,
            search_split=split.val_ids,
            snapshot_dir=str(model_dir / "analysis_debug"),
            discretizer_cache_path=model_dir / "discretizer_cache.json",
            force_rediscretize=force,
        )

    # Persist in the same shape as the web UI / thesis runner: the prepared
    # input as JSON plus the Petri net start/end places, which load_model
    # needs to rebuild the net. The tmp train.xes is already gone, nothing
    # below depends on it.
    serialized = result.prepared.to_dict()
    serialized["metadata"] = {
        "start_place": result.parse_result.start_place,
        "end_place": result.parse_result.end_place,
    }
    save_original_and_current(str(model_dir), serialized)
    result.domain.write(model_dir / "domain.pddl")
    (model_dir / "analysis_config.json").write_text(
        json.dumps(config_to_dict(result.config), indent=2), encoding="utf-8",
    )
    # One entry per TPE trial, to report how the search behaved.
    trials = [
        {"trial_number": r.trial_number, "score": r.score}
        for r in (result.trial_records or [])
    ]
    (model_dir / "search_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
    return model_dir


def load_model(model_dir: Path) -> LoadedModel:
    """Rebuild prepared input, Petri net model and PDDL domain from
    model_dir/current.json — the same reconstruction the thesis runner uses
    for its cache (evaluation/run_evaluation.py)."""
    model_dir = Path(model_dir)
    # 1. read what build_model persisted
    serialized = json.loads((model_dir / "current.json").read_text(encoding="utf-8"))
    config = config_from_dict(
        json.loads((model_dir / "analysis_config.json").read_text(encoding="utf-8"))
    )
    # 2. JSON -> prepared input -> Petri net (start/end places from metadata)
    prepared = PreparedDomainInput.from_dict(serialized)
    meta = serialized.get("metadata", {})
    petri_net_model = build_petrinet_model_from_prepared(
        prepared, meta.get("start_place"), meta.get("end_place"),
    )
    # 3. regenerate the PDDL domain with the same flags as build_model
    # (durative + costs + deadline). variant_map is only kept in memory,
    # so it can only be obtained by rebuilding the domain.
    domain, variant_map = build_domain_with_variant_map(
        prepared, config=config, domain_name=DOMAIN_NAME,
        use_durative=True, use_costs=True, has_deadline=True,
    )
    return LoadedModel(serialized, prepared, petri_net_model, domain, variant_map, config)
