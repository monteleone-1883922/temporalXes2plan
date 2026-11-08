"""Phase "model": TPE search + final network on the TRAIN part of a split.

The search scores candidates on the split's temporal validation set
(Pipeline.build_network(search_split=...)); the final network is built on
the whole TRAIN set (TRAIN_fit + VAL) with the winning config. Nothing from
TEST is read here.

Outputs go to <model_dir> (one per split, keyed by split_hash):
  current.json        serialized domain (same shape as the web UI / thesis runner)
  domain.pddl         PDDL domain
  analysis_config.json the AnalysisConfig actually used
  model_meta.json     how the model was made: split hash, discovery algorithm, seed,
                      number of TPE trials, coverage, unit_costs (what analysis_config.json lacks)
  search_trials.json  score of every TPE trial
  attribute_filter.json  what the two attribute filters did (lists and sizes)
  attributes.jsonl    one row per attribute of the log and its fate (kept, removed as
                      unused, as noise or by the cascade, ignored by the user), see model_tables
  transition_durations.jsonl  one row per transition with its duration statistics
  petri_net.png/.svg  picture of the Petri net (structure only), drawn from current.json
  split.json          copy of the split the model was built on, so the folder is
                      self-contained (the log's own split.json is overwritten by a new split)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core_utils import _load_log, save_original_and_current, write_traces_xes
from encoding.domain_builder import DomainBuilder, build_domain_with_variant_map
from encoding.prepared_graph_utils import build_petrinet_model_from_prepared
from encoding.prepared_input import PreparedDomainInput
from models import AnalysisConfig
from paper_eval.config import LogSpec, PaperEvalConfig
from paper_eval.model_tables import attribute_rows, transition_duration_rows
from paper_eval.prefixes import write_jsonl
from paper_eval.split import SplitResult, case_id, write_split
from pipeline import Pipeline

logger = logging.getLogger(__name__)

DOMAIN_NAME = "paper_eval"
PETRI_NET_PNG = "petri_net.png"
PETRI_NET_SVG = "petri_net.svg"


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
    data["keep_attributes"] = tuple(data.get("keep_attributes", ()))  # JSON gives a list
    return AnalysisConfig(**data)


def build_model(
    log_xes: Path, split: SplitResult, cfg: PaperEvalConfig, model_dir: Path, force: bool = False,
    log_spec: Optional[LogSpec] = None,
) -> Path:
    """Run the search + final build on TRAIN and persist the result in
    model_dir. Skipped when model_dir/current.json exists, unless force.

    log_spec supplies this log's attribute lists (ignore and keep); None means
    no lists, only the defaults and the automatic filters."""
    model_dir = Path(model_dir)
    # Cache: current.json is the last file written, so its presence means
    # a previous build completed.
    if (model_dir / "current.json").exists() and not force:
        built_with = read_unit_costs(model_dir)
        if built_with != cfg.unit_costs:
            raise ValueError(
                f"the model at {model_dir} was built with unit_costs={built_with}, but the current "
                f"value is {cfg.unit_costs}: its domain.pddl has other costs. Use --force to rebuild it."
            )
        logger.info("model already at %s — skipped", model_dir)
        if not (model_dir / PETRI_NET_PNG).exists():
            render_petri_net(model_dir)
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

    # The log's own ignore list ADDS to AnalysisConfig's default ignored set
    # (org:resource, ...); keep protects attributes from the two automatic
    # filters only. AnalysisConfig raises if an attribute is in both lists.
    ignore_list = tuple(log_spec.ignore_attributes) if log_spec else ()
    keep_list = tuple(log_spec.keep_attributes) if log_spec else ()
    analysis_config = AnalysisConfig(
        max_effect_groups_per_transition=cfg.max_effect_groups,
        initial_event_as_input=cfg.initial_event_as_input,
        ignored_attributes=AnalysisConfig().ignored_attributes | set(ignore_list),
        keep_attributes=keep_list,
        noise_filter=cfg.noise_filter,
        unused_attribute_filter=cfg.unused_attribute_filter,
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
            # Every action costs 1 when asked (RQ1); otherwise -log(probability) costs.
            unit_costs=cfg.unit_costs,
            # The cap on effect groups per transition keeps TPE configs whose
            # group product would not fit in memory out of the search.
            config=analysis_config,
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
    # What analysis_config.json does not say: which split, which discovery algorithm and
    # which seed made this model. Documentation of the run (it names no folder, keys no table).
    (model_dir / "model_meta.json").write_text(json.dumps({
        "split_hash": split.split_hash,
        "discovery_algorithm": cfg.discovery_algorithm,
        "seed": cfg.seed,
        "search_n_trials": cfg.search_n_trials,
        "coverage_percentage": cfg.coverage_percentage,
        "unit_costs": cfg.unit_costs,
    }, indent=2), encoding="utf-8")
    # What the attribute filters did, for the paper's per-log table and for
    # reproducibility. noise_filter comes from the final build; the unused
    # attributes are recomputed from its ParseResult by the same function the
    # encoding used.
    unused = DomainBuilder().unused_attributes(result.parse_result, result.config)
    attribute_report = {
        "ignored_by_user": sorted(ignore_list),
        "kept_by_user": sorted(keep_list),
        "noise_filter": result.noise_attributes,
        "unused_attribute_filter": sorted(unused),
        "attribute_catalog_size_before": len(result.parse_result.attribute_catalog),
        "attribute_catalog_size_after": len(result.prepared.attribute_catalog),
    }
    (model_dir / "attribute_filter.json").write_text(json.dumps(attribute_report, indent=2), encoding="utf-8")
    # The same information as plain rows, one per attribute, with everything known
    # now: the type, values and discretization of the attributes removed as unused
    # (still in parse_result's catalog) and the type of those removed as noise
    # (carried by BuildResult.noise_details). Written as the model is built because
    # current.json no longer has them.
    rows, _ = attribute_rows(
        parse_catalog=result.parse_result.attribute_catalog,
        final_catalog=result.prepared.attribute_catalog,
        unused=unused,
        noise=result.noise_attributes,
        noise_details=result.noise_details,
        ignored_by_user=ignore_list,
        kept_by_user=keep_list,
        analysed_types=result.attribute_categories,
    )
    write_jsonl(rows, model_dir / "attributes.jsonl")
    write_jsonl(transition_duration_rows(result.prepared.transitions), model_dir / "transition_durations.jsonl")
    write_split(split, model_dir / "split.json")
    # One entry per TPE trial, to report how the search behaved.
    trials = [
        {
            "trial_number": r.trial_number,
            "score": r.score,
            "rejected_reason": r.rejected_reason,
            "max_effect_groups": r.max_effect_groups,
        }
        for r in (result.trial_records or [])
    ]
    (model_dir / "search_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
    render_petri_net(model_dir)
    return model_dir


def render_petri_net(model_dir: Path) -> List[Path]:
    """Draw the net stored in model_dir/current.json as petri_net.png and petri_net.svg
    (the SVG can be zoomed, which large nets need). It is the net that later phases
    replay on, rebuilt like load_model does. The picture is documentation: a failure
    (for instance Graphviz missing) is logged and returns an empty list, the model stays valid."""
    model_dir = Path(model_dir)
    try:
        import pm4py

        _, _, net_model = _read_prepared(model_dir)
        written = []
        for name in (PETRI_NET_PNG, PETRI_NET_SVG):
            path = model_dir / name
            pm4py.save_vis_petri_net(
                net_model.petrinet, net_model.initial_marking, net_model.final_marking,
                str(path), rankdir="LR",
            )
            written.append(path)
        return written
    except Exception as exc:  # noqa: BLE001 - never fail a build over a picture
        logger.warning("could not draw the Petri net of %s: %s", model_dir, exc)
        return []


def read_unit_costs(model_dir: Path) -> bool:
    """Whether the model in model_dir was built with unit costs (False when
    model_meta.json is missing or does not say: the -log(probability) costs)."""
    path = Path(model_dir) / "model_meta.json"
    if not path.exists():
        return False
    return bool(json.loads(path.read_text(encoding="utf-8")).get("unit_costs", False))


def _read_prepared(model_dir: Path):
    """current.json -> (serialized dict, PreparedDomainInput, PetriNetModel); the start and
    end places come from the metadata build_model stored."""
    serialized = json.loads((Path(model_dir) / "current.json").read_text(encoding="utf-8"))
    prepared = PreparedDomainInput.from_dict(serialized)
    meta = serialized.get("metadata", {})
    petri_net_model = build_petrinet_model_from_prepared(
        prepared, meta.get("start_place"), meta.get("end_place"),
    )
    return serialized, prepared, petri_net_model


def load_model(model_dir: Path, use_costs: bool = True) -> LoadedModel:
    """Rebuild prepared input, Petri net model and PDDL domain from
    model_dir/current.json — the same reconstruction the thesis runner uses
    for its cache (evaluation/run_evaluation.py).

    use_costs=False rebuilds the domain without action costs (no total-cost): what RQ2
    uses, since it minimises total-time only. The model itself does not depend on it."""
    model_dir = Path(model_dir)
    # 1-2. read what build_model persisted: JSON -> prepared input -> Petri net
    serialized, prepared, petri_net_model = _read_prepared(model_dir)
    config = config_from_dict(
        json.loads((model_dir / "analysis_config.json").read_text(encoding="utf-8"))
    )
    # 3. regenerate the PDDL domain with the same flags as build_model
    # (durative + costs + deadline, unit costs as recorded in model_meta.json).
    # variant_map is only kept in memory, so it can only be obtained by
    # rebuilding the domain.
    domain, variant_map = build_domain_with_variant_map(
        prepared, config=config, domain_name=DOMAIN_NAME,
        use_durative=True, use_costs=use_costs, has_deadline=True,
        unit_costs=use_costs and read_unit_costs(model_dir),
    )
    return LoadedModel(serialized, prepared, petri_net_model, domain, variant_map, config)
