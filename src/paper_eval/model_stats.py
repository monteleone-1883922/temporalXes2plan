"""Phase "stats": descriptive statistics of the model built for one log.

These are the numbers the paper reports next to the timings, per log: size of
the discovered Petri net, attributes, effect groups, size of the PDDL domain
and statistics of the activity durations. They are computed from the files of
a model already built (model_dir), so no model has to be rebuilt.

Output: model_stats.json in the model folder.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from encoding.prepared_input import PreparedDomainInput
from paper_eval.build_model import load_model

STATS_FILE = "model_stats.json"


def _spread(values: Iterable[float]) -> Dict[str, Optional[float]]:
    """min, median, mean and max (None when there are no values)."""
    xs = list(values)
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(xs), "min": min(xs), "median": statistics.median(xs),
        "mean": statistics.fmean(xs), "max": max(xs),
    }


def attributes_read(prepared: PreparedDomainInput) -> Set[str]:
    """Attributes some guard reads: the guards of XOR branches, of effect
    groups and the preconditions. The other attributes of the catalog are
    only ever written."""
    read: Set[str] = set()
    for branches in prepared.xor_branches.values():
        for branch in branches:
            for clause in (branch.conditions or []):
                read |= {g.attribute for g in clause}
    for transition in prepared.transitions.values():
        for clause in (transition.preconditions or []):
            read |= {g.attribute for g in clause}
        for group in transition.effect_groups:
            for clause in group.guard:
                read |= {g.attribute for g in clause}
    return read


def _read_json(path: Path) -> Optional[Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def model_statistics(model_dir: Path, log_stats_path: Optional[Path] = None) -> Dict[str, Any]:
    """The statistics of the model in model_dir (see the module docstring).

    log_stats_path: the <log>.stats.json written by the prepare phase; its
    content goes under "log" (traces, events, variants, case durations...).
    """
    model_dir = Path(model_dir)
    model = load_model(model_dir)
    prepared = model.prepared

    # --- Petri net: counted on the graph stored with the model ------------
    kinds = Counter(node.type for node in prepared.nodes)
    # A "xor_split" is a place with several outgoing transitions; an
    # "and_split" a visible transition with several inputs or outputs.
    places = kinds["place"] + kinds["xor_split"]
    silent = kinds["silent"]
    visible = kinds["transition"] + kinds["and_split"]
    net = {
        "places": places,
        "transitions": visible + silent,
        "transitions_visible": visible,
        "transitions_silent": silent,
        "arcs": len(prepared.edges),
        "xor_splits": kinds["xor_split"],
        "and_splits": kinds["and_split"],
    }

    # --- attributes ---------------------------------------------------------
    catalog = prepared.attribute_catalog
    read = attributes_read(prepared)
    attributes = {
        "in_catalog": len(catalog),
        "by_type": dict(Counter(entry.attribute_type for entry in catalog.values())),
        "read_by_guards": len(read & set(catalog)),
    }

    # --- effect groups (= action variants before the guards are expanded) ---
    per_transition = {name: len(t.effect_groups) for name, t in prepared.transitions.items()}
    worst = max(per_transition, key=per_transition.get) if per_transition else None
    effect_groups = {
        "total": sum(per_transition.values()),
        "max_per_transition": per_transition[worst] if worst else 0,
        "transition_with_the_most": worst,
    }

    # --- PDDL domain ----------------------------------------------------------
    domain_file = model_dir / "domain.pddl"
    pddl = {
        "actions": len(model.variant_map),
        "domain_bytes": domain_file.stat().st_size if domain_file.is_file() else None,
        "duration_scale_factor": model.domain.duration_scale_factor,
    }

    # --- durations of the activities (seconds, before any rescaling) ---------
    durations = [t.duration for t in prepared.transitions.values() if t.duration is not None]
    duration_stats = {
        "transitions_with_duration": len(durations),
        "source": dict(Counter(d.source for d in durations)),
        "effective_min_s": _spread(d.effective_min for d in durations),
        "effective_max_s": _spread(d.effective_max for d in durations),
        "mean_s": _spread(d.mean for d in durations if d.mean is not None),
    }

    # --- how the model was built, from the files of the phases ---------------
    trials = _read_json(model_dir / "search_trials.json") or []
    accepted = [t for t in trials if not t.get("rejected_reason")]
    search = {
        "trials": len(trials),
        "rejected_at_rung_0": len(trials) - len(accepted),
        "best_score": max((t["score"] for t in accepted), default=None),
    }
    split = _read_json(model_dir.parent / "split.json")

    return {
        "model_dir": str(model_dir),
        "petri_net": net,
        "attributes": attributes,
        "effect_groups": effect_groups,
        "pddl": pddl,
        "durations": duration_stats,
        "search": search,
        "split": {
            "split_hash": split.get("split_hash"),
            "train_fit": len(split.get("train_fit_ids", [])),
            "val": len(split.get("val_ids", [])),
            "test": len(split.get("test_ids", [])),
        } if split else None,
        "attribute_filter": _read_json(model_dir / "attribute_filter.json"),
        "log": _read_json(log_stats_path) if log_stats_path else None,
    }


def run_stats(model_dir: Path, log_stats_path: Optional[Path] = None) -> Dict[str, Any]:
    """Compute the statistics and write model_dir/model_stats.json."""
    stats = model_statistics(model_dir, log_stats_path)
    (Path(model_dir) / STATS_FILE).write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return stats
