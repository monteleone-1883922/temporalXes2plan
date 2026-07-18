"""Flask blueprint exposing the REST API for the Petri net web UI."""

import hashlib
import json
import os
import tempfile
import threading
import traceback
import pandas as pd
from parsing.csv_loader import detect_mapping
from parsing.log_processor import LogProcessor
from parsing.csv_loader import REQUIRED_FIELDS
from parsing.log_validator import validate_event_log
from dataclasses import fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from models import AnalysisConfig
from pipeline import Pipeline
import core_utils as utils
from planning.planner_config import (
    load_planner_config, save_planner_config, config_to_dict,
    FastDownwardConfig, OpticConfig, _merge,
)
from encoding import build_domain_with_variant_map
from encoding.prepared_input import PreparedDomainInput
from encoding.prepared_graph_utils import build_petrinet_model_from_prepared
from dataclasses import asdict

from pm4py.objects.log.importer.xes import importer as _xes_importer
from parsing.csv_loader import csv_to_event_log
from parsing.log_validator import validate_partial_trace
from replay.trace_replayer import TraceReplayer

from flask import Blueprint, current_app, jsonify, request, send_file

from web.jobs import tracker
from encoding.problem_builder import ProblemBuilder

api = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_dir(config_name: str) -> str:
    base = os.path.abspath(current_app.config["DATA_DIR"])
    return str(os.path.join(base, os.path.basename(config_name)))


def _original_path(config_name: str) -> str:
    return os.path.join(_config_dir(config_name), "original.json")


def _current_path(config_name: str) -> str:
    return os.path.join(_config_dir(config_name), "current.json")


def _pddl_dir(config_name: str) -> str:
    return os.path.join(_config_dir(config_name), "pddl")


def _compute_current_hash(config_name: str) -> str:
    path = _current_path(config_name)
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _save_domain_state(
    pddl_dir: str, hash_str: str, use_durative: bool,
    use_costs: bool = False, has_deadline: bool = False,
) -> None:
    """Write domain state to .domain_hash. Usable from background threads (no Flask context)."""
    p = os.path.join(pddl_dir, ".domain_hash")
    os.makedirs(pddl_dir, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({
            "hash": hash_str,
            "use_durative": use_durative,
            "use_costs": use_costs,
            "has_deadline": has_deadline,
        }, fh)


def _read_domain_state(config_name: str) -> Dict[str, Any]:
    """Read domain state from .domain_hash. Only call from request handlers (uses Flask context)."""
    p = os.path.join(_pddl_dir(config_name), ".domain_hash")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _detect_sink_place(graph: Dict[str, Any]) -> str | None:
    """Return the unique sink place (no outgoing edges to transitions), or None."""
    _PLACE_TYPES = {"place", "xor_split"}
    node_by_id = {n["id"]: n for n in graph.get("nodes", [])}
    place_labels = {
        n["label"] for n in graph.get("nodes", []) if n.get("type") in _PLACE_TYPES
    }
    has_outgoing: set = set()
    for e in graph.get("edges", []):
        src = node_by_id.get(e.get("source"))
        if src and src.get("type") in _PLACE_TYPES:
            has_outgoing.add(src.get("label"))
    sinks = place_labels - has_outgoing
    return next(iter(sinks)) if len(sinks) == 1 else None


def _detect_source_place(graph: Dict[str, Any]) -> str | None:
    """Return the unique source place (no incoming edges from transitions), or None."""
    _PLACE_TYPES = {"place", "xor_split"}
    node_by_id = {n["id"]: n for n in graph.get("nodes", [])}
    place_labels = {
        n["label"] for n in graph.get("nodes", []) if n.get("type") in _PLACE_TYPES
    }
    has_incoming: set = set()
    for e in graph.get("edges", []):
        src = node_by_id.get(e.get("source"))
        tgt = node_by_id.get(e.get("target"))
        if src and src.get("type") not in _PLACE_TYPES and tgt and tgt.get("type") in _PLACE_TYPES:
            has_incoming.add(tgt.get("label"))
    sources = place_labels - has_incoming
    return next(iter(sources)) if len(sources) == 1 else None


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(data: Dict[str, Any], path: str) -> None:
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def _build_analysis_config(data: Dict[str, Any]) -> AnalysisConfig:
    """When called for a search=True run, `data` typically only contains the
    ~12 AnalysisConfig fields the optimizer doesn't tune (the GUI hides the
    rest) — the resulting AnalysisConfig is used as Pipeline.run()'s *base*
    config, not applied directly (see Pipeline.run's `config` docstring).
    """
    if "ignored_attributes" in data and isinstance(data["ignored_attributes"], list):
        data["ignored_attributes"] = set(data["ignored_attributes"])
    valid = {f.name for f in dc_fields(AnalysisConfig)}
    return AnalysisConfig(**{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Petri net CRUD
# ---------------------------------------------------------------------------

@api.route("/<config_name>/petri-net", methods=["GET"])
def get_current(config_name: str):
    path = _current_path(config_name)
    if not os.path.isfile(path):
        return jsonify({"error": "Configuration not found"}), 404
    return jsonify(_read_json(path))


@api.route("/<config_name>/petri-net/original", methods=["GET"])
def get_original(config_name: str):
    path = _original_path(config_name)
    if not os.path.isfile(path):
        return jsonify({"error": "Original configuration not found"}), 404
    return jsonify(_read_json(path))


@api.route("/<config_name>/petri-net/reset", methods=["POST"])
def reset(config_name: str):
    orig = _original_path(config_name)
    if not os.path.isfile(orig):
        return jsonify({"error": "Original configuration not found"}), 404
    _write_json(_read_json(orig), _current_path(config_name))
    return jsonify({"status": "ok"})


@api.route("/<config_name>/export", methods=["GET"])
def export_config(config_name: str):
    path = _current_path(config_name)
    if not os.path.isfile(path):
        return jsonify({"error": "Configuration not found"}), 404
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        path,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"{config_name}_{ts}.json",
    )


@api.route("/<config_name>/import", methods=["POST"])
def import_config(config_name: str):
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename or not file.filename.endswith(".json"):
        return jsonify({"error": "File must be .json"}), 400
    try:
        data = json.load(file)
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Invalid JSON: {exc}"}), 400
    required = {"graph", "transitions", "xor_splits", "attribute_catalog"}
    if not required.issubset(data.keys()):
        return jsonify({"error": f"Missing keys: {required - set(data.keys())}"}), 400
    _write_json(data, _current_path(config_name))
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# AnalysisConfig defaults
# ---------------------------------------------------------------------------
#TODO to be fixed when implementingg plan setup config improvements
@api.route("/analysis-config/defaults", methods=["GET"])
def analysis_config_defaults():
    cfg = AnalysisConfig()
    result: Dict[str, Any] = {}
    for f in dc_fields(cfg):
        val = getattr(cfg, f.name)
        result[f.name] = sorted(val) if isinstance(val, set) else val
    return jsonify(result)


# ---------------------------------------------------------------------------
# Feature 1 — Log upload (XES or CSV)
# ---------------------------------------------------------------------------

def _logs_dir() -> str:
    return os.path.join(current_app.config["PROJECT_ROOT"], "logs")


def _find_log_path(log_name: str) -> Optional[str]:
    """Return path to the log file (.xes preferred, then .csv), or None."""
    base = _logs_dir()
    for ext in (".xes", ".csv"):
        p = os.path.join(base, f"{log_name}{ext}")
        if os.path.isfile(p):
            return p
    return None


@api.route("/upload-log", methods=["POST"])
def upload_log():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file provided"}), 400

    filename_lower = file.filename.lower()
    if filename_lower.endswith(".xes"):
        file_type = "xes"
    elif filename_lower.endswith(".csv"):
        file_type = "csv"
    else:
        return jsonify({"error": "File must be a .xes or .csv event log"}), 400

    config_name = utils.sanitize_name(Path(file.filename).stem)
    logs_dir = _logs_dir()
    os.makedirs(logs_dir, exist_ok=True)

    ext = f".{file_type}"
    save_path = os.path.join(logs_dir, f"{config_name}{ext}")
    file.save(save_path)

    if file_type == "csv":

        try:
            columns = list(pd.read_csv(save_path, nrows=0).columns)
            detected_mapping = detect_mapping(columns)
        except Exception as exc:
            os.unlink(save_path)
            return jsonify({"error": f"Cannot read CSV headers: {exc}"}), 400
        return jsonify({
            "config_name": config_name,
            "log_path": save_path,
            "file_type": "csv",
            "columns": columns,
            "detected_mapping": detected_mapping,
        })

    return jsonify({"config_name": config_name, "log_path": save_path, "file_type": "xes"})


# ---------------------------------------------------------------------------
# Feature 2 — Log validation + pipeline execution
# ---------------------------------------------------------------------------

@api.route("/<log_name>/validate-log", methods=["POST"])
def validate_log(log_name: str):
    """Synchronously validate a log's standard fields and return categorised results.

    No pipeline thread is started. Used by the setup page before run-pipeline
    to surface blocking errors and ask user confirmation for warnings.
    """
    log_path = _find_log_path(log_name)
    if log_path is None:
        return jsonify({"error": f"Log file not found: {log_name}.xes or {log_name}.csv"}), 404

    if log_path.endswith(".csv"):
        mapping_path = os.path.join(_logs_dir(), f"{log_name}.mapping.json")
        if not os.path.isfile(mapping_path):
            return jsonify({"error": "CSV column mapping not found. Complete the setup form first."}), 400

    try:
        processor = LogProcessor(log_path)
        log, _, _ = processor.load_and_filter_log(coverage_percentage=1.0)
        result = validate_event_log(log)
    except Exception as exc:
        return jsonify({"error": f"Could not load log for validation: {exc}"}), 400

    return jsonify({
        "errors": result.errors,
        "warnings": result.warnings,
        "infos": result.infos,
        "missing_timestamp": result.missing_timestamp,
        "missing_lifecycle": result.missing_lifecycle,
    })


@api.route("/<log_name>/run-pipeline", methods=["POST"])
def run_pipeline(log_name: str):
    body = request.get_json(force=True) or {}
    pipeline_params = body.get("pipeline", {})
    config_dict = body.get("config", {})
    csv_mapping_body: Optional[Dict[str, Any]] = body.get("csv_mapping")

    log_path = _find_log_path(log_name)
    if log_path is None:
        return jsonify({"error": f"Log file not found: {log_name}.xes or {log_name}.csv"}), 404

    allow_missing_timestamp: bool = bool(body.get("allow_missing_timestamp", False))

    if log_path.endswith(".csv"):
        if not csv_mapping_body:
            return jsonify({"error": "CSV log requires column mapping. Provide 'csv_mapping' in the request body."}), 400

        missing = [f for f in REQUIRED_FIELDS if not csv_mapping_body.get(f)]
        if missing:
            return jsonify({"error": f"Missing required column mappings: {missing}"}), 400
        mapping_path = os.path.join(_logs_dir(), f"{log_name}.mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as fh:
            json.dump(csv_mapping_body, fh, indent=2)

    data_dir = current_app.config["DATA_DIR"]
    pddl_out = os.path.join(data_dir, log_name, "pddl")

    job_id = tracker.create()
    threading.Thread(
        target=_pipeline_thread,
        args=(job_id, log_path, data_dir, pddl_out, pipeline_params, config_dict,
              allow_missing_timestamp),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


def _pipeline_thread(
    job_id: str,
    log_path: str,
    data_dir: str,
    pddl_out: str,
    pipeline_params: Dict[str, Any],
    config_dict: Dict[str, Any],
    allow_missing_timestamp: bool = False,
) -> None:
    try:
        tracker.append_log(job_id, "Initializing configuration...")
        config = _build_analysis_config(config_dict)

        allowed_kwargs = {
            "domain_name", "discovery_algorithm",
            "coverage_percentage", "use_durative", "use_costs", "use_activity_classifier",
            "search",
        }
        kwargs = {k: v for k, v in pipeline_params.items() if k in allowed_kwargs}
        if "search" in kwargs:
            kwargs["search"] = bool(kwargs["search"])

        if kwargs.get("search"):
            tracker.append_log(job_id, "Searching best configuration (30 trials)...")
        tracker.append_log(job_id, f"Parsing {Path(log_path).name}...")
        pddl_path = Pipeline().run(
            log_path=log_path,
            data_dir=data_dir,
            pddl_output_dir=pddl_out,
            config=config,
            **kwargs,
        )

        config_name = utils.sanitize_name(Path(log_path).stem)

        # Write domain state so build_problem won't overwrite the pipeline-generated domain.
        use_durative = bool(pipeline_params.get("use_durative", False))
        use_costs = bool(pipeline_params.get("use_costs", False))
        current_json = Path(data_dir) / config_name / "current.json"
        if current_json.exists():
            h = hashlib.sha256(current_json.read_bytes()).hexdigest()
            _save_domain_state(pddl_out, h, use_durative, use_costs)

            # Persist has_timestamps in current.json metadata for informational warnings.
            if allow_missing_timestamp:
                data = json.loads(current_json.read_text(encoding="utf-8"))
                data.setdefault("metadata", {})["has_timestamps"] = False
                current_json.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )

        tracker.append_log(job_id, "Pipeline complete.")
        tracker.complete(job_id, {"config_name": config_name, "pddl_path": str(pddl_path)})

    except Exception:
        err = traceback.format_exc()
        tracker.append_log(job_id, f"Error: {err.splitlines()[-1]}")
        tracker.fail(job_id, err)


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------

@api.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    job = tracker.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ---------------------------------------------------------------------------
# Feature 3 — Problem builder + planner stub
# ---------------------------------------------------------------------------
@api.route("/<config_name>/build-problem", methods=["POST"])
def build_problem(config_name: str):
    body = request.get_json(force=True) or {}
    # Accept init_places (list, from partial trace) or init_place (str, from URL param click)
    _raw_places = body.get("init_places")
    _raw_place = body.get("init_place") or None
    if _raw_places:
        init_places: Optional[List[str]] = [p for p in _raw_places if p]
    elif _raw_place:
        init_places = [_raw_place]
    else:
        init_places = []
    init_effects = body.get("init", [])
    goal_sop = body.get("goal", [])
    metric = body.get("metric") or None   # "minimize_cost" | "minimize_time" | None
    require_completion = bool(body.get("require_completion", False))

    if (not goal_sop or not any(goal_sop)) and not require_completion:
        return jsonify({"error": "At least one goal clause or require_completion is required"}), 400

    pddl_out = _pddl_dir(config_name)
    current_path = _current_path(config_name)
    if not os.path.isfile(current_path):
        return jsonify({"error": "Configuration not found"}), 404
    data = _read_json(current_path)

    attribute_catalog = data.get("attribute_catalog", {})
    saved_state = _read_domain_state(config_name)
    end_place = (
        data.get("metadata", {}).get("end_place")
        or _detect_sink_place(data["graph"])
    )

    planner = body.get("planner", "fast_downward")
    use_durative = (planner == "optic")
    use_costs = (metric == "minimize_cost")

    # Parse and validate deadline (positive number, seconds).
    deadline_raw = body.get("deadline", None)
    try:
        deadline: Optional[float] = float(deadline_raw) if deadline_raw is not None else None
        if deadline is not None and deadline <= 0:
            deadline = None
    except (TypeError, ValueError):
        deadline = None

    has_deadline = deadline is not None

    # Rebuild domain whenever graph, use_durative, use_costs or has_deadline don't match saved state.
    current_hash = _compute_current_hash(config_name)
    need_rebuild = (
        not current_hash
        or current_hash != saved_state.get("hash")
        or use_durative != bool(saved_state.get("use_durative", False))
        or use_costs != bool(saved_state.get("use_costs", False))
        or has_deadline != bool(saved_state.get("has_deadline", False))
    )
    if need_rebuild:
        rebuilt, _variant_map = build_domain_with_variant_map(
            PreparedDomainInput.from_dict(data),
            domain_name=config_name,
            use_durative=use_durative,
            use_costs=use_costs,
            has_deadline=has_deadline,
        )
        rebuilt.write(Path(pddl_out) / "domain.pddl")
        _save_domain_state(pddl_out, current_hash, use_durative, use_costs, has_deadline)

    problem_text = ProblemBuilder().build(
        problem_name=f"{config_name}_prediction",
        domain_name=config_name,
        init_places=init_places,
        init_effects=init_effects,
        goal_sop=goal_sop,
        attribute_catalog=attribute_catalog,
        metric=metric,
        require_completion=require_completion,
        end_place=end_place,
        deadline=deadline,
    )

    os.makedirs(pddl_out, exist_ok=True)
    problem_path = os.path.join(pddl_out, "problem.pddl")
    with open(problem_path, "w", encoding="utf-8") as fh:
        fh.write(problem_text)

    warnings: List[str] = []
    if use_durative:
        transitions = data.get("transitions", {})
        no_duration = all(
            t.get("duration", {}).get("effective_max", 0) == 0
            for t in transitions.values()
        )
        if no_duration:
            warnings.append(
                "No action has a duration set. The temporal planner will receive "
                "zero-duration actions and results may be meaningless. "
                "Set action durations in the Petri net editor before running OPTIC."
            )

    return jsonify({
        "status": "built",
        "problem_path": problem_path,
        "problem_text": problem_text,
        "warnings": warnings,
    })


# ---------------------------------------------------------------------------
# Partial trace replay
# ---------------------------------------------------------------------------

class PartialTraceError(Exception):
    """Raised when the uploaded partial-trace file cannot be parsed."""


def _parse_uploaded_trace(file_bytes: bytes, file_type: str, mapping: Optional[Dict[str, Optional[str]]]):
    """Parse a single-trace XES or CSV upload into one pm4py Trace.

    Recreated from the deleted parsing/partial_trace_replayer.py — the
    parsing/validation logic itself never depended on the removed
    ActionRegistry-era modules, only on csv_loader/log_validator, still
    present and untouched.
    """
    try:
        if file_type == "csv":
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                log = csv_to_event_log(tmp.name, mapping)
        else:
            with tempfile.NamedTemporaryFile(suffix=".xes", delete=True) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                log = _xes_importer.apply(tmp.name)
    except Exception as exc:
        raise PartialTraceError(f"Invalid {file_type.upper()} file: {exc}") from exc

    if len(log) == 0:
        raise PartialTraceError(f"{file_type.upper()} file contains no traces.")
    if len(log) > 1:
        raise PartialTraceError(
            f"Expected a single-trace {file_type.upper()} file, found {len(log)} traces."
        )
    trace = log[0]
    errors = validate_partial_trace(trace)
    if errors:
        raise PartialTraceError("; ".join(errors))
    return trace


@api.route("/<config_name>/replay-partial-trace", methods=["POST"])
def replay_partial_trace(config_name: str):
    """Replay a single-trace XES or CSV file and return the derived init state.

    Uses replay.trace_replayer.TraceReplayer.replay_partial_trace (caso 3 of
    docs/trace_replayer_analysis.md) — the raw log-observed snapshot (marking
    + attributes), no guard/effect-group interpretation. Unlike the old
    parsing/partial_trace_replayer.py, this does not pre-emptively verify PDDL
    feasibility; only the warnings Phase 1 itself produces are surfaced.
    """

    current_path = _current_path(config_name)
    if not os.path.isfile(current_path):
        return jsonify({"error": "Configuration not found"}), 404

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file provided"}), 400

    filename_lower = f.filename.lower()
    if filename_lower.endswith(".xes"):
        file_type = "xes"
        mapping = None
    elif filename_lower.endswith(".csv"):
        file_type = "csv"
        mapping_path = os.path.join(_logs_dir(), f"{config_name}.mapping.json")
        if not os.path.isfile(mapping_path):
            return jsonify({
                "error": (
                    "No column mapping found for this configuration. "
                    "To use a CSV partial trace, first run the pipeline from a CSV log "
                    "and complete the column mapping on the setup page."
                )
            }), 400
        with open(mapping_path, "r", encoding="utf-8") as fh:
            mapping = json.load(fh)
    else:
        return jsonify({"error": "File must be a .xes or .csv file"}), 400

    current_data = _read_json(current_path)
    # Backfill metadata for configs generated before the metadata key was added.
    if "metadata" not in current_data:
        current_data["metadata"] = {
            "start_place": _detect_source_place(current_data.get("graph", {})) or "",
            "end_place": _detect_sink_place(current_data.get("graph", {})) or "",
        }
    start_place = current_data["metadata"].get("start_place") or _detect_source_place(current_data.get("graph", {}))
    end_place = current_data["metadata"].get("end_place") or _detect_sink_place(current_data.get("graph", {}))

    try:
        trace = _parse_uploaded_trace(f.read(), file_type, mapping)
    except PartialTraceError as exc:
        return jsonify({"error": str(exc)}), 400

    prepared = PreparedDomainInput.from_dict(current_data)
    pnm = build_petrinet_model_from_prepared(prepared, start_place, end_place)
    replayer = TraceReplayer(prepared, pnm, config=AnalysisConfig())
    snapshot = replayer.replay_partial_trace(trace)

    return jsonify({
        "init_places": sorted(snapshot.marking),
        "init_effects": [{"attribute": k, "value": v} for k, v in snapshot.attributes.items()],
        "n_events": len(trace),
        "warnings": snapshot.warnings,
    })


# ---------------------------------------------------------------------------
# Planner endpoints
# ---------------------------------------------------------------------------

@api.route("/planners", methods=["GET"])
def get_planners():
    """Return availability status for each supported planner."""
    from planning.planner_runner import get_planners_status
    return jsonify(get_planners_status())


# ── Planner configuration CRUD ────────────────────────────────────────────────

@api.route("/<config_name>/planner-config", methods=["GET"])
def get_planner_config(config_name: str):
    config_dir = _config_dir(config_name)
    if not os.path.isdir(config_dir):
        return jsonify({"error": "Configuration not found"}), 404
    cfg = load_planner_config(Path(config_dir))
    return jsonify(config_to_dict(cfg))


@api.route("/<config_name>/planner-config", methods=["PUT"])
def put_planner_config(config_name: str):
    config_dir = _config_dir(config_name)
    if not os.path.isdir(config_dir):
        return jsonify({"error": "Configuration not found"}), 404
    body = request.get_json(force=True) or {}

    from dataclasses import asdict
    saved = load_planner_config(Path(config_dir))
    if "fast_downward" in body:
        saved.fast_downward = _merge(FastDownwardConfig, body["fast_downward"])
    if "optic" in body:
        saved.optic = _merge(OpticConfig, body["optic"])
    save_planner_config(saved, Path(config_dir))
    return jsonify(config_to_dict(saved))


@api.route("/<config_name>/planner-config/reset", methods=["POST"])
def reset_planner_config(config_name: str):
    config_dir = _config_dir(config_name)
    if not os.path.isdir(config_dir):
        return jsonify({"error": "Configuration not found"}), 404
    body = request.get_json(force=True) or {}
    planner = body.get("planner")  # "fast_downward" | "optic" | None (reset both)

    saved = load_planner_config(Path(config_dir))
    if planner in (None, "fast_downward"):
        saved.fast_downward = FastDownwardConfig()
    if planner in (None, "optic"):
        saved.optic = OpticConfig()
    save_planner_config(saved, Path(config_dir))
    return jsonify(config_to_dict(saved))


@api.route("/<config_name>/run-planner", methods=["POST"])
def run_planner_endpoint(config_name: str):
    body = request.get_json(force=True) or {}
    planner = body.get("planner", "fast_downward")
    body_options = body.get("options", {})

    current_path = _current_path(config_name)
    if not os.path.isfile(current_path):
        return jsonify({"error": "Configuration not found"}), 404

    pddl_out = _pddl_dir(config_name)
    if not os.path.isfile(os.path.join(pddl_out, "problem.pddl")):
        return jsonify({"error": "problem.pddl not found — build the problem first."}), 404

    config_dir = Path(_config_dir(config_name))
    saved = load_planner_config(config_dir)
    if planner == "fast_downward":
        base = asdict(saved.fast_downward)
        base.update({k: v for k, v in body_options.items() if k in base})
        saved.fast_downward = FastDownwardConfig(**base)
        options = asdict(saved.fast_downward)
    elif planner == "optic":
        base = asdict(saved.optic)
        base.update({k: v for k, v in body_options.items() if k in base})
        saved.optic = OpticConfig(**base)
        options = asdict(saved.optic)
    else:
        options = body_options
    save_planner_config(saved, config_dir)

    job_id = tracker.create()
    threading.Thread(
        target=_planner_thread,
        args=(job_id, pddl_out, planner, options),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


# ---------------------------------------------------------------------------
# Petri net editing endpoints
# ---------------------------------------------------------------------------

@api.route("/<config_name>/transition/<path:activity_name>", methods=["PATCH"])
def patch_transition(config_name: str, activity_name: str):
    """Update cost, preconditions, and/or effects for a transition in current.json."""
    current_path = _current_path(config_name)
    if not os.path.isfile(current_path):
        return jsonify({"error": "Configuration not found"}), 404
    body = request.get_json(force=True) or {}
    data = _read_json(current_path)
    transitions = data.get("transitions", {})
    if activity_name not in transitions:
        return jsonify({"error": f"Transition '{activity_name}' not found"}), 404
    t = transitions[activity_name]
    if "cost" in body:
        t["cost"] = float(body["cost"])
    if "preconditions" in body:
        t["preconditions"] = body["preconditions"]
    if "effects" in body:
        t["effects"] = body["effects"]
    if "effect_groups" in body:
        t["effect_groups"] = body["effect_groups"]
    if "duration" in body:
        d = body["duration"]
        eff_min = float(d.get("effective_min", 0))
        eff_max = float(d.get("effective_max", 0))
        if eff_min < 0 or eff_max < eff_min:
            return jsonify({"error": "Invalid duration: min must be >= 0 and max must be >= min"}), 400
        t["duration"] = {"effective_min": eff_min, "effective_max": eff_max, "source": "external"}

        # If the log had no timestamps, check whether every transition now has a
        # valid duration — if so, temporal planning becomes available.
        if data.get("metadata", {}).get("has_timestamps") is False:
            if all("duration" in tr for tr in data["transitions"].values()):
                data.setdefault("metadata", {})["has_timestamps"] = True

    _write_json(data, current_path)
    return jsonify({"status": "ok"})


@api.route("/<config_name>/xor-split/<place_name>/<path:branch_activity>", methods=["PATCH"])
def patch_xor_branch(config_name: str, place_name: str, branch_activity: str):
    """Update conditions and/or probability for one branch of an XOR split in current.json."""
    current_path = _current_path(config_name)
    if not os.path.isfile(current_path):
        return jsonify({"error": "Configuration not found"}), 404
    body = request.get_json(force=True) or {}
    data = _read_json(current_path)
    xor_splits = data.get("xor_splits", {})
    if place_name not in xor_splits:
        return jsonify({"error": f"XOR split '{place_name}' not found"}), 404
    branches = xor_splits[place_name].get("branches", {})
    if branch_activity not in branches:
        return jsonify({"error": f"Branch '{branch_activity}' not found in XOR split '{place_name}'"}), 404
    branch = branches[branch_activity]
    if "conditions" in body:
        branch["conditions"] = body["conditions"]
    if "probability" in body:
        branch["probability"] = float(body["probability"])
    _write_json(data, current_path)
    return jsonify({"status": "ok"})


@api.route("/<config_name>/rebuild-domain", methods=["POST"])
def rebuild_domain(config_name: str):
    """Rebuild domain.pddl from the current edited current.json.

    Skips the rebuild if current.json hash and use_durative flag have not changed
    since the last successful build (state stored in pddl/.domain_hash).
    """
    current_path = _current_path(config_name)
    if not os.path.isfile(current_path):
        return jsonify({"error": "Configuration not found"}), 404

    current_hash = _compute_current_hash(config_name)
    saved_state = _read_domain_state(config_name)

    body = request.get_json(force=True, silent=True) or {}
    saved_durative = saved_state.get("use_durative", False)
    saved_costs = saved_state.get("use_costs", False)
    use_durative = bool(body.get("use_durative", saved_durative))
    use_costs = bool(body.get("use_costs", saved_costs))

    if (
        current_hash
        and current_hash == saved_state.get("hash")
        and use_durative == saved_durative
        and use_costs == saved_costs
    ):
        return jsonify({"status": "ok", "skipped": True})

    data = _read_json(current_path)
    domain, _variant_map = build_domain_with_variant_map(
        PreparedDomainInput.from_dict(data),
        domain_name=config_name, use_durative=use_durative, use_costs=use_costs,
    )
    pddl_dir_path = _pddl_dir(config_name)
    domain.write(Path(pddl_dir_path) / "domain.pddl")
    _save_domain_state(pddl_dir_path, current_hash, use_durative, use_costs)
    return jsonify({"status": "ok", "skipped": False})


def _planner_thread(
    job_id: str,
    pddl_out: str,
    planner: str,
    options: Dict[str, Any],
) -> None:
    try:
        from pathlib import Path as _Path
        from planning.planner_runner import run_planner

        def _log(msg: str) -> None:
            tracker.append_log(job_id, msg)

        _log(f"Starting {planner}...")
        result = run_planner(
            pddl_dir=_Path(pddl_out),
            planner=planner,
            options=options,
            log_fn=_log,
        )
        tracker.complete(job_id, {
            "success": result.success,
            "solvability": result.solvability,
            "plan_actions": result.plan_actions,
            "plan_text": result.plan_text,
            "metrics": result.metrics,
            "message": result.message,
            "planner": result.planner,
        })
    except Exception:
        err = traceback.format_exc()
        tracker.append_log(job_id, f"Error: {err.splitlines()[-1]}")
        tracker.fail(job_id, err)
