"""Flask blueprint exposing the REST API for the Petri net web UI."""

import hashlib
import json
import os
import tempfile
import threading
import traceback
from dataclasses import fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from models import AnalysisConfig
from pipeline import Pipeline
import core_utils as utils
from planning.planner_config import (
    load_planner_config, save_planner_config, config_to_dict,
    FastDownwardConfig, OpticConfig, _merge,
)
from encoding.domain_rebuilder import DomainRebuilder
from encoding.pddl_writer import PDDLWriter
from dataclasses import asdict

from flask import Blueprint, current_app, jsonify, request, send_file

from web.jobs import tracker
from encoding.problem_builder import ProblemBuilder

api = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_dir(config_name: str) -> str:
    base = os.path.abspath(current_app.config["DATA_DIR"])
    return os.path.join(base, os.path.basename(config_name))


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


def _save_domain_state(pddl_dir: str, hash_str: str, use_durative: bool) -> None:
    """Write domain state to .domain_hash. Usable from background threads (no Flask context)."""
    p = os.path.join(pddl_dir, ".domain_hash")
    os.makedirs(pddl_dir, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"hash": hash_str, "use_durative": use_durative}, fh)


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
# Feature 1 — XES log upload
# ---------------------------------------------------------------------------

@api.route("/upload-log", methods=["POST"])
def upload_log():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".xes"):
        return jsonify({"error": "File must be a .xes event log"}), 400

    config_name = utils.sanitize_name(Path(file.filename).stem)
    logs_dir = os.path.join(current_app.config["PROJECT_ROOT"], "logs")
    os.makedirs(logs_dir, exist_ok=True)

    save_path = os.path.join(logs_dir, f"{config_name}.xes")
    file.save(save_path)

    return jsonify({"config_name": config_name, "log_path": save_path})


# ---------------------------------------------------------------------------
# Feature 2 — Pipeline execution
# ---------------------------------------------------------------------------

@api.route("/<log_name>/run-pipeline", methods=["POST"])
def run_pipeline(log_name: str):
    body = request.get_json(force=True) or {}
    pipeline_params = body.get("pipeline", {})
    config_dict = body.get("config", {})

    log_path = os.path.join(
        current_app.config["PROJECT_ROOT"], "logs", f"{log_name}.xes"
    )
    if not os.path.isfile(log_path):
        return jsonify({"error": f"Log file not found: {log_name}.xes"}), 404

    data_dir = current_app.config["DATA_DIR"]
    pddl_out = os.path.join(data_dir, log_name, "pddl")

    job_id = tracker.create()
    threading.Thread(
        target=_pipeline_thread,
        args=(job_id, log_path, data_dir, pddl_out, pipeline_params, config_dict),
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
) -> None:
    try:


        tracker.append_log(job_id, "Initializing configuration...")
        config = _build_analysis_config(config_dict)

        allowed_kwargs = {
            "domain_name", "discovery_algorithm",
            "coverage_percentage", "use_durative", "use_activity_classifier",
        }
        kwargs = {k: v for k, v in pipeline_params.items() if k in allowed_kwargs}

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
        current_json = Path(data_dir) / config_name / "current.json"
        if current_json.exists():
            h = hashlib.sha256(current_json.read_bytes()).hexdigest()
            _save_domain_state(pddl_out, h, use_durative)

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
    init_place = body.get("init_place") or None
    init_effects = body.get("init", [])
    goal_sop = body.get("goal", [])

    if not goal_sop or not any(goal_sop):
        return jsonify({"error": "At least one goal clause is required"}), 400

    pddl_out = _pddl_dir(config_name)
    current_path = _current_path(config_name)
    if not os.path.isfile(current_path):
        return jsonify({"error": "Configuration not found"}), 404
    data = _read_json(current_path)

    attribute_catalog = data.get("attribute_catalog", {})

    problem_text = ProblemBuilder().build(
        problem_name=f"{config_name}_prediction",
        domain_name=config_name,
        init_place=init_place,
        init_effects=init_effects,
        goal_sop=goal_sop,
        attribute_catalog=attribute_catalog,
    )

    os.makedirs(pddl_out, exist_ok=True)
    problem_path = os.path.join(pddl_out, "problem.pddl")
    with open(problem_path, "w", encoding="utf-8") as fh:
        fh.write(problem_text)

    return jsonify({
        "status": "built",
        "problem_path": problem_path,
        "problem_text": problem_text,
    })


# ---------------------------------------------------------------------------
# Planner endpoints
# ---------------------------------------------------------------------------


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

    # Rebuild domain if hash or use_durative flag don't match the chosen planner.
    use_durative = (planner == "optic")
    current_hash = _compute_current_hash(config_name)
    saved_state = _read_domain_state(config_name)
    domain_is_current = (
        current_hash
        and current_hash == saved_state.get("hash")
        and use_durative == saved_state.get("use_durative")
    )
    if not domain_is_current:
        data = _read_json(current_path)
        domain = DomainRebuilder().rebuild(data, domain_name=config_name, use_durative=use_durative)
        Path(pddl_out).mkdir(parents=True, exist_ok=True)
        PDDLWriter().write_domain(domain, Path(pddl_out) / "domain.pddl")
        _save_domain_state(pddl_out, current_hash, use_durative)


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
    use_durative = bool(body.get("use_durative", saved_durative))

    if (
        current_hash
        and current_hash == saved_state.get("hash")
        and use_durative == saved_durative
    ):
        return jsonify({"status": "ok", "skipped": True})

    data = _read_json(current_path)
    domain = DomainRebuilder().rebuild(data, domain_name=config_name, use_durative=use_durative)
    pddl_dir_path = _pddl_dir(config_name)
    Path(pddl_dir_path).mkdir(parents=True, exist_ok=True)
    PDDLWriter().write_domain(domain, Path(pddl_dir_path) / "domain.pddl")
    _save_domain_state(pddl_dir_path, current_hash, use_durative)
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
