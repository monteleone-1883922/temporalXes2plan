"""Flask blueprint exposing the REST API for the Petri net web UI."""

import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict

from flask import Blueprint, current_app, jsonify, request, send_file

api = Blueprint("api", __name__, url_prefix="/api")


def _config_dir(config_name: str) -> str:
    base = os.path.abspath(current_app.config["DATA_DIR"])
    safe_name = os.path.basename(config_name)
    return os.path.join(base, safe_name)


def _original_path(config_name: str) -> str:
    return os.path.join(_config_dir(config_name), "original.json")


def _current_path(config_name: str) -> str:
    return os.path.join(_config_dir(config_name), "current.json")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(data: Dict[str, Any], path: str) -> None:
    """Atomic write: write to temp file then rename."""
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


@api.route("/<config_name>/petri-net", methods=["GET"])
def get_current(config_name: str):
    """Return the current (editable) Petri net configuration."""
    path = _current_path(config_name)
    if not os.path.isfile(path):
        return jsonify({"error": "Configuration not found"}), 404
    return jsonify(_read_json(path))


@api.route("/<config_name>/petri-net/original", methods=["GET"])
def get_original(config_name: str):
    """Return the original (immutable) Petri net configuration."""
    path = _original_path(config_name)
    if not os.path.isfile(path):
        return jsonify({"error": "Original configuration not found"}), 404
    return jsonify(_read_json(path))


@api.route("/<config_name>/petri-net/reset", methods=["POST"])
def reset(config_name: str):
    """Reset the current configuration to the original."""
    orig = _original_path(config_name)
    if not os.path.isfile(orig):
        return jsonify({"error": "Original configuration not found"}), 404
    original = _read_json(orig)
    _write_json(original, _current_path(config_name))
    return jsonify({"status": "ok"})


@api.route("/<config_name>/export", methods=["GET"])
def export_config(config_name: str):
    """Download the current configuration as a JSON file."""
    path = _current_path(config_name)
    if not os.path.isfile(path):
        return jsonify({"error": "Configuration not found"}), 404
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config_name}_{ts}.json"
    return send_file(
        path,
        mimetype="application/json",
        as_attachment=True,
        download_name=filename,
    )


@api.route("/<config_name>/import", methods=["POST"])
def import_config(config_name: str):
    """Upload and apply a previously exported JSON configuration."""
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
        missing = required - set(data.keys())
        return jsonify({"error": f"Missing keys: {missing}"}), 400

    _write_json(data, _current_path(config_name))
    return jsonify({"status": "ok"})
