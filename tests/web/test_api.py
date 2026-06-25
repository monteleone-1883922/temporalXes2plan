"""Tests for web.api Flask blueprint — uses Flask test client."""
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

import pytest

from web.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_DATA = {
    "graph": {
        "nodes": [
            {"id": "p_start", "type": "place", "label": "p_start"},
            {"id": "t1", "type": "transition", "label": "register", "is_silent": False},
            {"id": "p_end", "type": "place", "label": "p_end"},
        ],
        "edges": [
            {"source": "p_start", "target": "t1"},
            {"source": "t1", "target": "p_end"},
        ],
    },
    "metadata": {
        "start_place": "p_start",
        "end_place": "p_end",
    },
    "transitions": {
        "register": {
            "activity_name": "register",
            "input_places": ["p_start"],
            "preconditions": [],
            "effects": [],
            "effect_groups": [],
            "cost": 0.0,
            "duration": {"effective_min": 10.0, "effective_max": 30.0},
            "_meta": {"total_firings": 5, "related_effects": [], "incompatible_effects": []},
        }
    },
    "xor_splits": {
        "p_xor": {
            "branches": {
                "admit": {
                    "activity_name": "admit",
                    "conditions": [],
                    "probability": 0.7,
                    "_meta": {"cascade_level": 2, "total_samples": 10},
                },
                "discharge": {
                    "activity_name": "discharge",
                    "conditions": [],
                    "probability": 0.3,
                    "_meta": {"cascade_level": 2, "total_samples": 10},
                },
            }
        }
    },
    "attribute_catalog": {
        "status": {"type": "categorical", "possible_values": ["admitted", "discharged"]},
    },
}


@pytest.fixture()
def data_dir(tmp_path):
    """A temporary DATA_DIR with a 'test_cfg' configuration pre-loaded."""
    cfg_dir = tmp_path / "test_cfg"
    cfg_dir.mkdir()
    current = cfg_dir / "current.json"
    original = cfg_dir / "original.json"
    current.write_text(json.dumps(MINIMAL_DATA), encoding="utf-8")
    original.write_text(json.dumps(MINIMAL_DATA), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def app(data_dir):
    flask_app = create_app(data_dir=str(data_dir))
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# GET /api/<config_name>/petri-net
# ---------------------------------------------------------------------------

class TestGetCurrent:
    def test_returns_200_for_existing_config(self, client):
        resp = client.get("/api/test_cfg/petri-net")
        assert resp.status_code == 200

    def test_returns_graph_key(self, client):
        data = client.get("/api/test_cfg/petri-net").get_json()
        assert "graph" in data

    def test_returns_transitions_key(self, client):
        data = client.get("/api/test_cfg/petri-net").get_json()
        assert "transitions" in data

    def test_returns_404_for_missing_config(self, client):
        resp = client.get("/api/nonexistent/petri-net")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/<config_name>/petri-net/original
# ---------------------------------------------------------------------------

class TestGetOriginal:
    def test_returns_200_for_existing_config(self, client):
        resp = client.get("/api/test_cfg/petri-net/original")
        assert resp.status_code == 200

    def test_returns_404_for_missing_config(self, client):
        resp = client.get("/api/nonexistent/petri-net/original")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/<config_name>/petri-net/reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_ok(self, client):
        resp = client.post("/api/test_cfg/petri-net/reset")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_reset_restores_original(self, client, data_dir):
        # Modify current first
        current_path = data_dir / "test_cfg" / "current.json"
        modified = json.loads(current_path.read_text())
        modified["transitions"]["register"]["cost"] = 99.0
        current_path.write_text(json.dumps(modified))

        client.post("/api/test_cfg/petri-net/reset")

        restored = json.loads(current_path.read_text())
        assert restored["transitions"]["register"]["cost"] == 0.0

    def test_reset_404_for_missing_original(self, client, data_dir):
        (data_dir / "test_cfg" / "original.json").unlink()
        resp = client.post("/api/test_cfg/petri-net/reset")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/<config_name>/import
# ---------------------------------------------------------------------------

class TestImportConfig:
    def test_import_valid_json_returns_ok(self, client):
        payload = json.dumps(MINIMAL_DATA).encode()
        resp = client.post(
            "/api/test_cfg/import",
            data={"file": (BytesIO(payload), "export.json")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_import_missing_file_returns_400(self, client):
        resp = client.post("/api/test_cfg/import")
        assert resp.status_code == 400

    def test_import_wrong_extension_returns_400(self, client):
        resp = client.post(
            "/api/test_cfg/import",
            data={"file": (BytesIO(b"data"), "file.xml")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_import_invalid_json_returns_400(self, client):
        resp = client.post(
            "/api/test_cfg/import",
            data={"file": (BytesIO(b"NOT JSON"), "file.json")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_import_missing_keys_returns_400(self, client):
        incomplete = json.dumps({"graph": {}}).encode()
        resp = client.post(
            "/api/test_cfg/import",
            data={"file": (BytesIO(incomplete), "file.json")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/analysis-config/defaults
# ---------------------------------------------------------------------------

class TestAnalysisConfigDefaults:
    def test_returns_200(self, client):
        resp = client.get("/api/analysis-config/defaults")
        assert resp.status_code == 200

    def test_returns_dict_with_fields(self, client):
        data = client.get("/api/analysis-config/defaults").get_json()
        assert isinstance(data, dict)
        assert len(data) > 0


# ---------------------------------------------------------------------------
# GET /api/planners
# ---------------------------------------------------------------------------

class TestGetPlanners:
    def test_returns_200(self, client):
        resp = client.get("/api/planners")
        assert resp.status_code == 200

    def test_has_fast_downward_key(self, client):
        data = client.get("/api/planners").get_json()
        assert "fast_downward" in data

    def test_has_optic_key(self, client):
        data = client.get("/api/planners").get_json()
        assert "optic" in data


# ---------------------------------------------------------------------------
# GET /api/<config_name>/planner-config
# ---------------------------------------------------------------------------

class TestGetPlannerConfig:
    def test_returns_200_for_existing_config(self, client):
        resp = client.get("/api/test_cfg/planner-config")
        assert resp.status_code == 200

    def test_returns_fast_downward_and_optic_keys(self, client):
        data = client.get("/api/test_cfg/planner-config").get_json()
        assert "fast_downward" in data
        assert "optic" in data

    def test_returns_404_for_nonexistent_config(self, client):
        resp = client.get("/api/nonexistent/planner-config")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/<config_name>/planner-config
# ---------------------------------------------------------------------------

class TestPutPlannerConfig:
    def test_updates_fast_downward_timeout(self, client):
        resp = client.put(
            "/api/test_cfg/planner-config",
            json={"fast_downward": {"timeout": 120}},
        )
        assert resp.status_code == 200
        assert resp.get_json()["fast_downward"]["timeout"] == 120

    def test_partial_update_preserves_other_fields(self, client):
        resp = client.put(
            "/api/test_cfg/planner-config",
            json={"fast_downward": {"timeout": 90}},
        )
        data = resp.get_json()
        assert data["fast_downward"]["search"] == "astar_lmcut"

    def test_returns_404_for_nonexistent_config(self, client):
        resp = client.put(
            "/api/nonexistent/planner-config",
            json={"fast_downward": {"timeout": 10}},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/<config_name>/planner-config/reset
# ---------------------------------------------------------------------------

class TestResetPlannerConfig:
    def test_reset_returns_defaults(self, client):
        client.put("/api/test_cfg/planner-config", json={"fast_downward": {"timeout": 999}})
        resp = client.post("/api/test_cfg/planner-config/reset", json={})
        assert resp.status_code == 200
        assert resp.get_json()["fast_downward"]["timeout"] == 30

    def test_reset_single_planner(self, client):
        client.put("/api/test_cfg/planner-config", json={"optic": {"timeout": 999}})
        resp = client.post("/api/test_cfg/planner-config/reset", json={"planner": "optic"})
        assert resp.status_code == 200
        assert resp.get_json()["optic"]["timeout"] == 60

    def test_returns_404_for_nonexistent_config(self, client):
        resp = client.post("/api/nonexistent/planner-config/reset", json={})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /api/<config_name>/transition/<activity_name>
# ---------------------------------------------------------------------------

class TestPatchTransition:
    def test_update_cost(self, client):
        resp = client.patch(
            "/api/test_cfg/transition/register",
            json={"cost": 5.0},
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_cost_persisted(self, client, data_dir):
        client.patch("/api/test_cfg/transition/register", json={"cost": 7.5})
        data = json.loads((data_dir / "test_cfg" / "current.json").read_text())
        assert data["transitions"]["register"]["cost"] == 7.5

    def test_update_preconditions(self, client):
        new_prec = [[{"attribute": "status", "predicate": "=", "value": "admitted"}]]
        resp = client.patch("/api/test_cfg/transition/register", json={"preconditions": new_prec})
        assert resp.status_code == 200

    def test_update_effects(self, client):
        new_effects = [{"attribute": "status", "value": "discharged", "probability": 1.0, "preconditions": []}]
        resp = client.patch("/api/test_cfg/transition/register", json={"effects": new_effects})
        assert resp.status_code == 200

    def test_returns_404_for_missing_config(self, client):
        resp = client.patch("/api/nonexistent/transition/register", json={"cost": 1.0})
        assert resp.status_code == 404

    def test_returns_404_for_missing_transition(self, client):
        resp = client.patch("/api/test_cfg/transition/nonexistent_activity", json={"cost": 1.0})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /api/<config_name>/xor-split/<place_name>/<branch_activity>
# ---------------------------------------------------------------------------

class TestPatchXorBranch:
    def test_update_probability(self, client):
        resp = client.patch(
            "/api/test_cfg/xor-split/p_xor/admit",
            json={"probability": 0.9},
        )
        assert resp.status_code == 200

    def test_probability_persisted(self, client, data_dir):
        client.patch("/api/test_cfg/xor-split/p_xor/admit", json={"probability": 0.85})
        data = json.loads((data_dir / "test_cfg" / "current.json").read_text())
        assert data["xor_splits"]["p_xor"]["branches"]["admit"]["probability"] == 0.85

    def test_update_conditions(self, client):
        conditions = [[{"attribute": "status", "predicate": "=", "value": "admitted"}]]
        resp = client.patch("/api/test_cfg/xor-split/p_xor/admit", json={"conditions": conditions})
        assert resp.status_code == 200

    def test_returns_404_for_missing_config(self, client):
        resp = client.patch("/api/nonexistent/xor-split/p_xor/admit", json={"probability": 0.5})
        assert resp.status_code == 404

    def test_returns_404_for_missing_xor_split(self, client):
        resp = client.patch("/api/test_cfg/xor-split/nonexistent_place/admit", json={"probability": 0.5})
        assert resp.status_code == 404

    def test_returns_404_for_missing_branch(self, client):
        resp = client.patch("/api/test_cfg/xor-split/p_xor/nonexistent_branch", json={"probability": 0.5})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/<config_name>/rebuild-domain
# ---------------------------------------------------------------------------

class TestRebuildDomain:
    def test_returns_ok(self, client):
        resp = client.post("/api/test_cfg/rebuild-domain")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_creates_domain_pddl(self, client, data_dir):
        client.post("/api/test_cfg/rebuild-domain")
        assert (data_dir / "test_cfg" / "pddl" / "domain.pddl").exists()

    def test_skips_rebuild_when_unchanged(self, client):
        client.post("/api/test_cfg/rebuild-domain")
        resp = client.post("/api/test_cfg/rebuild-domain")
        assert resp.get_json()["skipped"] is True

    def test_rebuilds_after_change(self, client, data_dir):
        client.post("/api/test_cfg/rebuild-domain")
        # Modify current.json
        p = data_dir / "test_cfg" / "current.json"
        data = json.loads(p.read_text())
        data["transitions"]["register"]["cost"] = 9.9
        p.write_text(json.dumps(data))
        resp = client.post("/api/test_cfg/rebuild-domain")
        assert resp.get_json()["skipped"] is False

    def test_returns_404_for_missing_config(self, client):
        resp = client.post("/api/nonexistent/rebuild-domain")
        assert resp.status_code == 404

    def test_rebuilds_when_durative_flag_changes(self, client):
        client.post("/api/test_cfg/rebuild-domain", json={"use_durative": False})
        resp = client.post("/api/test_cfg/rebuild-domain", json={"use_durative": True})
        assert resp.get_json()["skipped"] is False

    def test_skips_rebuild_when_same_durative_flag(self, client):
        client.post("/api/test_cfg/rebuild-domain", json={"use_durative": True})
        resp = client.post("/api/test_cfg/rebuild-domain", json={"use_durative": True})
        assert resp.get_json()["skipped"] is True


# ---------------------------------------------------------------------------
# POST /api/<config_name>/build-problem
# ---------------------------------------------------------------------------

class TestBuildProblem:
    def test_returns_400_when_goal_empty(self, client):
        resp = client.post("/api/test_cfg/build-problem", json={"goal": []})
        assert resp.status_code == 400

    def test_returns_404_for_missing_config(self, client):
        resp = client.post(
            "/api/nonexistent/build-problem",
            json={"goal": [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]},
        )
        assert resp.status_code == 404

    def test_builds_problem_pddl(self, client, data_dir):
        resp = client.post(
            "/api/test_cfg/build-problem",
            json={"goal": [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "built"
        assert "problem_text" in data
        assert (data_dir / "test_cfg" / "pddl" / "problem.pddl").exists()


# ---------------------------------------------------------------------------
# POST /api/<config_name>/build-problem — require_completion
# ---------------------------------------------------------------------------

class TestBuildProblemRequireCompletion:
    def test_require_completion_only_goal_returns_200(self, client):
        resp = client.post(
            "/api/test_cfg/build-problem",
            json={"goal": [], "require_completion": True},
        )
        assert resp.status_code == 200

    def test_require_completion_goal_contains_marked_end_place(self, client):
        resp = client.post(
            "/api/test_cfg/build-problem",
            json={"goal": [], "require_completion": True},
        )
        assert "(marked p_end)" in resp.get_json()["problem_text"]

    def test_require_completion_false_no_marked_end_place_in_goal(self, client):
        resp = client.post(
            "/api/test_cfg/build-problem",
            json={
                "goal": [[{"attribute": "status", "predicate": "=", "value": "discharged"}]],
                "require_completion": False,
            },
        )
        assert "(marked p_end)" not in resp.get_json()["problem_text"]

    def test_require_completion_combined_with_attribute_goal(self, client):
        resp = client.post(
            "/api/test_cfg/build-problem",
            json={
                "goal": [[{"attribute": "status", "predicate": "=", "value": "discharged"}]],
                "require_completion": True,
            },
        )
        text = resp.get_json()["problem_text"]
        assert "(status_is discharged)" in text
        assert "(marked p_end)" in text

    def test_require_completion_uses_metadata_end_place(self, client, data_dir):
        resp = client.post(
            "/api/test_cfg/build-problem",
            json={"goal": [], "require_completion": True},
        )
        assert "(marked p_end)" in resp.get_json()["problem_text"]

    def test_require_completion_fallback_detect_sink(self, client, data_dir):
        # Overwrite current.json with no metadata but a detectable sink place
        import json as _json
        no_meta = {
            "graph": {
                "nodes": [
                    {"id": "p_start", "type": "place", "label": "p_start"},
                    {"id": "t1", "type": "transition", "label": "act", "is_silent": False},
                    {"id": "p_sink", "type": "place", "label": "p_sink"},
                ],
                "edges": [
                    {"source": "p_start", "target": "t1"},
                    {"source": "t1", "target": "p_sink"},
                ],
            },
            "transitions": {},
            "xor_splits": {},
            "attribute_catalog": {},
        }
        (data_dir / "test_cfg" / "current.json").write_text(_json.dumps(no_meta), encoding="utf-8")
        resp = client.post(
            "/api/test_cfg/build-problem",
            json={"goal": [], "require_completion": True},
        )
        assert "(marked p_sink)" in resp.get_json()["problem_text"]

    def test_empty_goal_without_require_completion_returns_400(self, client):
        resp = client.post("/api/test_cfg/build-problem", json={"goal": []})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/<config_name>/build-problem — deadline
# ---------------------------------------------------------------------------

class TestBuildProblemDeadline:
    _GOAL = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]

    def test_no_til_without_deadline(self, client):
        resp = client.post("/api/test_cfg/build-problem", json={"goal": self._GOAL})
        assert "(at " not in resp.get_json()["problem_text"]

    def test_deadline_til_in_problem_init(self, client):
        resp = client.post(
            "/api/test_cfg/build-problem",
            json={"goal": self._GOAL, "deadline": 3600},
        )
        assert "(at 3600.0 (deadline_exceeded))" in resp.get_json()["problem_text"]

    def test_deadline_rebuilds_domain_with_deadline_predicate(self, client, data_dir):
        client.post(
            "/api/test_cfg/build-problem",
            json={"goal": self._GOAL, "deadline": 3600, "planner": "optic"},
        )
        domain_text = (data_dir / "test_cfg" / "pddl" / "domain.pddl").read_text()
        assert "(deadline_exceeded)" in domain_text

    def test_no_deadline_domain_has_no_deadline_predicate(self, client, data_dir):
        # First build with deadline (creates domain with predicate), then remove it.
        client.post("/api/test_cfg/build-problem", json={"goal": self._GOAL, "deadline": 3600, "planner": "optic"})
        client.post("/api/test_cfg/build-problem", json={"goal": self._GOAL, "planner": "optic"})
        domain_text = (data_dir / "test_cfg" / "pddl" / "domain.pddl").read_text()
        assert "deadline_exceeded" not in domain_text

    def test_deadline_zero_not_in_problem(self, client):
        resp = client.post(
            "/api/test_cfg/build-problem",
            json={"goal": self._GOAL, "deadline": 0},
        )
        assert "(at " not in resp.get_json()["problem_text"]

    def test_deadline_negative_not_in_problem(self, client):
        resp = client.post(
            "/api/test_cfg/build-problem",
            json={"goal": self._GOAL, "deadline": -5},
        )
        assert "(at " not in resp.get_json()["problem_text"]

    def test_deadline_domain_state_saved(self, client, data_dir):
        client.post(
            "/api/test_cfg/build-problem",
            json={"goal": self._GOAL, "deadline": 120, "planner": "optic"},
        )
        import json as _json
        state = _json.loads((data_dir / "test_cfg" / "pddl" / ".domain_hash").read_text())
        assert state.get("has_deadline") is True


# ---------------------------------------------------------------------------
# GET /api/jobs/<job_id>
# ---------------------------------------------------------------------------

class TestGetJob:
    def test_returns_404_for_unknown_job(self, client):
        resp = client.get("/api/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_returns_job_status_for_known_job(self, client):
        from web.jobs import tracker
        job_id = tracker.create()
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "running"


# ---------------------------------------------------------------------------
# POST /api/<config_name>/run-planner
# ---------------------------------------------------------------------------

class TestRunPlanner:
    def _build_problem(self, client):
        client.post(
            "/api/test_cfg/build-problem",
            json={"goal": [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]},
        )

    def test_returns_404_when_problem_missing(self, client):
        resp = client.post("/api/test_cfg/run-planner", json={"planner": "fast_downward"})
        assert resp.status_code == 404

    def test_returns_404_for_missing_config(self, client):
        resp = client.post("/api/nonexistent/run-planner", json={"planner": "fast_downward"})
        assert resp.status_code == 404

    def test_returns_job_id_when_problem_exists(self, client):
        self._build_problem(client)
        resp = client.post("/api/test_cfg/run-planner", json={"planner": "fast_downward"})
        assert resp.status_code == 200
        assert "job_id" in resp.get_json()

    def test_fast_downward_writes_non_durative_domain(self, client, data_dir):
        self._build_problem(client)
        client.post("/api/test_cfg/run-planner", json={"planner": "fast_downward"})
        domain_text = (data_dir / "test_cfg" / "pddl" / "domain.pddl").read_text()
        assert ":durative-actions" not in domain_text

    def test_optic_writes_domain_hash_as_durative(self, client, data_dir):
        _GOAL = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        client.post("/api/test_cfg/build-problem", json={"goal": _GOAL, "planner": "optic"})
        import json as _json
        state = _json.loads((data_dir / "test_cfg" / "pddl" / ".domain_hash").read_text())
        assert state["use_durative"] is True

    def test_fast_downward_writes_domain_hash_as_non_durative(self, client, data_dir):
        self._build_problem(client)
        import json as _json
        state = _json.loads((data_dir / "test_cfg" / "pddl" / ".domain_hash").read_text())
        assert state["use_durative"] is False

    def test_domain_not_rebuilt_when_hash_and_flag_match(self, client, data_dir):
        self._build_problem(client)
        mtime_before = (data_dir / "test_cfg" / "pddl" / "domain.pddl").stat().st_mtime
        self._build_problem(client)
        mtime_after = (data_dir / "test_cfg" / "pddl" / "domain.pddl").stat().st_mtime
        assert mtime_before == mtime_after

    def test_domain_rebuilt_when_planner_switches_to_optic(self, client, data_dir):
        _GOAL = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        client.post("/api/test_cfg/build-problem", json={"goal": _GOAL, "planner": "fast_downward"})
        mtime_before = (data_dir / "test_cfg" / "pddl" / "domain.pddl").stat().st_mtime
        client.post("/api/test_cfg/build-problem", json={"goal": _GOAL, "planner": "optic"})
        mtime_after = (data_dir / "test_cfg" / "pddl" / "domain.pddl").stat().st_mtime
        assert mtime_after > mtime_before

    def test_use_costs_saved_in_domain_state(self, client, data_dir):
        _GOAL = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        client.post(
            "/api/test_cfg/build-problem",
            json={"goal": _GOAL, "metric": "minimize_cost"},
        )
        import json as _json
        state = _json.loads((data_dir / "test_cfg" / "pddl" / ".domain_hash").read_text())
        assert state["use_costs"] is True

    def test_use_costs_false_saved_without_metric(self, client, data_dir):
        _GOAL = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        client.post("/api/test_cfg/build-problem", json={"goal": _GOAL})
        import json as _json
        state = _json.loads((data_dir / "test_cfg" / "pddl" / ".domain_hash").read_text())
        assert state["use_costs"] is False

    def test_domain_rebuilt_when_metric_changes_to_minimize_cost(self, client, data_dir):
        _GOAL = [[{"attribute": "status", "predicate": "=", "value": "discharged"}]]
        client.post("/api/test_cfg/build-problem", json={"goal": _GOAL})
        mtime_before = (data_dir / "test_cfg" / "pddl" / "domain.pddl").stat().st_mtime
        client.post("/api/test_cfg/build-problem", json={"goal": _GOAL, "metric": "minimize_cost"})
        mtime_after = (data_dir / "test_cfg" / "pddl" / "domain.pddl").stat().st_mtime
        assert mtime_after > mtime_before
