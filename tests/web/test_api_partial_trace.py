"""Tests for POST /api/<config_name>/replay-partial-trace."""
import io
import json

import pytest

from web.app import create_app


# ---------------------------------------------------------------------------
# XES helpers
# ---------------------------------------------------------------------------

def _xes(events: list) -> bytes:
    parts = [
        '<?xml version="1.0" encoding="utf-8" ?>',
        '<log xes.version="1.0" xmlns="http://www.xes-standard.org/">',
        "<trace>",
        '<string key="concept:name" value="case_1"/>',
    ]
    for ev in events:
        name = ev.get("concept:name", "activity")
        parts.append("<event>")
        parts.append(f'<string key="concept:name" value="{name}"/>')
        parts.append('<date key="time:timestamp" value="2024-01-01T08:00:00.000+00:00"/>')
        for k, v in ev.items():
            if k in ("concept:name", "time:timestamp"):
                continue
            parts.append(f'<string key="{k}" value="{v}"/>')
        parts.append("</event>")
    parts += ["</trace>", "</log>"]
    return "\n".join(parts).encode("utf-8")


def _xes_two_traces() -> bytes:
    parts = [
        '<?xml version="1.0" encoding="utf-8" ?>',
        '<log xes.version="1.0" xmlns="http://www.xes-standard.org/">',
        "<trace>",
        '<string key="concept:name" value="case_1"/>',
        "<event>",
        '<string key="concept:name" value="register"/>',
        '<date key="time:timestamp" value="2024-01-01T08:00:00.000+00:00"/>',
        "</event>",
        "</trace>",
        "<trace>",
        '<string key="concept:name" value="case_2"/>',
        "<event>",
        '<string key="concept:name" value="register"/>',
        '<date key="time:timestamp" value="2024-01-01T09:00:00.000+00:00"/>',
        "</event>",
        "</trace>",
        "</log>",
    ]
    return "\n".join(parts).encode("utf-8")


def _xes_empty_trace() -> bytes:
    parts = [
        '<?xml version="1.0" encoding="utf-8" ?>',
        '<log xes.version="1.0" xmlns="http://www.xes-standard.org/">',
        "<trace>",
        '<string key="concept:name" value="case_1"/>',
        "</trace>",
        "</log>",
    ]
    return "\n".join(parts).encode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_DATA = {
    "graph": {
        "nodes": [
            {"id": "p_start", "type": "place", "label": "p_start"},
            {"id": "t1", "type": "transition", "label": "register"},
            {"id": "p_end", "type": "place", "label": "p_end"},
        ],
        "edges": [
            {"source": "p_start", "target": "t1"},
            {"source": "t1", "target": "p_end"},
        ],
    },
    "metadata": {"start_place": "p_start", "end_place": "p_end"},
    "transitions": {
        "register": {
            "activity_name": "register",
            "input_places": ["p_start"],
            "preconditions": [],
            "effects": [],
            "effect_groups": [],
            "cost": 0.0,
            "duration": None,
        }
    },
    "xor_splits": {},
    "attribute_catalog": {
        "status": {
            "type": "categorical",
            "possible_values": ["admitted", "discharged"],
            "bin_boundaries": [],
        },
    },
}


@pytest.fixture()
def data_dir(tmp_path):
    cfg_dir = tmp_path / "test_cfg"
    cfg_dir.mkdir()
    (cfg_dir / "current.json").write_text(json.dumps(MINIMAL_DATA), encoding="utf-8")
    (cfg_dir / "original.json").write_text(json.dumps(MINIMAL_DATA), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def app(data_dir):
    flask_app = create_app(data_dir=str(data_dir))
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


def _upload(client, xes_bytes: bytes, filename: str = "trace.xes"):
    return client.post(
        "/api/test_cfg/replay-partial-trace",
        data={"file": (io.BytesIO(xes_bytes), filename)},
        content_type="multipart/form-data",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReplayPartialTrace:

    def test_returns_404_for_missing_config(self, client):
        resp = client.post(
            "/api/nonexistent/replay-partial-trace",
            data={"file": (io.BytesIO(b""), "trace.xes")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 404

    def test_returns_400_when_no_file(self, client):
        resp = client.post("/api/test_cfg/replay-partial-trace")
        assert resp.status_code == 400
        assert "No file" in resp.get_json()["error"]

    def test_returns_400_when_file_not_xes(self, client):
        resp = client.post(
            "/api/test_cfg/replay-partial-trace",
            data={"file": (io.BytesIO(b"{}"), "data.json")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert ".xes" in resp.get_json()["error"]

    def test_returns_400_when_unknown_activity(self, client):
        resp = _upload(client, _xes([{"concept:name": "ghost_activity"}]))
        assert resp.status_code == 400
        assert "not found in the Petri net" in resp.get_json()["error"]

    def test_returns_400_when_replay_stuck(self, client):
        # register requires p_start; if we pass two events that expect different places
        # we simulate a stuck scenario by sending two 'register' events — second one
        # expects p_start but token is at p_end.
        resp = _upload(client, _xes([
            {"concept:name": "register"},
            {"concept:name": "register"},
        ]))
        assert resp.status_code == 400
        assert "Replay stuck" in resp.get_json()["error"]

    def test_returns_400_when_multiple_traces(self, client):
        resp = _upload(client, _xes_two_traces())
        assert resp.status_code == 400
        assert "single-trace" in resp.get_json()["error"]

    def test_returns_200_with_init_places_and_effects(self, client):
        resp = _upload(client, _xes([{"concept:name": "register", "status": "admitted"}]))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["init_places"] == ["p_end"]
        assert {"attribute": "status", "value": "admitted"} in data["init_effects"]
        assert data["n_events"] == 1
        assert data["replayed_activities"] == ["register"]

    def test_returns_200_with_warnings_for_unknown_attributes(self, client):
        resp = _upload(client, _xes([{"concept:name": "register", "ward": "icu"}]))
        assert resp.status_code == 200
        data = resp.get_json()
        assert any("not in attribute catalog" in w for w in data["warnings"])

    def test_empty_trace_returns_start_place(self, client):
        resp = _upload(client, _xes_empty_trace())
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["init_places"] == ["p_start"]
        assert data["init_effects"] == []
        assert data["n_events"] == 0
