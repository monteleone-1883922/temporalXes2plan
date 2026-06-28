"""Unit tests for parsing.partial_trace_replayer.PartialTraceReplayer."""
import pytest

from parsing.partial_trace_replayer import PartialTraceError, PartialTraceReplayer


# ---------------------------------------------------------------------------
# XES byte helpers — build XML via list concatenation (no textwrap.dedent)
# so the <?xml declaration is always at column 0.
# ---------------------------------------------------------------------------

def _xes(events: list) -> bytes:
    """Build a minimal single-trace XES bytes payload."""
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
            if isinstance(v, bool):
                parts.append(f'<boolean key="{k}" value="{str(v).lower()}"/>')
            elif isinstance(v, (int, float)):
                parts.append(f'<float key="{k}" value="{v}"/>')
            else:
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
# current_data helpers
# ---------------------------------------------------------------------------

def _make_data(
    activities: list,
    catalog: dict | None = None,
    tau_after: str | None = None,
) -> dict:
    """Build current_data for a linear sequence: p_start → A → p_1 → B → p_2 …"""
    places = ["p_start"] + [f"p_{i}" for i in range(1, len(activities) + 1)]
    nodes = [{"id": p, "type": "place", "label": p} for p in places]
    edges = []
    transitions = {}

    for i, act in enumerate(activities):
        node_id = f"t_{act}"
        nodes.append({"id": node_id, "type": "transition", "label": act})
        edges.append({"source": places[i], "target": node_id})
        edges.append({"source": node_id, "target": places[i + 1]})
        transitions[act] = {"input_places": [places[i]]}

    if tau_after:
        tau_id = "t_tau_0"
        tau_place = "p_after_tau"
        nodes.append({"id": tau_id, "type": "transition", "label": ""})
        nodes.append({"id": tau_place, "type": "place", "label": tau_place})
        edges.append({"source": tau_after, "target": tau_id})
        edges.append({"source": tau_id, "target": tau_place})

    return {
        "graph": {"nodes": nodes, "edges": edges},
        "transitions": transitions,
        "attribute_catalog": catalog or {},
        "metadata": {"start_place": "p_start", "end_place": places[-1]},
    }


# ---------------------------------------------------------------------------
# Tests — replay logic
# ---------------------------------------------------------------------------

class TestReplayBasic:

    def test_empty_trace_returns_start_place(self):
        data = _make_data(["register"])
        result = PartialTraceReplayer().replay(_xes_empty_trace(), data)
        assert result["init_places"] == ["p_start"]
        assert result["init_effects"] == []
        assert result["n_events"] == 0

    def test_single_event_advances_token(self):
        data = _make_data(["register"])
        result = PartialTraceReplayer().replay(
            _xes([{"concept:name": "register"}]), data
        )
        assert result["init_places"] == ["p_1"]
        assert result["replayed_activities"] == ["register"]

    def test_two_events_advance_token_twice(self):
        data = _make_data(["register", "admit"])
        result = PartialTraceReplayer().replay(
            _xes([{"concept:name": "register"}, {"concept:name": "admit"}]), data
        )
        assert result["init_places"] == ["p_2"]
        assert result["replayed_activities"] == ["register", "admit"]
        assert result["n_events"] == 2

    def test_token_at_tau_only_place_is_valid(self):
        last_place = "p_1"
        data = _make_data(["register"], tau_after=last_place)
        result = PartialTraceReplayer().replay(
            _xes([{"concept:name": "register"}]), data
        )
        assert result["init_places"] == [last_place]
        assert not result["warnings"]

    def test_and_split_produces_multiple_tokens(self):
        data = {
            "graph": {
                "nodes": [
                    {"id": "p_start", "type": "place", "label": "p_start"},
                    {"id": "t_split", "type": "transition", "label": "split"},
                    {"id": "p_A", "type": "place", "label": "p_A"},
                    {"id": "p_B", "type": "place", "label": "p_B"},
                ],
                "edges": [
                    {"source": "p_start", "target": "t_split"},
                    {"source": "t_split", "target": "p_A"},
                    {"source": "t_split", "target": "p_B"},
                ],
            },
            "transitions": {"split": {"input_places": ["p_start"]}},
            "attribute_catalog": {},
            "metadata": {"start_place": "p_start", "end_place": "p_B"},
        }
        result = PartialTraceReplayer().replay(_xes([{"concept:name": "split"}]), data)
        assert set(result["init_places"]) == {"p_A", "p_B"}

    def test_and_join_consumes_multiple_tokens(self):
        data = {
            "graph": {
                "nodes": [
                    {"id": "p_start", "type": "place", "label": "p_start"},
                    {"id": "t_split", "type": "transition", "label": "split"},
                    {"id": "p_A", "type": "place", "label": "p_A"},
                    {"id": "p_B", "type": "place", "label": "p_B"},
                    {"id": "t_join", "type": "transition", "label": "join"},
                    {"id": "p_end", "type": "place", "label": "p_end"},
                ],
                "edges": [
                    {"source": "p_start", "target": "t_split"},
                    {"source": "t_split", "target": "p_A"},
                    {"source": "t_split", "target": "p_B"},
                    {"source": "p_A", "target": "t_join"},
                    {"source": "p_B", "target": "t_join"},
                    {"source": "t_join", "target": "p_end"},
                ],
            },
            "transitions": {
                "split": {"input_places": ["p_start"]},
                "join": {"input_places": ["p_A", "p_B"]},
            },
            "attribute_catalog": {},
            "metadata": {"start_place": "p_start", "end_place": "p_end"},
        }
        result = PartialTraceReplayer().replay(
            _xes([{"concept:name": "split"}, {"concept:name": "join"}]), data
        )
        assert result["init_places"] == ["p_end"]

    def test_unknown_activity_raises_error(self):
        data = _make_data(["register"])
        with pytest.raises(PartialTraceError, match="not found in the Petri net"):
            PartialTraceReplayer().replay(
                _xes([{"concept:name": "unknown_activity"}]), data
            )

    def test_stuck_replay_raises_error(self):
        data = _make_data(["register", "admit"])
        # "admit" expects p_1 but token is at p_start
        with pytest.raises(PartialTraceError, match="Replay stuck at 'admit'"):
            PartialTraceReplayer().replay(
                _xes([{"concept:name": "admit"}]), data
            )

    def test_multiple_traces_raises_error(self):
        data = _make_data(["register"])
        with pytest.raises(PartialTraceError, match="single-trace"):
            PartialTraceReplayer().replay(_xes_two_traces(), data)

    def test_invalid_xes_raises_error(self):
        data = _make_data(["register"])
        with pytest.raises(PartialTraceError, match="Invalid XES"):
            PartialTraceReplayer().replay(b"this is not xml", data)


# ---------------------------------------------------------------------------
# Tests — attribute pre-processing
# ---------------------------------------------------------------------------

class TestAttributePreprocessing:

    def test_categorical_attribute_extracted(self):
        catalog = {
            "status": {"type": "categorical", "possible_values": ["admitted", "discharged"], "bin_boundaries": []},
        }
        data = _make_data(["register"], catalog=catalog)
        result = PartialTraceReplayer().replay(
            _xes([{"concept:name": "register", "status": "admitted"}]), data
        )
        assert {"attribute": "status", "value": "admitted"} in result["init_effects"]

    def test_boolean_true_variants_normalized(self):
        catalog = {"critical": {"type": "boolean", "possible_values": ["true", "false"], "bin_boundaries": []}}
        data = _make_data(["register"], catalog=catalog)
        for raw in ("True", "1", "yes", "true"):
            result = PartialTraceReplayer().replay(
                _xes([{"concept:name": "register", "critical": raw}]), data
            )
            assert {"attribute": "critical", "value": "true"} in result["init_effects"]

    def test_boolean_false_normalized(self):
        catalog = {"critical": {"type": "boolean", "possible_values": ["true", "false"], "bin_boundaries": []}}
        data = _make_data(["register"], catalog=catalog)
        result = PartialTraceReplayer().replay(
            _xes([{"concept:name": "register", "critical": "false"}]), data
        )
        assert {"attribute": "critical", "value": "false"} in result["init_effects"]

    def test_numerical_value_discretized_correctly(self):
        catalog = {
            "crp": {
                "type": "numerical",
                "possible_values": ["lte_50_0", "gte_50_0"],
                "bin_boundaries": [50.0],
            }
        }
        data = _make_data(["register"], catalog=catalog)
        result = PartialTraceReplayer().replay(
            _xes([{"concept:name": "register", "crp": 30.0}]), data
        )
        effects = {e["attribute"]: e["value"] for e in result["init_effects"]}
        assert effects["crp"] == "lte_50_0"

    def test_last_event_attribute_wins(self):
        catalog = {"status": {"type": "categorical", "possible_values": ["admitted", "discharged"], "bin_boundaries": []}}
        data = _make_data(["register", "admit"], catalog=catalog)
        result = PartialTraceReplayer().replay(
            _xes([
                {"concept:name": "register", "status": "admitted"},
                {"concept:name": "admit", "status": "discharged"},
            ]),
            data,
        )
        effects = {e["attribute"]: e["value"] for e in result["init_effects"]}
        assert effects["status"] == "discharged"

    def test_unknown_attribute_produces_warning_not_error(self):
        data = _make_data(["register"], catalog={})
        result = PartialTraceReplayer().replay(
            _xes([{"concept:name": "register", "ward": "icu"}]), data
        )
        assert result["init_effects"] == []
        assert any("not in attribute catalog" in w for w in result["warnings"])

    def test_categorical_value_not_in_possible_values_produces_warning(self):
        catalog = {"status": {"type": "categorical", "possible_values": ["admitted"], "bin_boundaries": []}}
        data = _make_data(["register"], catalog=catalog)
        result = PartialTraceReplayer().replay(
            _xes([{"concept:name": "register", "status": "unknown_value"}]), data
        )
        assert result["init_effects"] == []
        assert any("not a known category" in w for w in result["warnings"])

    def test_numerical_no_boundaries_produces_warning(self):
        catalog = {"crp": {"type": "numerical", "possible_values": ["lte_50_0"], "bin_boundaries": []}}
        data = _make_data(["register"], catalog=catalog)
        result = PartialTraceReplayer().replay(
            _xes([{"concept:name": "register", "crp": 30.0}]), data
        )
        assert result["init_effects"] == []
        assert any("no bin boundaries" in w for w in result["warnings"])

    def test_empty_trace_warning_emitted(self):
        data = _make_data(["register"])
        result = PartialTraceReplayer().replay(_xes_empty_trace(), data)
        assert any("0 events" in w for w in result["warnings"])
