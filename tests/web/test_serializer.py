"""Tests for web.serializer.serialize_parse_result."""
from unittest.mock import MagicMock

from web.serializer import serialize_parse_result


def _make_parse_result(start_place: str = "p_start", end_place: str = "p_end"):
    pnm = MagicMock()
    pnm.xor_splits = {}
    pnm.petrinet.transitions = []
    pnm.petrinet.places = []
    pnm.petrinet.arcs = []
    pnm.trans_inputs = {}
    pnm.trans_outputs = {}
    pnm.silent_transitions = {}

    result = MagicMock()
    result.petri_net_model = pnm
    result.transitions = {}
    result.start_place = start_place
    result.end_place = end_place
    result.attribute_catalog = {}
    return result


class TestSerializeParseResultMetadata:
    def test_metadata_key_present(self):
        out = serialize_parse_result(_make_parse_result())
        assert "metadata" in out

    def test_metadata_contains_start_place(self):
        out = serialize_parse_result(_make_parse_result(start_place="p_start"))
        assert out["metadata"]["start_place"] == "p_start"

    def test_metadata_contains_end_place(self):
        out = serialize_parse_result(_make_parse_result(end_place="p_end"))
        assert out["metadata"]["end_place"] == "p_end"

    def test_metadata_values_match_parse_result(self):
        out = serialize_parse_result(_make_parse_result(start_place="p_s", end_place="p_e"))
        assert out["metadata"] == {"start_place": "p_s", "end_place": "p_e"}
