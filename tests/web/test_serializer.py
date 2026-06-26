"""Tests for web.serializer.serialize_parse_result."""
from unittest.mock import MagicMock

from models import AttributeCatalogEntry
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


class TestSerializeAttributeCatalog:

    def _result_with_catalog(self, catalog: dict):
        r = _make_parse_result()
        r.attribute_catalog = catalog
        return r

    def test_numerical_attribute_includes_bin_boundaries(self):
        entry = AttributeCatalogEntry(
            attribute_type="numerical",
            possible_values={"lte_50_0", "gte_50_0"},
            bin_boundaries=[50.0],
        )
        out = serialize_parse_result(self._result_with_catalog({"crp": entry}))
        assert out["attribute_catalog"]["crp"]["bin_boundaries"] == [50.0]

    def test_non_numerical_attribute_has_empty_bin_boundaries(self):
        entry = AttributeCatalogEntry(
            attribute_type="categorical",
            possible_values={"admitted", "discharged"},
            bin_boundaries=None,
        )
        out = serialize_parse_result(self._result_with_catalog({"status": entry}))
        assert out["attribute_catalog"]["status"]["bin_boundaries"] == []

    def test_boolean_attribute_has_empty_bin_boundaries(self):
        entry = AttributeCatalogEntry(
            attribute_type="boolean",
            possible_values=set(),
            bin_boundaries=None,
        )
        out = serialize_parse_result(self._result_with_catalog({"critical": entry}))
        assert out["attribute_catalog"]["critical"]["bin_boundaries"] == []

    def test_bin_boundaries_present_in_catalog_key(self):
        entry = AttributeCatalogEntry(
            attribute_type="numerical",
            possible_values={"lte_10_0", "gte_10_0"},
            bin_boundaries=[10.0],
        )
        out = serialize_parse_result(self._result_with_catalog({"score": entry}))
        assert "bin_boundaries" in out["attribute_catalog"]["score"]
