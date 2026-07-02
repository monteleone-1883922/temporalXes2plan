"""Tests for web.serializer.serialize_parse_result."""
from unittest.mock import MagicMock

import pytest

from models import AttributeCatalogEntry, EffectInfo
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


# ---------------------------------------------------------------------------
# Transitions — output_places field (Step 0c)
# ---------------------------------------------------------------------------

def _make_transition_info(input_places=None, output_label=None):
    """Return a minimal TransitionInfo mock."""
    t_info = MagicMock()
    t_info.input_places = input_places or []
    t_info.effects = {}
    t_info.effect_groups = []
    t_info.duration = None
    t_info.total_firings = 0
    t_info.related_effects = []
    t_info.incompatible_effects = []
    t_info.xor_branch = None
    return t_info


def _make_transition(label: str, name: str):
    t = MagicMock()
    t.label = label
    t.name = name
    return t


def _make_place(name: str):
    p = MagicMock()
    p.name = name
    return p


class TestTransitionOutputPlaces:
    """Verify that serialize_parse_result includes output_places for each transition."""

    def _make_result_with_transition(self, act_name: str, output_place_names):
        pnm = MagicMock()
        pnm.xor_splits = {}
        pnm.petrinet.places = []
        pnm.petrinet.arcs = []
        pnm.trans_inputs = {}
        pnm.silent_transitions = {}

        t = _make_transition(label=act_name, name=f"t_{act_name}")
        places = [_make_place(n) for n in output_place_names]

        pnm.petrinet.transitions = [t]
        pnm.trans_outputs = {t: set(places)}

        result = MagicMock()
        result.petri_net_model = pnm
        result.start_place = "p_start"
        result.end_place = "p_end"
        result.attribute_catalog = {}
        result.transitions = {act_name: _make_transition_info()}
        return result

    def test_output_places_key_present(self):
        r = self._make_result_with_transition("act_a", ["p1"])
        out = serialize_parse_result(r)
        assert "output_places" in out["transitions"]["act_a"]

    def test_output_places_is_list(self):
        r = self._make_result_with_transition("act_a", ["p1"])
        out = serialize_parse_result(r)
        assert isinstance(out["transitions"]["act_a"]["output_places"], list)

    def test_output_places_contains_correct_place(self):
        r = self._make_result_with_transition("act_a", ["p1"])
        out = serialize_parse_result(r)
        assert "p1" in out["transitions"]["act_a"]["output_places"]

    def test_output_places_sorted(self):
        r = self._make_result_with_transition("act_a", ["p_z", "p_a", "p_m"])
        out = serialize_parse_result(r)
        places = out["transitions"]["act_a"]["output_places"]
        assert places == sorted(places)

    def test_output_places_empty_when_no_outputs(self):
        result = _make_parse_result()
        t_info = _make_transition_info()
        result.transitions = {"act_b": t_info}
        out = serialize_parse_result(result)
        assert out["transitions"]["act_b"]["output_places"] == []

    def test_input_places_still_present(self):
        r = self._make_result_with_transition("act_a", ["p_out"])
        out = serialize_parse_result(r)
        assert "input_places" in out["transitions"]["act_a"]
