"""Tests for encoding.effect_encoder standalone helpers."""
import pytest

from models import AttributeCatalogEntry
from encoding.effect_encoder import boolean_effect, categorical_effect, value_to_pddl_effects
from encoding.pddl_model import PDDLEffect


class TestBooleanEffect:

    def test_true_value_no_negation(self):
        result = boolean_effect("urgent", True, set())
        assert result == [PDDLEffect.set_attr_true("urgent")]

    def test_true_value_with_negation(self):
        result = boolean_effect("urgent", True, {"urgent"})
        assert PDDLEffect.set_attr_true("urgent") in result
        assert PDDLEffect.clear_attr_false("urgent") in result

    def test_false_value_no_negation(self):
        result = boolean_effect("urgent", False, set())
        assert result == [PDDLEffect.clear_attr_true("urgent")]

    def test_false_value_with_negation(self):
        result = boolean_effect("urgent", False, {"urgent"})
        assert PDDLEffect.set_attr_false("urgent") in result
        assert PDDLEffect.clear_attr_true("urgent") in result

    def test_string_true(self):
        result = boolean_effect("admitted", "True", set())
        assert result == [PDDLEffect.set_attr_true("admitted")]


class TestCategoricalEffect:

    def test_set_value_clears_others(self):
        entry = AttributeCatalogEntry("categorical", {"flu", "cold"})
        result = categorical_effect("diagnosis", "flu", entry, set())
        assert PDDLEffect.set_attr_is("diagnosis", "flu") in result
        assert PDDLEffect.clear_attr_is("diagnosis", "cold") in result

    def test_negated_attribute_emits_is_not(self):
        entry = AttributeCatalogEntry("categorical", {"flu", "cold"})
        result = categorical_effect("diagnosis", "flu", entry, {"diagnosis"})
        assert PDDLEffect.set_attr_is_not("diagnosis", "cold") in result
        assert PDDLEffect.clear_attr_is_not("diagnosis", "flu") in result

    def test_single_value(self):
        entry = AttributeCatalogEntry("categorical", {"flu"})
        result = categorical_effect("diagnosis", "flu", entry, set())
        assert result == [PDDLEffect.set_attr_is("diagnosis", "flu")]


class TestValueToPddlEffects:

    def test_dispatches_boolean(self):
        entry = AttributeCatalogEntry("boolean", set())
        result = value_to_pddl_effects("urgent", True, entry, set())
        assert PDDLEffect.set_attr_true("urgent") in result

    def test_dispatches_categorical(self):
        entry = AttributeCatalogEntry("categorical", {"flu", "cold"})
        result = value_to_pddl_effects("diagnosis", "flu", entry, set())
        assert PDDLEffect.set_attr_is("diagnosis", "flu") in result

    def test_dispatches_numerical(self):
        entry = AttributeCatalogEntry("numerical", {"lte_10_0", "gte_10_0"})
        result = value_to_pddl_effects("crp", "lte_10_0", entry, set())
        assert PDDLEffect.set_attr_is("crp", "lte_10_0") in result
        assert PDDLEffect.clear_attr_is("crp", "gte_10_0") in result
