"""Tests for encoding.domain_builder.DomainBuilder."""
import pytest

from encoding.domain_builder import DomainBuilder


class TestBuildTypes:

    def test_always_has_petri_element_hierarchy(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        type_map = {t.name: t.parent for t in domain.types}

        assert "petri_element" in type_map
        assert type_map["place"] == "petri_element"
        assert type_map["transition"] == "petri_element"

    def test_categorical_attribute_generates_value_type(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        type_names = {t.name for t in domain.types}

        assert "diagnosis_val" in type_names

    def test_numerical_attribute_generates_value_type(self, simple_parse_result):
        from models import AttributeCatalogEntry
        simple_parse_result.attribute_catalog["crp"] = AttributeCatalogEntry(
            attribute_type="numerical",
            possible_values={"lte_10_0", "gte_10_0"},
        )
        domain = DomainBuilder().build(simple_parse_result)
        type_names = {t.name for t in domain.types}

        assert "crp_val" in type_names

    def test_boolean_attribute_no_value_type(self, simple_parse_result):
        from models import AttributeCatalogEntry
        simple_parse_result.attribute_catalog["urgent"] = AttributeCatalogEntry(
            attribute_type="boolean",
            possible_values=set(),
        )
        domain = DomainBuilder().build(simple_parse_result)
        type_names = {t.name for t in domain.types}

        assert "urgent_val" not in type_names

    def test_empty_catalog_only_petri_types(self, tau_parse_result):
        domain = DomainBuilder().build(tau_parse_result)
        type_names = {t.name for t in domain.types}

        assert type_names == {"petri_element", "place", "transition"}


class TestBuildConstants:

    def test_places_as_constants(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        place_consts = {c.name for c in domain.constants if c.type_name == "place"}

        assert place_consts == {"p_start", "p_mid", "p_end"}

    def test_transitions_as_constants(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        trans_consts = {c.name for c in domain.constants if c.type_name == "transition"}

        assert trans_consts == {"activity_a", "activity_b"}

    def test_tau_transitions_as_constants(self, tau_parse_result):
        domain = DomainBuilder().build(tau_parse_result)
        trans_consts = {c.name for c in domain.constants if c.type_name == "transition"}

        assert "tau_0" in trans_consts
        assert "activity_a" in trans_consts

    def test_categorical_value_objects(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        diag_vals = {
            c.name for c in domain.constants if c.type_name == "diagnosis_val"
        }

        assert diag_vals == {"flu", "cold", "covid"}

    def test_no_value_objects_for_boolean(self, simple_parse_result):
        from models import AttributeCatalogEntry
        simple_parse_result.attribute_catalog["urgent"] = AttributeCatalogEntry(
            attribute_type="boolean",
            possible_values=set(),
        )
        domain = DomainBuilder().build(simple_parse_result)
        urgent_vals = {
            c.name for c in domain.constants if c.type_name == "urgent_val"
        }

        assert urgent_vals == set()


class TestBuildPredicates:

    def test_marked_predicate_exists(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        marked = [p for p in domain.predicates if p.name == "marked"]

        assert len(marked) == 1
        assert marked[0].parameters == [("?x", "petri_element")]

    def test_categorical_positive_predicate_always_present(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        pred_names = {p.name for p in domain.predicates}

        assert "diagnosis_is" in pred_names

    def test_categorical_negative_predicate_absent_when_not_negated(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        pred_names = {p.name for p in domain.predicates}

        assert "diagnosis_is_not" not in pred_names

    def test_categorical_negative_predicate_present_when_negated(self, negated_parse_result):
        domain = DomainBuilder().build(negated_parse_result)
        pred_names = {p.name for p in domain.predicates}

        assert "diagnosis_is" in pred_names
        assert "diagnosis_is_not" in pred_names

    def test_categorical_predicate_has_typed_parameter(self, negated_parse_result):
        domain = DomainBuilder().build(negated_parse_result)
        diag_is = next(p for p in domain.predicates if p.name == "diagnosis_is")
        diag_is_not = next(p for p in domain.predicates if p.name == "diagnosis_is_not")

        assert diag_is.parameters == [("?v", "diagnosis_val")]
        assert diag_is_not.parameters == [("?v", "diagnosis_val")]

    def test_boolean_false_predicate_absent_when_not_negated(self, simple_parse_result):
        from models import AttributeCatalogEntry
        simple_parse_result.attribute_catalog["urgent"] = AttributeCatalogEntry(
            attribute_type="boolean", possible_values=set(),
        )
        domain = DomainBuilder().build(simple_parse_result)
        pred_names = {p.name for p in domain.predicates}

        assert "urgent_true" in pred_names
        assert "urgent_false" not in pred_names

    def test_boolean_false_predicate_present_when_negated(self, simple_parse_result):
        from models import AttributeCatalogEntry
        simple_parse_result.attribute_catalog["urgent"] = AttributeCatalogEntry(
            attribute_type="boolean", possible_values=set(),
        )
        simple_parse_result.negated_attributes = {"urgent"}
        domain = DomainBuilder().build(simple_parse_result)
        pred_names = {p.name for p in domain.predicates}

        urgent_true = next(p for p in domain.predicates if p.name == "urgent_true")
        urgent_false = next(p for p in domain.predicates if p.name == "urgent_false")
        assert urgent_true.parameters == []
        assert urgent_false.parameters == []

    def test_mixed_catalog_no_negated(self, simple_parse_result, catalog_mixed):
        simple_parse_result.attribute_catalog = catalog_mixed
        simple_parse_result.negated_attributes = set()
        domain = DomainBuilder().build(simple_parse_result)
        pred_names = {p.name for p in domain.predicates}

        assert "diagnosis_is" in pred_names
        assert "crp_is" in pred_names
        assert "urgent_true" in pred_names
        assert "diagnosis_is_not" not in pred_names
        assert "crp_is_not" not in pred_names
        assert "urgent_false" not in pred_names

    def test_mixed_catalog_all_negated(self, simple_parse_result, catalog_mixed):
        simple_parse_result.attribute_catalog = catalog_mixed
        simple_parse_result.negated_attributes = {"diagnosis", "crp", "urgent"}
        domain = DomainBuilder().build(simple_parse_result)
        pred_names = {p.name for p in domain.predicates}

        expected = {
            "marked",
            "diagnosis_is", "diagnosis_is_not",
            "crp_is", "crp_is_not",
            "urgent_true", "urgent_false",
        }
        assert expected.issubset(pred_names)


class TestBuildDomain:

    def test_domain_name(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result, domain_name="my_process")
        assert domain.name == "my_process"

    def test_default_requirements(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        assert domain.requirements == [":strips", ":typing"]

    def test_actions_are_populated(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        assert len(domain.actions) > 0

    def test_action_count_sequence_net(self, simple_parse_result):
        domain = DomainBuilder().build(simple_parse_result)
        mark_actions = [a for a in domain.actions if a.name.startswith("mark_")]
        exec_actions = [a for a in domain.actions if a.name.startswith("execute_")]

        assert len(mark_actions) == 2
        assert len(exec_actions) == 2
