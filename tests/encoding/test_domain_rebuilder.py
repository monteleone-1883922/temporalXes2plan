"""Tests for encoding.domain_rebuilder.DomainRebuilder."""
import math
import pytest

from encoding.domain_rebuilder import DomainRebuilder
from encoding.pddl_model import PDDLAction, PDDLDurativeAction


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _node(id_, type_, label):
    return {"id": id_, "type": type_, "label": label}


def _edge(src, tgt):
    return {"source": src, "target": tgt}


@pytest.fixture
def minimal_data():
    """p_start → register → p_end.  No effects, no XOR, no tau."""
    return {
        "graph": {
            "nodes": [
                _node("n1", "place", "p_start"),
                _node("n2", "transition", "register"),
                _node("n3", "place", "p_end"),
            ],
            "edges": [_edge("n1", "n2"), _edge("n2", "n3")],
        },
        "transitions": {
            "register": {
                "activity_name": "register",
                "input_places": ["p_start"],
                "preconditions": [],
                "effect_groups": [],
                "cost": 0.0,
                "duration": None,
            }
        },
        "xor_splits": {},
        "attribute_catalog": {},
    }


@pytest.fixture
def data_with_categorical_effect():
    """register changes 'status' (categorical) to 'admitted'."""
    return {
        "graph": {
            "nodes": [
                _node("n1", "place", "p_start"),
                _node("n2", "transition", "register"),
                _node("n3", "place", "p_end"),
            ],
            "edges": [_edge("n1", "n2"), _edge("n2", "n3")],
        },
        "transitions": {
            "register": {
                "activity_name": "register",
                "input_places": ["p_start"],
                "preconditions": [],
                "effect_groups": [
                    {
                        "guard": [],
                        "probability": 1.0,
                        "assignments": [{"attribute": "status", "value": "admitted"}],
                    }
                ],
                "cost": 0.0,
                "duration": None,
            }
        },
        "xor_splits": {},
        "attribute_catalog": {
            "status": {"type": "categorical", "possible_values": ["admitted", "discharged"]},
        },
    }


@pytest.fixture
def data_with_guarded_effect_groups():
    """register has two guarded effect groups (OR in guard → 2 action slots)."""
    return {
        "graph": {
            "nodes": [
                _node("n1", "place", "p_start"),
                _node("n2", "transition", "register"),
                _node("n3", "place", "p_end"),
            ],
            "edges": [_edge("n1", "n2"), _edge("n2", "n3")],
        },
        "transitions": {
            "register": {
                "activity_name": "register",
                "input_places": ["p_start"],
                "preconditions": [],
                "effect_groups": [
                    {
                        "guard": [
                            [{"attribute": "status", "predicate": "=", "value": "admitted"}],
                            [{"attribute": "status", "predicate": "=", "value": "discharged"}],
                        ],
                        "probability": 0.8,
                        "assignments": [{"attribute": "status", "value": "treated"}],
                    }
                ],
                "cost": 0.0,
                "duration": None,
            }
        },
        "xor_splits": {},
        "attribute_catalog": {
            "status": {"type": "categorical", "possible_values": ["admitted", "discharged", "treated"]},
        },
    }


@pytest.fixture
def data_with_xor_split():
    """register → p_xor → {admit, discharge}.  XOR with no conditions (level 2)."""
    return {
        "graph": {
            "nodes": [
                _node("n1", "place", "p_start"),
                _node("n2", "transition", "register"),
                _node("n3", "xor_split", "p_xor"),
                _node("n4", "transition", "admit"),
                _node("n5", "transition", "discharge"),
                _node("n6", "place", "p_admitted"),
                _node("n7", "place", "p_discharged"),
            ],
            "edges": [
                _edge("n1", "n2"), _edge("n2", "n3"),
                _edge("n3", "n4"), _edge("n4", "n6"),
                _edge("n3", "n5"), _edge("n5", "n7"),
            ],
        },
        "transitions": {
            "register": {
                "activity_name": "register",
                "input_places": ["p_start"],
                "preconditions": [],
                "effect_groups": [],
                "cost": 0.0,
                "duration": None,
            },
            "admit": {
                "activity_name": "admit",
                "input_places": ["p_xor"],
                "preconditions": [],
                "effect_groups": [],
                "cost": 0.0,
                "duration": None,
            },
            "discharge": {
                "activity_name": "discharge",
                "input_places": ["p_xor"],
                "preconditions": [],
                "effect_groups": [],
                "cost": 0.0,
                "duration": None,
            },
        },
        "xor_splits": {
            "p_xor": {
                "branches": {
                    "admit": {
                        "activity_name": "admit",
                        "conditions": [],
                        "probability": 0.7,
                    },
                    "discharge": {
                        "activity_name": "discharge",
                        "conditions": [],
                        "probability": 0.3,
                    },
                }
            }
        },
        "attribute_catalog": {},
    }


@pytest.fixture
def data_with_tau():
    """p_start → tau_0 (silent) → p_mid → register → p_end."""
    return {
        "graph": {
            "nodes": [
                _node("n1", "place", "p_start"),
                _node("n2", "silent", "tau_0"),
                _node("n3", "place", "p_mid"),
                _node("n4", "transition", "register"),
                _node("n5", "place", "p_end"),
            ],
            "edges": [
                _edge("n1", "n2"), _edge("n2", "n3"),
                _edge("n3", "n4"), _edge("n4", "n5"),
            ],
        },
        "transitions": {
            "register": {
                "activity_name": "register",
                "input_places": ["p_mid"],
                "preconditions": [],
                "effect_groups": [],
                "cost": 0.0,
                "duration": None,
            }
        },
        "xor_splits": {},
        "attribute_catalog": {},
    }


@pytest.fixture
def data_with_duration():
    """register has duration data (10–30s)."""
    return {
        "graph": {
            "nodes": [
                _node("n1", "place", "p_start"),
                _node("n2", "transition", "register"),
                _node("n3", "place", "p_end"),
            ],
            "edges": [_edge("n1", "n2"), _edge("n2", "n3")],
        },
        "transitions": {
            "register": {
                "activity_name": "register",
                "input_places": ["p_start"],
                "preconditions": [],
                "effect_groups": [],
                "cost": 0.0,
                "duration": {"effective_min": 10.0, "effective_max": 30.0},
            }
        },
        "xor_splits": {},
        "attribute_catalog": {},
    }


# ---------------------------------------------------------------------------
# Basic rebuild
# ---------------------------------------------------------------------------

class TestRebuildBasic:
    def test_returns_domain(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        assert domain is not None

    def test_domain_name(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data, domain_name="test_domain")
        assert domain.name == "test_domain"

    def test_strips_requirement(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        assert ":strips" in domain.requirements

    def test_typing_requirement(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        assert ":typing" in domain.requirements

    def test_one_action_for_simple_transition(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        action_names = [a.name for a in domain.actions]
        assert "execute_register" in action_names

    def test_action_has_input_place_as_precondition(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        action = next(a for a in domain.actions if a.name == "execute_register")
        prec_attrs = {c.attribute for c in action.preconditions}
        assert "p_start" in prec_attrs

    def test_action_clears_input_place(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        action = next(a for a in domain.actions if a.name == "execute_register")
        clear_attrs = {e.attribute for e in action.effects if e.clear}
        assert "p_start" in clear_attrs

    def test_action_marks_transition_itself(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        action = next(a for a in domain.actions if a.name == "execute_register")
        mark_attrs = {e.attribute for e in action.effects if not e.clear}
        assert "register" in mark_attrs
        assert "p_end" not in mark_attrs

    def test_place_marking_action_created(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        names = {a.name for a in domain.actions}
        assert "mark_places_from_register" in names

    def test_place_marking_action_consumes_transition_and_marks_output(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        mark_action = next(a for a in domain.actions if a.name == "mark_places_from_register")
        clear_attrs = {e.attribute for e in mark_action.effects if e.clear}
        mark_attrs = {e.attribute for e in mark_action.effects if not e.clear}
        assert "register" in clear_attrs
        assert "p_end" in mark_attrs

    def test_place_constants_present(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        constant_names = {c.name for c in domain.constants}
        assert "p_start" in constant_names
        assert "p_end" in constant_names

    def test_transition_constants_present(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        constant_names = {c.name for c in domain.constants}
        assert "register" in constant_names


# ---------------------------------------------------------------------------
# Categorical effects
# ---------------------------------------------------------------------------

class TestCategoricalEffect:
    def test_effect_sets_attribute(self, data_with_categorical_effect):
        domain = DomainRebuilder().rebuild(data_with_categorical_effect)
        action = next(a for a in domain.actions if a.name == "execute_register")
        effect_attrs = {e.attribute for e in action.effects}
        assert "status" in effect_attrs

    def test_attribute_type_in_domain(self, data_with_categorical_effect):
        domain = DomainRebuilder().rebuild(data_with_categorical_effect)
        type_names = {t.name for t in domain.types}
        assert "status_val" in type_names

    def test_attribute_values_as_constants(self, data_with_categorical_effect):
        domain = DomainRebuilder().rebuild(data_with_categorical_effect)
        constant_names = {c.name for c in domain.constants}
        assert "admitted" in constant_names
        assert "discharged" in constant_names

    def test_unguarded_group_no_cost(self, data_with_categorical_effect):
        domain = DomainRebuilder().rebuild(data_with_categorical_effect)
        action = next(a for a in domain.actions if a.name == "execute_register")
        assert action.base_cost is None


# ---------------------------------------------------------------------------
# Guarded effect groups — Asse C
# ---------------------------------------------------------------------------

class TestGuardedEffectGroups:
    def test_guard_with_two_or_clauses_produces_two_actions(self, data_with_guarded_effect_groups):
        domain = DomainRebuilder().rebuild(data_with_guarded_effect_groups)
        action_names = [a.name for a in domain.actions]
        assert "execute_register_v0" in action_names
        assert "execute_register_v1" in action_names

    def test_guard_clause_becomes_precondition(self, data_with_guarded_effect_groups):
        domain = DomainRebuilder().rebuild(data_with_guarded_effect_groups)
        v0 = next(a for a in domain.actions if a.name == "execute_register_v0")
        prec_attrs = {c.attribute for c in v0.preconditions}
        assert "status" in prec_attrs

    def test_two_variants_have_different_guard_preconditions(self, data_with_guarded_effect_groups):
        domain = DomainRebuilder().rebuild(data_with_guarded_effect_groups)
        v0 = next(a for a in domain.actions if a.name == "execute_register_v0")
        v1 = next(a for a in domain.actions if a.name == "execute_register_v1")
        # Both have status in precs but different values
        v0_status = {c.value for c in v0.preconditions if c.attribute == "status"}
        v1_status = {c.value for c in v1.preconditions if c.attribute == "status"}
        assert v0_status != v1_status


# ---------------------------------------------------------------------------
# XOR splits
# ---------------------------------------------------------------------------

class TestXorSplits:
    def test_xor_branch_action_created(self, data_with_xor_split):
        domain = DomainRebuilder().rebuild(data_with_xor_split)
        names = {a.name for a in domain.actions}
        assert "execute_admit" in names
        assert "execute_discharge" in names

    def test_xor_branch_without_conditions_adds_log_prob_cost(self, data_with_xor_split):
        domain = DomainRebuilder().rebuild(data_with_xor_split)
        admit = next(a for a in domain.actions if a.name == "execute_admit")
        expected_cost = -math.log(0.7)
        assert admit.base_cost is not None
        assert abs(admit.base_cost - expected_cost) < 1e-9


# ---------------------------------------------------------------------------
# Tau (silent) transitions
# ---------------------------------------------------------------------------

class TestTauActions:
    def test_tau_action_created(self, data_with_tau):
        domain = DomainRebuilder().rebuild(data_with_tau)
        names = {a.name for a in domain.actions}
        assert "execute_tau_0" in names

    def test_tau_action_has_input_place_precondition(self, data_with_tau):
        domain = DomainRebuilder().rebuild(data_with_tau)
        tau = next(a for a in domain.actions if a.name == "execute_tau_0")
        prec_attrs = {c.attribute for c in tau.preconditions}
        assert "p_start" in prec_attrs

    def test_tau_action_marks_itself(self, data_with_tau):
        domain = DomainRebuilder().rebuild(data_with_tau)
        tau = next(a for a in domain.actions if a.name == "execute_tau_0")
        mark_attrs = {e.attribute for e in tau.effects if not e.clear}
        assert "tau_0" in mark_attrs
        assert "p_mid" not in mark_attrs

    def test_tau_place_marking_action_created(self, data_with_tau):
        domain = DomainRebuilder().rebuild(data_with_tau)
        names = {a.name for a in domain.actions}
        assert "mark_places_from_tau_0" in names

    def test_tau_place_marking_action_consumes_tau_and_marks_output(self, data_with_tau):
        domain = DomainRebuilder().rebuild(data_with_tau)
        mark_action = next(a for a in domain.actions if a.name == "mark_places_from_tau_0")
        clear_attrs = {e.attribute for e in mark_action.effects if e.clear}
        mark_attrs = {e.attribute for e in mark_action.effects if not e.clear}
        assert "tau_0" in clear_attrs
        assert "p_mid" in mark_attrs

    def test_tau_action_is_instantaneous(self, data_with_tau):
        domain = DomainRebuilder().rebuild(data_with_tau, use_durative=True)
        tau = next(a for a in domain.actions if a.name == "execute_tau_0")
        assert isinstance(tau, PDDLAction)


# ---------------------------------------------------------------------------
# Durative actions
# ---------------------------------------------------------------------------

class TestDurativeActions:
    def test_no_durative_requirement_when_flag_false(self, data_with_duration):
        domain = DomainRebuilder().rebuild(data_with_duration, use_durative=False)
        assert ":durative-actions" not in domain.requirements

    def test_durative_requirement_when_flag_true(self, data_with_duration):
        domain = DomainRebuilder().rebuild(data_with_duration, use_durative=True)
        assert ":durative-actions" in domain.requirements

    def test_durative_action_instance(self, data_with_duration):
        domain = DomainRebuilder().rebuild(data_with_duration, use_durative=True)
        action = next(a for a in domain.actions if a.name == "execute_register")
        assert isinstance(action, PDDLDurativeAction)

    def test_durative_action_duration_bounds(self, data_with_duration):
        domain = DomainRebuilder().rebuild(data_with_duration, use_durative=True)
        action = next(a for a in domain.actions if a.name == "execute_register")
        assert isinstance(action, PDDLDurativeAction)
        assert action.duration_min == 10.0
        assert action.duration_max == 30.0

    def test_no_duration_data_stays_instantaneous(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data, use_durative=True)
        action = next(a for a in domain.actions if a.name == "execute_register")
        assert isinstance(action, PDDLAction)


# ---------------------------------------------------------------------------
# has_costs
# ---------------------------------------------------------------------------

class TestHasCosts:

    def test_has_costs_false_when_no_action_has_cost(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        assert domain.has_costs is False

    def test_no_action_costs_requirement_when_no_cost(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data)
        assert ":action-costs" not in domain.requirements
        assert ":numeric-fluents" not in domain.requirements

    def test_has_costs_true_when_action_has_cost(self, minimal_data):
        minimal_data["transitions"]["register"]["cost"] = 1.5
        domain = DomainRebuilder().rebuild(minimal_data, use_costs=True)
        assert domain.has_costs is True

    def test_action_costs_requirements_added_when_has_costs(self, minimal_data):
        minimal_data["transitions"]["register"]["cost"] = 0.693
        domain = DomainRebuilder().rebuild(minimal_data, use_costs=True)
        assert ":action-costs" in domain.requirements
        assert ":numeric-fluents" in domain.requirements

    def test_has_costs_true_even_without_explicit_action_cost(self, minimal_data):
        domain = DomainRebuilder().rebuild(minimal_data, use_costs=True)
        assert domain.has_costs is True


# ---------------------------------------------------------------------------
# has_deadline
# ---------------------------------------------------------------------------

class TestHasDeadline:

    def test_no_deadline_by_default(self, data_with_duration):
        domain = DomainRebuilder().rebuild(data_with_duration, use_durative=True)
        assert domain.has_deadline is False

    def test_deadline_sets_has_deadline(self, data_with_duration):
        domain = DomainRebuilder().rebuild(data_with_duration, use_durative=True, has_deadline=True)
        assert domain.has_deadline is True

    def test_deadline_ignored_when_not_durative(self, data_with_duration):
        domain = DomainRebuilder().rebuild(data_with_duration, use_durative=False, has_deadline=True)
        assert domain.has_deadline is False

    def test_deadline_ignored_when_no_durative_actions(self, minimal_data):
        # minimal_data has no duration → actions are instantaneous even with use_durative=True
        domain = DomainRebuilder().rebuild(minimal_data, use_durative=True, has_deadline=True)
        assert domain.has_deadline is False
