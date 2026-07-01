"""Unit tests for evaluation.plan_validator."""
import pytest

from evaluation.plan_validator import (
    ValidationReport,
    _normalise_token_flow,
    build_variant_effects_lookup,
    resolve_action_to_transition,
    validate_plan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition(
    input_places=None,
    output_places=None,
    preconditions=None,
):
    return {
        "input_places": input_places or [],
        "output_places": output_places or [],
        "preconditions": preconditions or [],
    }


def _make_serialized(transitions: dict) -> dict:
    return {
        "transitions": transitions,
        "attribute_catalog": {},
        "metadata": {"start_place": "p_start", "end_place": "p_end"},
        "graph": {"nodes": [], "edges": []},
    }


# ---------------------------------------------------------------------------
# resolve_action_to_transition — name normalisation
# ---------------------------------------------------------------------------

class TestResolveActionToTransition:
    def _serialized(self, name):
        return _make_serialized({name: _make_transition()})

    def test_direct_lookup(self):
        s = self._serialized("register")
        assert resolve_action_to_transition("register", s) is not None

    def test_resolve_strips_exec_prefix(self):
        s = self._serialized("register")
        assert resolve_action_to_transition("exec_register", s) is not None

    def test_resolve_strips_version_suffix(self):
        s = self._serialized("register")
        assert resolve_action_to_transition("register_v3", s) is not None

    def test_resolve_strips_detdup_suffix(self):
        s = self._serialized("register")
        assert resolve_action_to_transition("register_DETDUP", s) is not None

    def test_resolve_strips_combined_suffixes(self):
        s = self._serialized("register")
        assert resolve_action_to_transition("exec_register_DETDUP_v2", s) is not None

    def test_resolve_unknown_returns_none(self):
        s = self._serialized("register")
        assert resolve_action_to_transition("unknown_action", s) is None


# ---------------------------------------------------------------------------
# _normalise_token_flow
# ---------------------------------------------------------------------------

class TestNormaliseTokenFlow:
    def test_strips_exec_prefix(self):
        assert _normalise_token_flow("exec_register") == "register"

    def test_strips_version_suffix(self):
        assert _normalise_token_flow("register_v3") == "register"

    def test_strips_detdup_suffix(self):
        assert _normalise_token_flow("register_DETDUP") == "register"

    def test_strips_exec_and_version(self):
        assert _normalise_token_flow("exec_register_v2") == "register"

    def test_no_change_plain_name(self):
        assert _normalise_token_flow("register") == "register"


# ---------------------------------------------------------------------------
# validate_plan — valid cases
# ---------------------------------------------------------------------------

class TestValidPlan:
    def test_valid_plan_single_step(self):
        s = _make_serialized({
            "register": _make_transition(
                input_places=["p_start"],
                output_places=["p_end"],
            )
        })
        report = validate_plan(["register"], ["p_start"], s)
        assert report.valid is True
        assert report.error_step is None

    def test_valid_plan_multi_step(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
            "act_b": _make_transition(input_places=["p1"], output_places=["p2"]),
        })
        report = validate_plan(["act_a", "act_b"], ["p0"], s)
        assert report.valid is True
        assert "p2" in report.final_places

    def test_valid_plan_tokens_transferred_correctly(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        report = validate_plan(["act_a"], ["p0"], s)
        assert "p0" not in report.final_places
        assert "p1" in report.final_places

    def test_steps_executed_counts_non_silent(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        report = validate_plan(["tau_skip", "act_a", "tau_end"], ["p0"], s)
        assert report.steps_executed == 1


# ---------------------------------------------------------------------------
# validate_plan — failure cases
# ---------------------------------------------------------------------------

class TestInvalidPlan:
    def test_unknown_action_returns_invalid(self):
        s = _make_serialized({})
        report = validate_plan(["nonexistent"], ["p_start"], s)
        assert report.valid is False
        assert report.error_action == "nonexistent"

    def test_missing_token_returns_invalid(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p_missing"], output_places=["p1"]),
        })
        report = validate_plan(["act_a"], ["p0"], s)
        assert report.valid is False
        assert "act_a" == report.error_action

    def test_missing_precondition_returns_invalid_when_variant_effects_supplied(self):
        s = _make_serialized({
            "act_a": _make_transition(
                input_places=["p0"],
                output_places=["p1"],
                preconditions=[[{"attribute": "status", "predicate": "=", "value": "ready"}]],
            ),
        })
        report = validate_plan(["act_a"], ["p0"], s, variant_effects={})
        assert report.valid is False
        assert "act_a" == report.error_action

    def test_first_error_step_correct(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
            "act_b": _make_transition(input_places=["p_wrong"], output_places=["p2"]),
        })
        report = validate_plan(["act_a", "act_b"], ["p0"], s)
        assert report.valid is False
        assert report.error_step == 1
        assert report.error_action == "act_b"

    def test_error_reason_not_none_on_failure(self):
        s = _make_serialized({})
        report = validate_plan(["ghost"], ["p0"], s)
        assert report.error_reason is not None


# ---------------------------------------------------------------------------
# validate_plan — silent actions
# ---------------------------------------------------------------------------

class TestSilentActions:
    def test_tau_actions_skipped(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        report = validate_plan(["tau_start", "act_a", "tau_end"], ["p0"], s)
        assert report.valid is True

    def test_mark_actions_skipped(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        report = validate_plan(["mark_places_from_p0", "act_a"], ["p0"], s)
        assert report.valid is True

    def test_only_silent_actions_valid(self):
        s = _make_serialized({})
        report = validate_plan(["tau_skip", "mark_places_from_start"], ["p0"], s)
        assert report.valid is True

    def test_silent_actions_do_not_consume_tokens(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        report = validate_plan(["tau_noop", "act_a"], ["p0"], s)
        assert report.valid is True
        assert "p1" in report.final_places


# ---------------------------------------------------------------------------
# validate_plan — attribute_checked flag
# ---------------------------------------------------------------------------

class TestAttributeChecked:
    def test_attribute_checked_false_without_variant_effects(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        report = validate_plan(["act_a"], ["p0"], s)
        assert report.attribute_checked is False

    def test_attribute_checked_true_with_variant_effects(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        report = validate_plan(["act_a"], ["p0"], s, variant_effects={"act_a": {}})
        assert report.attribute_checked is True

    def test_attribute_checked_true_even_on_failure(self):
        s = _make_serialized({})
        report = validate_plan(["nonexistent"], ["p0"], s, variant_effects={})
        assert report.attribute_checked is True

    def test_attribute_checked_false_on_token_failure_without_variant_effects(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p_missing"], output_places=["p1"]),
        })
        report = validate_plan(["act_a"], ["p0"], s)
        assert report.attribute_checked is False


# ---------------------------------------------------------------------------
# validate_plan — preconditions satisfied
# ---------------------------------------------------------------------------

class TestPreconditions:
    def test_precondition_satisfied_by_init_attr(self):
        s = _make_serialized({
            "act_a": _make_transition(
                input_places=["p0"],
                output_places=["p1"],
                preconditions=[[{"attribute": "status", "predicate": "=", "value": "ready"}]],
            ),
        })
        report = validate_plan(
            ["act_a"], ["p0"], s,
            init_attributes=[{"attribute": "status", "value": "ready"}],
            variant_effects={},
        )
        assert report.valid is True

    def test_precondition_not_equal_satisfied(self):
        s = _make_serialized({
            "act_a": _make_transition(
                input_places=["p0"],
                output_places=["p1"],
                preconditions=[[{"attribute": "status", "predicate": "<>", "value": "done"}]],
            ),
        })
        report = validate_plan(
            ["act_a"], ["p0"], s,
            init_attributes=[{"attribute": "status", "value": "ready"}],
            variant_effects={},
        )
        assert report.valid is True

    def test_preconditions_not_checked_without_variant_effects(self):
        s = _make_serialized({
            "act_a": _make_transition(
                input_places=["p0"],
                output_places=["p1"],
                preconditions=[[{"attribute": "status", "predicate": "=", "value": "done"}]],
            ),
        })
        report = validate_plan(["act_a"], ["p0"], s)
        assert report.valid is True
        assert report.attribute_checked is False


# ---------------------------------------------------------------------------
# validate_plan — exact variant effects
# ---------------------------------------------------------------------------

class TestVariantEffects:
    def test_exact_effects_applied_from_variant(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        ve = {"exec_act_a_v1": {"status": "done"}}
        report = validate_plan(["exec_act_a_v1"], ["p0"], s, variant_effects=ve)
        assert report.valid is True
        assert report.final_attributes.get("status") == "done"

    def test_precondition_satisfied_by_previous_variant_effect(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
            "act_b": _make_transition(
                input_places=["p1"],
                output_places=["p2"],
                preconditions=[[{"attribute": "status", "predicate": "=", "value": "done"}]],
            ),
        })
        ve = {"act_a": {"status": "done"}}
        report = validate_plan(["act_a", "act_b"], ["p0"], s, variant_effects=ve)
        assert report.valid is True

    def test_precondition_violation_with_exact_effects_fails(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
            "act_b": _make_transition(
                input_places=["p1"],
                output_places=["p2"],
                preconditions=[[{"attribute": "status", "predicate": "=", "value": "done"}]],
            ),
        })
        ve = {"act_a": {"status": "pending"}}
        report = validate_plan(["act_a", "act_b"], ["p0"], s, variant_effects=ve)
        assert report.valid is False
        assert report.error_action == "act_b"

    def test_unknown_variant_name_uses_empty_effects(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        report = validate_plan(["act_a"], ["p0"], s, variant_effects={})
        assert report.valid is True
        assert report.final_attributes == {}

    def test_init_attributes_loaded_when_variant_effects_supplied(self):
        s = _make_serialized({
            "act_a": _make_transition(
                input_places=["p0"],
                output_places=["p1"],
                preconditions=[[{"attribute": "phase", "predicate": "=", "value": "open"}]],
            ),
        })
        report = validate_plan(
            ["act_a"], ["p0"], s,
            init_attributes=[{"attribute": "phase", "value": "open"}],
            variant_effects={},
        )
        assert report.valid is True

    def test_init_attributes_ignored_without_variant_effects(self):
        # Tokens move correctly even without variant_effects; init_attributes
        # are not loaded into the state machine.
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        report = validate_plan(
            ["act_a"], ["p0"], s,
            init_attributes=[{"attribute": "x", "value": "1"}],
        )
        assert report.valid is True
        assert report.final_attributes == {}

    def test_multiple_effects_from_variant_all_applied(self):
        s = _make_serialized({
            "act_a": _make_transition(input_places=["p0"], output_places=["p1"]),
        })
        ve = {"act_a": {"status": "done", "amount": "high"}}
        report = validate_plan(["act_a"], ["p0"], s, variant_effects=ve)
        assert report.final_attributes == {"status": "done", "amount": "high"}


# ---------------------------------------------------------------------------
# build_variant_effects_lookup
# ---------------------------------------------------------------------------

class TestBuildVariantEffectsLookup:
    def _make_registry(self, variants_dict):
        from unittest.mock import MagicMock
        registry = MagicMock()
        registry._variants = variants_dict
        return registry

    def _make_action(self, name, effects):
        from unittest.mock import MagicMock
        action = MagicMock()
        action.name = name
        action.effects = effects
        return action

    def test_extracts_attr_is_effect(self):
        from encoding.pddl_model import PDDLEffect
        action = self._make_action("exec_register_v0", [PDDLEffect.set_attr_is("status", "done")])
        registry = self._make_registry({"exec_register": [action]})
        lookup = build_variant_effects_lookup(registry)
        assert lookup["exec_register_v0"] == {"status": "done"}

    def test_extracts_attr_true_effect(self):
        from encoding.pddl_model import PDDLEffect
        action = self._make_action("exec_pay", [PDDLEffect.set_attr_true("urgent")])
        registry = self._make_registry({"exec_pay": [action]})
        lookup = build_variant_effects_lookup(registry)
        assert lookup["exec_pay"] == {"urgent": "true"}

    def test_extracts_attr_false_effect(self):
        from encoding.pddl_model import PDDLEffect
        action = self._make_action("exec_pay", [PDDLEffect.set_attr_false("active")])
        registry = self._make_registry({"exec_pay": [action]})
        lookup = build_variant_effects_lookup(registry)
        assert lookup["exec_pay"] == {"active": "false"}

    def test_clear_effects_skipped(self):
        from encoding.pddl_model import PDDLEffect
        action = self._make_action("exec_pay", [PDDLEffect.clear_attr_is("status", "done")])
        registry = self._make_registry({"exec_pay": [action]})
        lookup = build_variant_effects_lookup(registry)
        assert lookup["exec_pay"] == {}

    def test_marked_effects_skipped(self):
        from encoding.pddl_model import PDDLEffect
        action = self._make_action("exec_pay", [PDDLEffect(kind="marked", attribute="p_end")])
        registry = self._make_registry({"exec_pay": [action]})
        lookup = build_variant_effects_lookup(registry)
        assert lookup["exec_pay"] == {}

    def test_variant_with_no_effects_maps_to_empty(self):
        action = self._make_action("exec_pay", [])
        registry = self._make_registry({"exec_pay": [action]})
        lookup = build_variant_effects_lookup(registry)
        assert lookup["exec_pay"] == {}

    def test_multiple_variants_all_present(self):
        from encoding.pddl_model import PDDLEffect
        v0 = self._make_action("exec_pay_v0", [PDDLEffect.set_attr_is("status", "ok")])
        v1 = self._make_action("exec_pay_v1", [PDDLEffect.set_attr_is("status", "fail")])
        registry = self._make_registry({"exec_pay": [v0, v1]})
        lookup = build_variant_effects_lookup(registry)
        assert lookup["exec_pay_v0"] == {"status": "ok"}
        assert lookup["exec_pay_v1"] == {"status": "fail"}

    def test_multiple_effects_on_same_variant(self):
        from encoding.pddl_model import PDDLEffect
        effects = [
            PDDLEffect.set_attr_is("status", "done"),
            PDDLEffect.set_attr_true("urgent"),
        ]
        action = self._make_action("exec_register_v0", effects)
        registry = self._make_registry({"exec_register": [action]})
        lookup = build_variant_effects_lookup(registry)
        assert lookup["exec_register_v0"] == {"status": "done", "urgent": "true"}
