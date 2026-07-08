"""Unit tests for encoding.problem_builder.ProblemBuilder."""
import pytest

from encoding.problem_builder import ProblemBuilder


CATALOG_MIXED = {
    "status": {"type": "categorical", "possible_values": ["admitted", "discharged"]},
    "critical": {"type": "boolean", "possible_values": ["true", "false"]},
    "crp": {"type": "numerical", "possible_values": ["lt_50", "gte_50"]},
}

CATALOG_BOOL_ONLY = {
    "active": {"type": "boolean", "possible_values": ["true", "false"]},
}

CATALOG_EMPTY: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(**kwargs) -> str:
    defaults = dict(
        problem_name="test_prob",
        domain_name="test_dom",
        init_effects=[],
        goal_sop=[[{"attribute": "status", "predicate": "=", "value": "discharged"}]],
        attribute_catalog=CATALOG_MIXED,
        init_places=None,
    )
    defaults.update(kwargs)
    return ProblemBuilder().build(**defaults)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class TestProblemBuilderStructure:
    def test_output_contains_define(self):
        result = _build()
        assert "(define (problem test_prob)" in result

    def test_output_contains_domain(self):
        result = _build()
        assert "(:domain test_dom)" in result

    def test_output_contains_init_section(self):
        result = _build()
        assert "(:init" in result

    def test_output_contains_goal_section(self):
        result = _build()
        assert "(:goal" in result

    def test_output_is_valid_pddl_parens(self):
        result = _build()
        assert result.count("(") == result.count(")")


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestProblemBuilderInit:
    def test_empty_init_has_no_atoms(self):
        # attribute_catalog=CATALOG_EMPTY -- with CATALOG_MIXED, every
        # categorical/numerical attribute not set by init_effects gets a
        # default "_val_none" atom (see _build_init_atoms's fill-in loop),
        # so this only holds for a genuinely empty catalog.
        result = _build(init_effects=[], attribute_catalog=CATALOG_EMPTY)
        lines = [l.strip() for l in result.splitlines()]
        init_idx = next(i for i, l in enumerate(lines) if "(:init" in l)
        end_idx = next(i for i, l in enumerate(lines) if i > init_idx and l == ")")
        init_body = lines[init_idx + 1: end_idx]
        assert all(l == "" for l in init_body)

    def test_init_places_adds_marked_atoms(self):
        result = _build(init_places=["place_start"])
        assert "(marked place_start)" in result

    def test_multiple_init_places_add_multiple_marked_atoms(self):
        # Place ids go through PDDLEffect.to_pddl()'s sanitize_name() same as
        # any other attribute/place name -- "p_A"/"p_B" come out lowercased.
        result = _build(init_places=["p_A", "p_B"])
        assert "(marked p_a)" in result
        assert "(marked p_b)" in result

    def test_categorical_init_effect(self):
        # Values go through sanitize_value(attr, value), which prefixes with
        # "{attr}_val_" -- the same PDDL constant name the domain encoder
        # uses (core_utils.sanitize_value), not the bare log value.
        result = _build(
            init_effects=[{"attribute": "status", "value": "admitted"}],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(status_is status_val_admitted)" in result

    @pytest.mark.xfail(
        reason=(
            "Known bug in ProblemBuilder._build_init_atoms: `value` is "
            "assigned from utils.sanitize_value(attr, ...) (e.g. "
            "'critical_val_true') before the `value == \"true\"` check, "
            "which can now never match -- boolean init effects always "
            "resolve to set_attr_false regardless of the actual value. "
            "Flagged to the user, not fixed here (tests-only task)."
        ),
        strict=True,
    )
    def test_boolean_true_init_effect(self):
        result = _build(
            init_effects=[{"attribute": "critical", "value": "true"}],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(critical_true)" in result

    def test_boolean_false_init_effect(self):
        result = _build(
            init_effects=[{"attribute": "critical", "value": "false"}],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(critical_false)" in result

    def test_numerical_init_effect(self):
        result = _build(
            init_effects=[{"attribute": "crp", "value": "lt_50"}],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(crp_is crp_val_lt_50)" in result

    @pytest.mark.xfail(
        reason=(
            "Same known bug as test_boolean_true_init_effect: the boolean "
            "'critical' effect always resolves to set_attr_false."
        ),
        strict=True,
    )
    def test_multiple_init_effects(self):
        result = _build(
            init_effects=[
                {"attribute": "status", "value": "admitted"},
                {"attribute": "critical", "value": "true"},
            ],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(status_is status_val_admitted)" in result
        assert "(critical_true)" in result

    def test_init_places_before_effects(self):
        result = _build(
            init_places=["p_start"],
            init_effects=[{"attribute": "status", "value": "admitted"}],
            attribute_catalog=CATALOG_MIXED,
        )
        marked_idx = result.index("(marked p_start)")
        effect_idx = result.index("(status_is status_val_admitted)")
        assert marked_idx < effect_idx

    @pytest.mark.xfail(
        reason=(
            "Known bug in ProblemBuilder._build_init_atoms: `value` is "
            "already the sanitized '{attr}_val_empty' string (sanitize_value "
            "never returns an empty string, even for raw_value='') by the "
            "time `if not value: continue` runs, so the empty-value skip "
            "never actually triggers for non-bool attributes. Flagged to "
            "the user, not fixed here (tests-only task)."
        ),
        strict=True,
    )
    def test_effect_without_value_skipped_for_non_bool(self, tmp_path):
        # Use a catalog where "flag" is categorical and goal uses a different attr
        catalog = {"flag": {"type": "categorical", "possible_values": ["on", "off"]}}
        result = ProblemBuilder().build(
            problem_name="p",
            domain_name="d",
            init_effects=[{"attribute": "flag", "value": ""}],
            goal_sop=[[{"attribute": "flag", "predicate": "=", "value": "on"}]],
            attribute_catalog=catalog,
        )
        # The :init block must not contain flag_is (empty value is skipped)
        init_section = result[result.index("(:init"): result.index("(:goal")]
        assert "flag_is" not in init_section


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

class TestProblemBuilderGoal:
    def test_single_condition_no_and_wrapper(self):
        result = _build(
            goal_sop=[[{"attribute": "status", "predicate": "=", "value": "discharged"}]],
        )
        assert "(status_is status_val_discharged)" in result
        assert "(and (status_is status_val_discharged))" not in result

    def test_and_clause_multiple_conditions(self):
        result = _build(
            goal_sop=[[
                {"attribute": "status", "predicate": "=", "value": "discharged"},
                {"attribute": "critical", "predicate": "=", "value": "false"},
            ]],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(and (status_is status_val_discharged) (critical_false))" in result

    def test_or_clauses_multiple_clauses(self):
        result = _build(
            goal_sop=[
                [{"attribute": "status", "predicate": "=", "value": "discharged"}],
                [{"attribute": "critical", "predicate": "=", "value": "false"}],
            ],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(or " in result
        assert "(status_is status_val_discharged)" in result
        assert "(critical_false)" in result

    def test_boolean_goal_true_predicate(self):
        result = _build(
            goal_sop=[[{"attribute": "critical", "predicate": "=", "value": "true"}]],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(critical_true)" in result

    def test_boolean_goal_false_predicate(self):
        result = _build(
            goal_sop=[[{"attribute": "critical", "predicate": "=", "value": "false"}]],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(critical_false)" in result

    def test_not_equal_predicate_categorical(self):
        result = _build(
            goal_sop=[[{"attribute": "status", "predicate": "<>", "value": "admitted"}]],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(status_is_not status_val_admitted)" in result

    def test_not_equal_bool_becomes_opposite(self):
        result = _build(
            goal_sop=[[{"attribute": "critical", "predicate": "<>", "value": "true"}]],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(critical_false)" in result

    def test_empty_goal_sop_emits_and(self):
        result = _build(goal_sop=[])
        assert "(and)" in result

    def test_sop_with_empty_clauses_only_emits_and(self):
        result = _build(goal_sop=[[], []])
        assert "(and)" in result


# ---------------------------------------------------------------------------
# Metric and costs
# ---------------------------------------------------------------------------

class TestProblemBuilderMetric:

    def test_no_metric_section_by_default(self):
        result = _build()
        assert "(:metric" not in result

    def test_metric_minimize_cost_emitted(self):
        result = _build(metric="minimize_cost")
        assert "(:metric minimize (total-cost))" in result

    def test_metric_minimize_time(self):
        result = _build(metric="minimize_time")
        assert "(:metric minimize (total-time))" in result

    def test_metric_minimize_time_independent_of_cost_metric(self):
        result = _build(metric="minimize_time")
        assert "(:metric minimize (total-time))" in result
        assert "total-cost" not in result

    def test_metric_section_after_goal(self):
        result = _build(metric="minimize_time")
        goal_pos = result.index("(:goal")
        metric_pos = result.index("(:metric")
        assert metric_pos > goal_pos

    def test_total_cost_init_atom_with_minimize_cost(self):
        result = _build(metric="minimize_cost")
        assert "(= (total-cost) 0)" in result

    def test_no_total_cost_init_atom_without_metric(self):
        result = _build()
        assert "(= (total-cost) 0)" not in result

    def test_total_cost_init_before_attribute_effects(self):
        result = _build(
            metric="minimize_cost",
            init_effects=[{"attribute": "status", "value": "admitted"}],
        )
        cost_pos = result.index("(= (total-cost) 0)")
        effect_pos = result.index("(status_is status_val_admitted)")
        assert cost_pos < effect_pos

    def test_parentheses_balanced_with_metric(self):
        result = _build(metric="minimize_cost")
        assert result.count("(") == result.count(")")


# ---------------------------------------------------------------------------
# require_completion flag
# ---------------------------------------------------------------------------

class TestProblemBuilderRequireCompletion:

    def test_no_completion_by_default(self):
        result = _build()
        assert "(marked" not in result.split("(:goal")[1]

    def test_require_completion_adds_end_place_to_goal(self):
        result = _build(require_completion=True, end_place="p_end")
        goal_section = result.split("(:goal")[1]
        assert "(marked p_end)" in goal_section

    def test_require_completion_with_empty_sop(self):
        result = _build(goal_sop=[], require_completion=True, end_place="p_end")
        goal_section = result.split("(:goal")[1]
        assert "(marked p_end)" in goal_section
        assert "(or" not in goal_section

    def test_require_completion_distributes_over_or_clauses(self):
        result = _build(
            goal_sop=[
                [{"attribute": "status", "predicate": "=", "value": "discharged"}],
                [{"attribute": "status", "predicate": "=", "value": "admitted"}],
            ],
            require_completion=True,
            end_place="p_end",
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(or" in result
        assert result.count("(marked p_end)") == 2

    def test_require_completion_without_end_place_ignored(self):
        result = _build(require_completion=True, end_place=None)
        assert "(marked" not in result.split("(:goal")[1]

    def test_require_completion_single_clause_no_or_wrapper(self):
        result = _build(
            goal_sop=[[{"attribute": "status", "predicate": "=", "value": "discharged"}]],
            require_completion=True,
            end_place="p_end",
            attribute_catalog=CATALOG_MIXED,
        )
        goal_section = result.split("(:goal")[1]
        assert "(or" not in goal_section
        assert "(and (status_is status_val_discharged) (marked p_end))" in goal_section


# ---------------------------------------------------------------------------
# Deadline TIL
# ---------------------------------------------------------------------------

class TestProblemBuilderDeadline:

    def test_no_til_by_default(self):
        result = _build()
        assert "(at " not in result

    def test_deadline_til_in_init(self):
        result = _build(deadline=3600.0)
        assert "(deadline_ok)" in result
        assert "(at 3600.0 (not (deadline_ok)))" in result

    def test_deadline_format_one_decimal(self):
        result = _build(deadline=120.0)
        assert "(at 120.0 (not (deadline_ok)))" in result

    def test_deadline_til_before_attribute_effects(self):
        result = _build(
            deadline=300.0,
            init_effects=[{"attribute": "status", "value": "admitted"}],
        )
        til_pos = result.index("(at 300.0")
        effect_pos = result.index("(status_is status_val_admitted)")
        assert til_pos < effect_pos

    def test_deadline_zero_not_emitted(self):
        result = _build(deadline=0.0)
        assert "(at " not in result

    def test_deadline_negative_not_emitted(self):
        result = _build(deadline=-10.0)
        assert "(at " not in result


# ---------------------------------------------------------------------------
# minimize_weighted metric (Step 0b)
# ---------------------------------------------------------------------------

class TestProblemBuilderMinimizeWeighted:

    def test_minimize_weighted_emits_combined_expression(self):
        result = _build(metric="minimize_weighted")
        assert "(+ (total-time) (* " in result
        assert "(total-cost)" in result

    def test_minimize_weighted_default_cost_weight(self):
        result = _build(metric="minimize_weighted")
        assert "0.001" in result

    def test_minimize_weighted_custom_cost_weight(self):
        result = _build(metric="minimize_weighted", cost_weight=0.5)
        assert "0.5" in result
        assert "(+ (total-time) (* 0.5 (total-cost)))" in result

    def test_minimize_weighted_includes_total_cost_init(self):
        result = _build(metric="minimize_weighted")
        assert "(= (total-cost) 0)" in result

    def test_minimize_cost_unchanged(self):
        result = _build(metric="minimize_cost")
        assert "(:metric minimize (total-cost))" in result
        assert "(+ (total-time)" not in result

    def test_minimize_time_unchanged(self):
        result = _build(metric="minimize_time")
        assert "(:metric minimize (total-time))" in result
        assert "(+ (total-time)" not in result

    def test_minimize_weighted_parentheses_balanced(self):
        result = _build(metric="minimize_weighted")
        assert result.count("(") == result.count(")")
