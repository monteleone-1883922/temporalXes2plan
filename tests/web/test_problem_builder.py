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
        init_place=None,
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
        result = _build(init_effects=[])
        lines = [l.strip() for l in result.splitlines()]
        init_idx = next(i for i, l in enumerate(lines) if "(:init" in l)
        end_idx = next(i for i, l in enumerate(lines) if i > init_idx and l == ")")
        init_body = lines[init_idx + 1: end_idx]
        assert all(l == "" for l in init_body)

    def test_init_place_adds_marked_atom(self):
        result = _build(init_place="place_start")
        assert "(marked place_start)" in result

    def test_categorical_init_effect(self):
        result = _build(
            init_effects=[{"attribute": "status", "value": "admitted"}],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(status_is admitted)" in result

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
        assert "(crp_is lt_50)" in result

    def test_multiple_init_effects(self):
        result = _build(
            init_effects=[
                {"attribute": "status", "value": "admitted"},
                {"attribute": "critical", "value": "true"},
            ],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(status_is admitted)" in result
        assert "(critical_true)" in result

    def test_init_place_before_effects(self):
        result = _build(
            init_place="p_start",
            init_effects=[{"attribute": "status", "value": "admitted"}],
            attribute_catalog=CATALOG_MIXED,
        )
        marked_idx = result.index("(marked p_start)")
        effect_idx = result.index("(status_is admitted)")
        assert marked_idx < effect_idx

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
        assert "(status_is discharged)" in result
        assert "(and (status_is discharged))" not in result

    def test_and_clause_multiple_conditions(self):
        result = _build(
            goal_sop=[[
                {"attribute": "status", "predicate": "=", "value": "discharged"},
                {"attribute": "critical", "predicate": "=", "value": "false"},
            ]],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(and (status_is discharged) (critical_false))" in result

    def test_or_clauses_multiple_clauses(self):
        result = _build(
            goal_sop=[
                [{"attribute": "status", "predicate": "=", "value": "discharged"}],
                [{"attribute": "critical", "predicate": "=", "value": "false"}],
            ],
            attribute_catalog=CATALOG_MIXED,
        )
        assert "(or " in result
        assert "(status_is discharged)" in result
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
        assert "(status_is_not admitted)" in result

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
        effect_pos = result.index("(status_is admitted)")
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
        assert "(and (status_is discharged) (marked p_end))" in goal_section


# ---------------------------------------------------------------------------
# Deadline TIL
# ---------------------------------------------------------------------------

class TestProblemBuilderDeadline:

    def test_no_til_by_default(self):
        result = _build()
        assert "(at " not in result

    def test_deadline_til_in_init(self):
        result = _build(deadline=3600.0)
        assert "(at 3600.0 (deadline_exceeded))" in result

    def test_deadline_format_one_decimal(self):
        result = _build(deadline=120.0)
        assert "(at 120.0 (deadline_exceeded))" in result

    def test_deadline_til_before_attribute_effects(self):
        result = _build(
            deadline=300.0,
            init_effects=[{"attribute": "status", "value": "admitted"}],
        )
        til_pos = result.index("(at 300.0")
        effect_pos = result.index("(status_is admitted)")
        assert til_pos < effect_pos

    def test_deadline_zero_not_emitted(self):
        result = _build(deadline=0.0)
        assert "(at " not in result

    def test_deadline_negative_not_emitted(self):
        result = _build(deadline=-10.0)
        assert "(at " not in result
