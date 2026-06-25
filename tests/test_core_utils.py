"""Unit tests for core_utils — sanitize_name, convert_interval_to_lte_gte, discretize_value."""
import pytest

from core_utils import sanitize_name, convert_interval_to_lte_gte, discretize_value


# ---------------------------------------------------------------------------
# sanitize_name
# ---------------------------------------------------------------------------

class TestSanitizeName:
    def test_lowercase(self):
        assert sanitize_name("MyActivity") == "myactivity"

    def test_spaces_to_underscore(self):
        assert sanitize_name("ER Registration") == "er_registration"

    def test_strips_case_prefix(self):
        assert sanitize_name("case:concept:name") == "concept_name"

    def test_strips_case_prefix_case_insensitive(self):
        assert sanitize_name("CASE:status") == "status"

    def test_colon_to_underscore(self):
        assert sanitize_name("concept:name") == "concept_name"

    def test_dash_to_underscore(self):
        assert sanitize_name("my-activity") == "my_activity"

    def test_parens_removed(self):
        # parens stripped (no placeholder), space→_, /→_
        assert sanitize_name("CRP (mg/L)") == "crp_mg_l"

    def test_slash_to_underscore(self):
        assert sanitize_name("a/b") == "a_b"

    def test_backslash_to_underscore(self):
        assert sanitize_name("a\\b") == "a_b"

    def test_empty_string_returns_empty(self):
        assert sanitize_name("") == ""

    def test_already_clean(self):
        assert sanitize_name("register_patient") == "register_patient"

    def test_mixed_complex(self):
        result = sanitize_name("case:concept:name")
        assert result == "concept_name"


# ---------------------------------------------------------------------------
# convert_interval_to_lte_gte
# ---------------------------------------------------------------------------

class TestConvertIntervalToLteGte:
    def test_lower_bound_inf(self):
        assert convert_interval_to_lte_gte("(-inf-6.15]") == "lte_6_15"

    def test_upper_bound_inf(self):
        assert convert_interval_to_lte_gte("(12.5-inf)") == "gte_12_5"

    def test_bounded_interval(self):
        assert convert_interval_to_lte_gte("(10.5-20.0]") == "gte_10_5_lte_20_0"

    def test_integer_bounds(self):
        result = convert_interval_to_lte_gte("(5-10]")
        assert "5" in result
        assert "10" in result

    def test_zero_lower_bound_inf(self):
        assert convert_interval_to_lte_gte("(-inf-0.0]") == "lte_0_0"


# ---------------------------------------------------------------------------
# discretize_value
# ---------------------------------------------------------------------------

class TestDiscretizeValue:
    INTERVALS = {"crp": [10.0, 50.0, 150.0]}

    def test_below_first_threshold(self):
        assert discretize_value("crp", 5.0, self.INTERVALS) == "lte_10_0"

    def test_at_first_threshold(self):
        assert discretize_value("crp", 10.0, self.INTERVALS) == "lte_10_0"

    def test_between_thresholds(self):
        assert discretize_value("crp", 25.0, self.INTERVALS) == "gte_10_0_lte_50_0"

    def test_above_last_threshold(self):
        assert discretize_value("crp", 200.0, self.INTERVALS) == "gte_150_0"

    def test_at_last_threshold(self):
        assert discretize_value("crp", 150.0, self.INTERVALS) == "gte_150_0"

    def test_no_intervals_returns_str(self):
        assert discretize_value("crp", 25.0, None) == "25.0"

    def test_attr_not_in_intervals_returns_str(self):
        assert discretize_value("unknown_attr", 25.0, {"crp": [10.0]}) == "25.0"

    def test_value_none_returns_str(self):
        assert discretize_value("crp", None, self.INTERVALS) == "None"

    def test_interval_string_passed_through_converter(self):
        result = discretize_value("crp", "(-inf-6.15]", None)
        assert result == "lte_6_15"

    def test_non_numeric_string_returned_as_is(self):
        result = discretize_value("status", "admitted", None)
        assert result == "admitted"

    def test_empty_intervals_list_returns_str(self):
        assert discretize_value("crp", 25.0, {"crp": []}) == "25.0"
