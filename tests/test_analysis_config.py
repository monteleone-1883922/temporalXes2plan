"""AnalysisConfig: the attribute lists must be consistent."""
import pytest

from models import AnalysisConfig


class TestKeepVersusIgnored:
    def test_no_conflict_by_default(self):
        config = AnalysisConfig()
        assert config.keep_attributes == ()
        assert config.noise_filter is False
        assert config.unused_attribute_filter is False

    def test_keeping_a_default_ignored_attribute_raises(self):
        with pytest.raises(ValueError, match="org:resource"):
            AnalysisConfig(keep_attributes=("org:resource",))

    def test_keeping_a_custom_ignored_attribute_raises(self):
        ignored = AnalysisConfig().ignored_attributes | {"junk"}
        with pytest.raises(ValueError, match="junk"):
            AnalysisConfig(ignored_attributes=ignored, keep_attributes=("junk",))

    def test_disjoint_lists_are_fine(self):
        ignored = AnalysisConfig().ignored_attributes | {"junk"}
        config = AnalysisConfig(ignored_attributes=ignored, keep_attributes=("useful",))
        assert config.keep_attributes == ("useful",)

    def test_message_names_both_lists(self):
        with pytest.raises(ValueError) as exc:
            AnalysisConfig(keep_attributes=("org:resource",))
        assert "keep_attributes" in str(exc.value) and "ignored_attributes" in str(exc.value)
