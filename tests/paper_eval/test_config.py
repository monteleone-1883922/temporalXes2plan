"""The per-log attribute lists in paper_eval.config.LOGS must be consistent."""
import pytest

from models import AnalysisConfig
from paper_eval.config import LOGS


@pytest.mark.parametrize("spec", LOGS.values(), ids=lambda s: s.name)
def test_ignore_and_keep_lists_do_not_overlap(spec):
    assert not set(spec.ignore_attributes) & set(spec.keep_attributes)


@pytest.mark.parametrize("spec", LOGS.values(), ids=lambda s: s.name)
def test_keep_does_not_name_a_default_ignored_attribute(spec):
    # keep only protects from the automatic filters; AnalysisConfig raises otherwise
    assert not set(spec.keep_attributes) & set(AnalysisConfig().ignored_attributes)


def test_sepsis_keeps_diagnose_from_the_noise_filter():
    assert "Diagnose" in LOGS["sepsis"].keep_attributes
