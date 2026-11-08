"""unit_costs of paper_eval: recorded with the model, and a cached model must agree with it."""
import json

import pytest

from paper_eval.build_model import build_model, read_unit_costs
from paper_eval.config import PaperEvalConfig


def _model_dir(tmp_path, meta=None):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "current.json").write_text("{}")  # what marks a built model
    if meta is not None:
        (model_dir / "model_meta.json").write_text(json.dumps(meta))
    return model_dir


class TestReadUnitCosts:
    def test_recorded_value(self, tmp_path):
        assert read_unit_costs(_model_dir(tmp_path, {"unit_costs": True})) is True

    def test_recorded_false(self, tmp_path):
        assert read_unit_costs(_model_dir(tmp_path, {"unit_costs": False})) is False

    def test_a_model_without_meta_has_probability_costs(self, tmp_path):
        assert read_unit_costs(_model_dir(tmp_path)) is False

    def test_a_meta_written_before_the_option_has_probability_costs(self, tmp_path):
        assert read_unit_costs(_model_dir(tmp_path, {"discovery_algorithm": "powl"})) is False


class TestCachedModelMustAgree:
    def test_a_model_built_with_other_costs_is_refused(self, tmp_path):
        model_dir = _model_dir(tmp_path, {"unit_costs": False})
        with pytest.raises(ValueError, match="unit_costs"):
            build_model(tmp_path / "log.xes", None, PaperEvalConfig(unit_costs=True), model_dir)

    def test_the_other_way_round_is_refused_too(self, tmp_path):
        model_dir = _model_dir(tmp_path, {"unit_costs": True})
        with pytest.raises(ValueError, match="--force"):
            build_model(tmp_path / "log.xes", None, PaperEvalConfig(unit_costs=False), model_dir)

    def test_a_model_with_the_same_costs_is_reused(self, tmp_path):
        model_dir = _model_dir(tmp_path, {"unit_costs": True})
        assert build_model(tmp_path / "log.xes", None, PaperEvalConfig(unit_costs=True), model_dir) == model_dir
