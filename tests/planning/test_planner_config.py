"""Unit tests for planning.planner_config — pure I/O-free logic + file round-trip."""
import json

import pytest

from planning.planner_config import (
    FastDownwardConfig,
    OpticConfig,
    PlannerConfig,
    _merge,
    config_to_dict,
    load_planner_config,
    save_planner_config,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_fast_downward_defaults(self):
        cfg = FastDownwardConfig()
        assert cfg.search == "astar_lmcut"
        assert cfg.timeout == 30
        assert cfg.memory_mb is None

    def test_optic_defaults(self):
        cfg = OpticConfig()
        assert cfg.stop_at_first is True
        assert cfg.ignore_costs is False
        assert cfg.timeout == 60
        assert cfg.memory_mb == 4000

    def test_planner_config_has_both_sub_configs(self):
        cfg = PlannerConfig()
        assert isinstance(cfg.fast_downward, FastDownwardConfig)
        assert isinstance(cfg.optic, OpticConfig)


# ---------------------------------------------------------------------------
# _merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_overrides_single_field(self):
        result = _merge(FastDownwardConfig, {"timeout": 120})
        assert result.timeout == 120
        assert result.search == "astar_lmcut"   # default preserved

    def test_merge_all_fields(self):
        result = _merge(FastDownwardConfig, {"search": "astar_blind", "timeout": 60, "memory_mb": 2048})
        assert result.search == "astar_blind"
        assert result.timeout == 60
        assert result.memory_mb == 2048

    def test_merge_ignores_unknown_keys(self):
        result = _merge(FastDownwardConfig, {"timeout": 45, "nonexistent_key": "boom"})
        assert result.timeout == 45
        assert not hasattr(result, "nonexistent_key")

    def test_merge_empty_dict_returns_defaults(self):
        result = _merge(OpticConfig, {})
        assert result == OpticConfig()

    def test_merge_optic_flags(self):
        result = _merge(OpticConfig, {"stop_at_first": False, "ignore_costs": True})
        assert result.stop_at_first is False
        assert result.ignore_costs is True


# ---------------------------------------------------------------------------
# config_to_dict
# ---------------------------------------------------------------------------

class TestConfigToDict:
    def test_returns_plain_dict(self):
        d = config_to_dict(PlannerConfig())
        assert isinstance(d, dict)
        assert "fast_downward" in d
        assert "optic" in d

    def test_values_are_json_serializable(self):
        d = config_to_dict(PlannerConfig())
        json.dumps(d)   # should not raise

    def test_fd_fields_present(self):
        d = config_to_dict(PlannerConfig())
        fd = d["fast_downward"]
        assert "search" in fd
        assert "timeout" in fd
        assert "memory_mb" in fd

    def test_optic_fields_present(self):
        d = config_to_dict(PlannerConfig())
        op = d["optic"]
        assert "stop_at_first" in op
        assert "ignore_costs" in op
        assert "timeout" in op
        assert "memory_mb" in op

    def test_custom_values_preserved(self):
        cfg = PlannerConfig(
            fast_downward=FastDownwardConfig(search="lama_first", timeout=120),
        )
        d = config_to_dict(cfg)
        assert d["fast_downward"]["search"] == "lama_first"
        assert d["fast_downward"]["timeout"] == 120


# ---------------------------------------------------------------------------
# load / save round-trip
# ---------------------------------------------------------------------------

class TestLoadSave:
    def test_save_and_load_roundtrip(self, tmp_path):
        original = PlannerConfig(
            fast_downward=FastDownwardConfig(search="astar_blind", timeout=90, memory_mb=2000),
            optic=OpticConfig(stop_at_first=False, ignore_costs=True, timeout=45, memory_mb=3000),
        )
        save_planner_config(original, tmp_path)
        loaded = load_planner_config(tmp_path)
        assert loaded.fast_downward == original.fast_downward
        assert loaded.optic == original.optic

    def test_load_missing_file_returns_defaults(self, tmp_path):
        cfg = load_planner_config(tmp_path)
        assert cfg == PlannerConfig()

    def test_load_partial_file_merges_defaults(self, tmp_path):
        # File contains only fast_downward section
        (tmp_path / "planner_config.json").write_text(
            json.dumps({"fast_downward": {"timeout": 99}}), encoding="utf-8"
        )
        cfg = load_planner_config(tmp_path)
        assert cfg.fast_downward.timeout == 99
        assert cfg.optic == OpticConfig()   # optic untouched

    def test_load_corrupt_file_returns_defaults(self, tmp_path):
        (tmp_path / "planner_config.json").write_text("NOT JSON", encoding="utf-8")
        cfg = load_planner_config(tmp_path)
        assert cfg == PlannerConfig()

    def test_save_creates_file(self, tmp_path):
        save_planner_config(PlannerConfig(), tmp_path)
        assert (tmp_path / "planner_config.json").exists()

    def test_save_is_valid_json(self, tmp_path):
        save_planner_config(PlannerConfig(), tmp_path)
        raw = (tmp_path / "planner_config.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        assert "fast_downward" in data
        assert "optic" in data
