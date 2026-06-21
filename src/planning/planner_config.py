"""Planner configuration dataclasses with load/save helpers."""

import json
import os
import tempfile
from dataclasses import dataclass, field, fields as dc_fields, asdict
from pathlib import Path
from typing import Any, Dict, Optional


FILENAME = "planner_config.json"


@dataclass
class FastDownwardConfig:
    search: str = "astar_lmcut"
    timeout: int = 30
    memory_mb: Optional[int] = None  # None = no limit; passed as --overall-memory-limit


@dataclass
class OpticConfig:
    stop_at_first: bool = True
    ignore_costs: bool = False
    timeout: int = 60
    memory_mb: int = 4000


@dataclass
class PlannerConfig:
    fast_downward: FastDownwardConfig = field(default_factory=FastDownwardConfig)
    optic: OpticConfig = field(default_factory=OpticConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge(cls, saved: Dict[str, Any]):
    """Instantiate cls overriding defaults with values present in saved."""
    valid = {f.name for f in dc_fields(cls)}
    return cls(**{k: saved[k] for k in valid if k in saved})


def load_planner_config(config_dir: Path) -> PlannerConfig:
    """Load config from config_dir/planner_config.json, falling back to defaults.

    Args:
        config_dir: Directory that may contain planner_config.json.

    Returns:
        PlannerConfig merging saved values onto defaults.
    """
    path = config_dir / FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return PlannerConfig(
            fast_downward=_merge(FastDownwardConfig, raw.get("fast_downward", {})),
            optic=_merge(OpticConfig, raw.get("optic", {})),
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return PlannerConfig()


def save_planner_config(config: PlannerConfig, config_dir: Path) -> None:
    """Atomically write config to config_dir/planner_config.json.

    Args:
        config: PlannerConfig to persist.
        config_dir: Target directory (must exist).
    """
    path = config_dir / FILENAME
    data = {
        "fast_downward": asdict(config.fast_downward),
        "optic": asdict(config.optic),
    }
    fd, tmp = tempfile.mkstemp(dir=str(config_dir), suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def config_to_dict(config: PlannerConfig) -> Dict[str, Any]:
    """Serialize PlannerConfig to a plain dict for JSON responses."""
    return {
        "fast_downward": asdict(config.fast_downward),
        "optic": asdict(config.optic),
    }
