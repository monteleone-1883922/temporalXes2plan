"""Utilities for persisting and updating the Petri net graph.

save_original_and_current: writes original.json and current.json to disk
after the pipeline serializes a ParseResult.  Future functions here will
handle GUI-driven graph edits (node moves, manual arc changes, etc.).
"""
from typing import Any, Dict, List, Tuple

import core_utils as utils
from encoding import ActionRegistry, PDDLEffect
from models import ParseResult

logger = utils.get_logger(__name__)




def update_parse_result(registry: ActionRegistry, result: ParseResult) -> None:
    """Fill effect_groups on each TransitionInfo from the ActionRegistry.

    For each execute_<act> action, collects the positive attribute assignments
    from every variant, deduplicates identical groups, and assigns the list to
    result.transitions[act].effect_groups.

    Args:
        registry: ActionRegistry (post-EffectDuplicator).
        result: ParseResult with transitions dict to annotate.
    """

    _POSITIVE_KINDS = {"attr_is", "attr_true", "attr_false"}

    def _eff_value(eff: PDDLEffect) -> str:
        if eff.kind == "attr_true":
            return "true"
        if eff.kind == "attr_false":
            return "false"
        return str(eff.value) if eff.value is not None else "true"

    for base_name in registry._order:
        if not base_name.startswith("execute_"):
            continue
        trans_name = base_name[len("execute_"):]
        if trans_name not in result.transitions:
            continue

        seen: set = set()
        groups: List[List[Tuple[str, str]]] = []

        for variant in registry._variants[base_name]:
            group: List[Tuple[str, str]] = sorted(
                (eff.attribute, _eff_value(eff))
                for eff in variant.effects
                if eff.kind in _POSITIVE_KINDS and not eff.clear
            )
            if not group:
                continue
            key = tuple(group)
            if key not in seen:
                seen.add(key)
                groups.append(group)

        result.transitions[trans_name].effect_groups = groups


def save_original_and_current(
    config_dir: str, data: Dict[str, Any]
) -> tuple[bool, bool]:
    """Atomically write original.json and current.json if they do not exist.

    Args:
        config_dir: Directory for the configuration (e.g. data/sepsis/).
        data: Serialized Petri net dict to persist.

    Returns:
        Tuple (original_written, current_written) — True when the file was
        created, False when it already existed and was skipped.
    """
    import json
    import os
    import tempfile

    os.makedirs(config_dir, exist_ok=True)

    written: list[bool] = []
    for filename in ("original.json", "current.json"):
        path = os.path.join(config_dir, filename)
        if os.path.exists(path):
            logger.info("Skipping %s — already exists", path)
            written.append(False)
            continue
        fd, tmp_path = tempfile.mkstemp(dir=config_dir, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
            logger.info("Written %s", path)
            written.append(True)
        except Exception:
            os.unlink(tmp_path)
            raise

    return (written[0], written[1])
