"""Utilities for persisting and (in future) updating the Petri net graph.

save_original_and_current: writes original.json and current.json to disk
after the pipeline serializes a ParseResult.  Future functions here will
handle GUI-driven graph edits (node moves, manual arc changes, etc.).
"""
from typing import Any, Dict

import core_utils as utils

logger = utils.get_logger(__name__)


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
