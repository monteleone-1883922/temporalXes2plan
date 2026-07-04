"""Prefix sampler for evaluation traces.

Converts a pm4py Trace into a PrefixSample by:
1. Sampling a random prefix percentage in [min_prefix_pct, max_prefix_pct].
2. Calling EvalAPI.replay_trace() with the full trace + n_prefix to derive
   the init state using tau-aware backtracking replay.
3. Computing prefix and full-trace durations from timestamps when available.

Returns None when the trace is too short to produce a meaningful prefix or
when replay fails (non-blocking failure).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from evaluation.eval_api import EvalAPI

logger = logging.getLogger(__name__)


@dataclass
class PrefixSample:
    """A sampled prefix from a test trace with its derived init state.

    Attributes:
        prefix_events: Events included in the prefix (list of pm4py Event).
        suffix_events: Events NOT included in the prefix (complement).
        prefix_ratio: Actual fraction of events used as prefix (0 < ratio ≤ 1).
        init_places: Marking derived by replaying the prefix.
        init_effects: Attribute assignments derived from the prefix replay.
        replayed_activities: Ordered list of activity names replayed.
        final_event_attributes: Raw attribute dict of the last prefix event.
        prefix_duration_s: Duration of the prefix in seconds, or None if no
            timestamps are available.
        full_duration_s: Duration of the full trace in seconds, or None if no
            timestamps are available.
        warnings: Non-blocking warnings emitted during replay.
    """

    prefix_events: List[Any]
    suffix_events: List[Any]
    prefix_ratio: float
    init_places: List[str]
    init_effects: List[Dict[str, Any]]
    replayed_activities: List[str]
    final_event_attributes: Dict[str, Any]
    prefix_duration_s: Optional[float]
    full_duration_s: Optional[float]
    warnings: List[str] = field(default_factory=list)


def sample_prefix(
    trace: Any,
    serialized: Dict[str, Any],
    api: EvalAPI,
    min_prefix_pct: float = 0.2,
    max_prefix_pct: float = 0.8,
    seed: Optional[int] = None,
    min_prefix_events: int = 1,
    tau_max_depth: int = 10,
) -> Optional[PrefixSample]:
    """Sample a random prefix from a trace and derive its init state via replay.

    Args:
        trace: A pm4py Trace object from the test set.
        serialized: The current.json dict (output of serialize_parse_result).
        api: An EvalAPI instance used to call replay_trace().
        min_prefix_pct: Minimum fraction of events to include in the prefix.
        max_prefix_pct: Maximum fraction of events to include in the prefix.
        seed: Optional random seed for reproducibility.
        min_prefix_events: Minimum number of events required in the prefix.
            Returns None if the prefix would be shorter.
        tau_max_depth: Maximum tau chain depth for BFS search during replay.

    Returns:
        PrefixSample on success, or None if the trace is too short or replay
        fails.
    """
    events = list(trace)
    n = len(events)

    rng = random.Random(seed)
    prefix_pct = rng.uniform(min_prefix_pct, max_prefix_pct)
    n_prefix = max(1, round(n * prefix_pct))

    if n_prefix < min_prefix_events:
        logger.debug(
            "Trace too short: prefix would have %d events (min %d required).",
            n_prefix, min_prefix_events,
        )
        return None

    prefix_events = events[:n_prefix]
    suffix_events = events[n_prefix:]
    prefix_ratio = n_prefix / n

    try:
        replay = api.replay_trace(trace, serialized, n_prefix=n_prefix, tau_max_depth=tau_max_depth)
    except Exception as exc:
        logger.warning("Replay failed: %s", exc)
        return None

    prefix_duration_s = _duration_seconds(prefix_events)
    full_duration_s = _duration_seconds(events)
    final_attrs = {eff["attribute"]: eff["value"] for eff in replay.final_effects}

    return PrefixSample(
        prefix_events=prefix_events,
        suffix_events=suffix_events,
        prefix_ratio=prefix_ratio,
        init_places=replay.init_places,
        init_effects=replay.init_effects,
        replayed_activities=replay.replayed_activities,
        final_event_attributes=final_attrs,
        prefix_duration_s=prefix_duration_s,
        full_duration_s=full_duration_s,
        warnings=replay.warnings,
    )


# ---------------------------------------------------------------------------
# Internal helpers


def _duration_seconds(events: List[Any]) -> Optional[float]:
    """Compute elapsed time in seconds between the first and last event.

    Args:
        events: Ordered list of pm4py Event objects.

    Returns:
        Duration in seconds, or None if fewer than two events have timestamps.
    """
    if len(events) < 2:
        return None

    timestamps = []
    for e in events:
        ts = e.get("time:timestamp")
        if ts is not None:
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                timestamps.append(ts)

    if len(timestamps) < 2:
        return None

    delta = timestamps[-1] - timestamps[0]
    return delta.total_seconds()


def _event_attributes(event: Any) -> Dict[str, Any]:
    """Extract non-metadata attributes from a pm4py Event.

    Args:
        event: A pm4py Event object.

    Returns:
        Dict of attribute key → value, excluding internal pm4py metadata keys.
    """
    _META = frozenset({
        "concept:name", "time:timestamp", "lifecycle:transition",
        "org:resource", "org:group", "org:role", "@@index", "@@classifier",
    })
    return {k: v for k, v in event.items() if k not in _META}
