"""Prefix sampler for evaluation traces.

Converts a pm4py Trace into a PrefixSample by:
1. Sampling a random prefix percentage in [min_prefix_pct, max_prefix_pct].
2. Calling replay.trace_replayer.TraceReplayer.replay_evaluation_split() with
   the full trace + n_prefix to derive the state at the cut point (Fase 1
   resolves the whole trace, Fase 2 verifies only the tail — see that
   module's docstring, caso 2).
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

from replay.trace_replayer import TraceReplayer

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
        final_event_attributes: Attribute state accumulated over the whole
            trace (the real end-of-trace target Q3's goal is drawn from — see
            replay.trace_replayer.EvaluationReplayOutcome.expected_final_attributes).
        prefix_duration_s: Duration of the prefix in seconds, or None if no
            timestamps are available.
        full_duration_s: Duration of the full trace in seconds, or None if no
            timestamps are available.
        warnings: Non-blocking warnings emitted during replay.
        is_replayable: True when the tail (cut -> end) both completes without
            blocking and matches the real final attribute state exactly — see
            EvaluationReplayOutcome.reached_end/matches_expected_final_state.
        max_modeled_duration_s: Sum of effective_max (domain-declared upper
            duration bound) over every non-tau transition actually fired in
            the tail — see
            replay.trace_replayer.EvaluationReplayOutcome.max_modeled_duration_seconds.
        reached_within_time: True when the tail reaches the end (reached_end)
            AND max_modeled_duration_s fits within the trace's real
            remaining duration (_remaining_budget) — a *sufficient* (not
            merely necessary) condition: even using the domain's
            worst-case duration bound for every fired transition, the
            total still fits under the time the log actually took. False
            (not None) when timestamps are unavailable, since Q2 is never
            built in that case anyway (see query_builder.build_q2).
    """

    prefix_events: List[Any]
    suffix_events: List[Any]
    prefix_ratio: float
    init_places: List[str]
    init_effects: List[Dict[str, Any]]
    final_event_attributes: Dict[str, Any]
    prefix_duration_s: Optional[float]
    full_duration_s: Optional[float]
    warnings: List[str] = field(default_factory=list)
    is_replayable: bool = True
    max_modeled_duration_s: float = 0.0
    reached_within_time: bool = True


def sample_prefix(
    trace: Any,
    replayer: TraceReplayer,
    min_prefix_pct: float = 0.2,
    max_prefix_pct: float = 0.8,
    seed: Optional[int] = None,
    min_prefix_events: int = 1,
) -> Optional[PrefixSample]:
    """Sample a random prefix from a trace and derive its init state via replay.

    Args:
        trace: A pm4py Trace object from the test set.
        replayer: A TraceReplayer built once for the log's domain (same
            PreparedDomainInput/PetriNetModel/config for every test trace).
        min_prefix_pct: Minimum fraction of events to include in the prefix.
        max_prefix_pct: Maximum fraction of events to include in the prefix.
        seed: Optional random seed for reproducibility.
        min_prefix_events: Minimum number of events required in the prefix.
            Returns None if the prefix would be shorter.

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
        outcome = replayer.replay_evaluation_split(trace, n_prefix)
    except Exception as exc:
        logger.warning("Replay failed: %s", exc)
        return None

    prefix_duration_s = _duration_seconds(prefix_events)
    full_duration_s = _duration_seconds(events)
    init_effects = [
        {"attribute": attr, "value": value}
        for attr, value in outcome.split_attributes.items()
    ]

    budget_s = _budget_from_durations(full_duration_s, prefix_duration_s)
    reached_within_time = (
        outcome.reached_end
        and budget_s is not None
        and outcome.max_modeled_duration_seconds <= budget_s
    )

    return PrefixSample(
        prefix_events=prefix_events,
        suffix_events=suffix_events,
        prefix_ratio=prefix_ratio,
        init_places=sorted(outcome.split_marking),
        init_effects=init_effects,
        final_event_attributes=outcome.expected_final_attributes,
        prefix_duration_s=prefix_duration_s,
        full_duration_s=full_duration_s,
        warnings=outcome.warnings,
        is_replayable=outcome.reached_end and outcome.matches_expected_final_state,
        max_modeled_duration_s=outcome.max_modeled_duration_seconds,
        reached_within_time=reached_within_time,
    )


# ---------------------------------------------------------------------------
# Internal helpers


def _budget_from_durations(
    full_duration_s: Optional[float], prefix_duration_s: Optional[float],
) -> Optional[float]:
    """full_duration_s - prefix_duration_s, or None if either is None.

    Single source of truth for this arithmetic — _remaining_budget below
    (the PrefixSample-based public helper, used by query_builder.py and
    metrics_collector.py) and sample_prefix's own reached_within_time
    computation both call this, so the two can never drift apart.
    """
    if full_duration_s is None or prefix_duration_s is None:
        return None
    return full_duration_s - prefix_duration_s


def _remaining_budget(prefix_sample: PrefixSample) -> Optional[float]:
    """Compute the remaining time budget for the suffix.

    Lives here (not in query_builder.py, which imports it from here) so
    that sample_prefix can also use the same formula (via
    _budget_from_durations) without a circular import: query_builder.py
    already imports PrefixSample from this module.

    Args:
        prefix_sample: A PrefixSample with optional duration fields.

    Returns:
        full_duration_s - prefix_duration_s in seconds, or None if either is None.
    """
    return _budget_from_durations(prefix_sample.full_duration_s, prefix_sample.prefix_duration_s)


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
