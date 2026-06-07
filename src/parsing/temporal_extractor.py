import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import core_utils as utils
from models import PetriNetLog

logger = utils.get_logger(__name__)


def _delta_seconds(ts_from: Any, ts_to: Any) -> float:
    """
    Compute (ts_to - ts_from).total_seconds(), handling mixed timezone awareness.

    pm4py may load timestamps as timezone-aware or naive depending on the XES file.
    When both sides of a subtraction differ in awareness, Python raises TypeError.
    This helper strips tzinfo from both sides only when necessary.
    """
    try:
        return (ts_to - ts_from).total_seconds()
    except TypeError:
        if hasattr(ts_from, "replace"):
            ts_from = ts_from.replace(tzinfo=None)
        if hasattr(ts_to, "replace"):
            ts_to = ts_to.replace(tzinfo=None)
        return (ts_to - ts_from).total_seconds()


@dataclass
class ExternalDuration:
    """
    User-supplied duration bounds for a single action.

    These values are used directly as PDDL duration inequality bounds
    without any statistical derivation.

    Attributes:
        min_duration: Minimum duration in seconds.
        max_duration: Maximum duration in seconds.
    """
    min_duration: float
    max_duration: float

    def __post_init__(self) -> None:
        if self.min_duration > self.max_duration:
            raise ValueError(
                f"min_duration ({self.min_duration}) must be <= max_duration ({self.max_duration})"
            )


@dataclass
class ActionDurationStats:
    """
    Unified duration statistics for a single action, suitable for generating
    PDDL durative-action duration constraints.

    For data derived from the log (source 'lifecycle' or 'inter_event'),
    effective_min and effective_max are computed as:
        effective_min = max(observed_min, mean - std_dev)
        effective_max = min(observed_max, mean + std_dev)

    For externally supplied data (source 'external'), effective_min and
    effective_max are set directly from the user-provided bounds; all
    statistical fields remain None.

    Attributes:
        effective_min: Lower bound for the PDDL duration inequality (seconds).
        effective_max: Upper bound for the PDDL duration inequality (seconds).
        source: Origin of the data — 'lifecycle', 'inter_event', or 'external'.
        mean: Mean duration in seconds (log-derived sources only).
        std_dev: Standard deviation in seconds (log-derived sources only).
        observed_min: Smallest raw duration observed in the log (log-derived only).
        observed_max: Largest raw duration observed in the log (log-derived only).
        count: Number of observations used to compute the statistics (log-derived only).
    """
    effective_min: float
    effective_max: float
    source: str
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    observed_min: Optional[float] = None
    observed_max: Optional[float] = None
    count: Optional[int] = None


def _stats_to_duration(
    durations: List[float],
    source: str,
) -> ActionDurationStats:
    """
    Compute ActionDurationStats from a list of observed durations.

    Args:
        durations: Non-empty list of durations in seconds.
        source: Either 'lifecycle' or 'inter_event'.

    Returns:
        ActionDurationStats with effective_min/max derived from the formula
        max(obs_min, mean - std) and min(obs_max, mean + std).
    """
    n = len(durations)
    mean = sum(durations) / n
    variance = sum((d - mean) ** 2 for d in durations) / n
    std_dev = math.sqrt(variance)
    obs_min = min(durations)
    obs_max = max(durations)

    effective_min = max(obs_min, mean - std_dev)
    effective_max = min(obs_max, mean + std_dev)

    # Ensure effective_min <= effective_max even in degenerate cases
    if effective_min > effective_max:
        effective_min, effective_max = obs_min, obs_max

    return ActionDurationStats(
        effective_min=effective_min,
        effective_max=effective_max,
        source=source,
        mean=mean,
        std_dev=std_dev,
        observed_min=obs_min,
        observed_max=obs_max,
        count=n,
    )


class TemporalExtractor:
    """
    Extracts action duration statistics from an XES event log.

    Supports three strategies:
    - Lifecycle-based: uses start/complete event pairs for exact durations.
    - Inter-event-time: estimates durations from gaps between consecutive labeled
      steps in a PetriNetLog (estimate_from_inter_event_times_pn).
    - External: accepts user-provided min/max bounds directly.

    The main entry point is :meth:`extract`, which combines all three strategies
    with configurable priority and fallback behaviour.
    """

    def __init__(self, log: Any = None) -> None:
        """
        Args:
            log: A pm4py EventLog, ideally the full_lifecycle_log (i.e. before
                 filtering to complete-only events) so that start/complete pairs
                 are available for extract_from_lifecycle. Optional when only
                 estimate_from_inter_event_times_pn or from_external are used.
        """
        self.log = log

    # ------------------------------------------------------------------
    # Case 1 — lifecycle start/complete pairs
    # ------------------------------------------------------------------

    def extract_from_lifecycle(self) -> Dict[str, ActionDurationStats]:
        """
        Compute durations from start/complete lifecycle event pairs.

        For each case, pairs every 'start' event with the earliest following
        'complete' event that shares the same concept:name.  Activities without
        at least one valid pair are omitted from the result.

        Requires a raw event log containing start events (passed to __init__).
        Returns an empty dict and logs a warning when no log is available.

        Returns:
            Mapping from sanitized activity name to ActionDurationStats,
            with source='lifecycle'.
        """
        if self.log is None:
            logger.warning("extract_from_lifecycle: no log provided, returning empty dict")
            return {}
        raw: Dict[str, List[float]] = defaultdict(list)
        skipped_invalid = 0
        orphan_complete: Dict[str, int] = defaultdict(int)
        negative_delta: Dict[str, int] = defaultdict(int)

        for trace in self.log:
            # Collect pending start timestamps per activity name within this case
            pending_starts: Dict[str, List[Any]] = defaultdict(list)

            for event in trace:
                lifecycle = event.get("lifecycle:transition", "").lower()
                activity = utils.sanitize_name(event.get("concept:name", ""))
                timestamp = event.get("time:timestamp")

                if not activity or timestamp is None:
                    skipped_invalid += 1
                    continue

                if lifecycle == "start":
                    pending_starts[activity].append(timestamp)
                elif lifecycle == "complete":
                    if pending_starts[activity]:
                        start_ts = pending_starts[activity].pop(0)
                        delta = _delta_seconds(start_ts, timestamp)
                        if delta >= 0:
                            raw[activity].append(delta)
                        else:
                            negative_delta[activity] += 1
                            logger.debug(
                                "lifecycle: negative delta (%.1fs) for '%s', skipped",
                                delta, activity,
                            )
                    else:
                        orphan_complete[activity] += 1

        if skipped_invalid:
            logger.warning(
                "extract_from_lifecycle: skipped %d events with missing activity name or timestamp",
                skipped_invalid,
            )
        if orphan_complete:
            logger.warning(
                "extract_from_lifecycle: %d complete events had no matching start — %s",
                sum(orphan_complete.values()),
                dict(orphan_complete),
            )
        if negative_delta:
            logger.warning(
                "extract_from_lifecycle: %d negative-duration pairs discarded — %s",
                sum(negative_delta.values()),
                dict(negative_delta),
            )

        result: Dict[str, ActionDurationStats] = {}
        for activity, durations in raw.items():
            if durations:
                result[activity] = _stats_to_duration(durations, source="lifecycle")
                logger.debug(
                    "lifecycle: %s — n=%d, mean=%.1fs, eff=[%.1f, %.1f]",
                    activity,
                    result[activity].count,
                    result[activity].mean,
                    result[activity].effective_min,
                    result[activity].effective_max,
                )

        logger.info("extract_from_lifecycle: found data for %d activities", len(result))
        return result

    # ------------------------------------------------------------------
    # Case 3 — inter-event time estimation from PetriNetLog
    # ------------------------------------------------------------------

    def estimate_from_inter_event_times_pn(
        self, petri_net_log: PetriNetLog
    ) -> Dict[str, ActionDurationStats]:
        """
        Estimate durations from elapsed time between consecutive labeled steps.

        For each TraceExecution, considers only labeled (non-tau) steps in order.
        The estimated duration for step i is timestamp[i+1] - timestamp[i].
        The last labeled step of every execution is excluded (no successor).

        Activity names are taken directly from FiringStep.activity_name (already
        sanitized). Tau steps are excluded entirely — they do not consume time
        in the model and carry no timestamps.

        Args:
            petri_net_log: Pre-built PetriNetLog.

        Returns:
            Mapping from sanitized activity name to ActionDurationStats,
            with source='inter_event'.
        """
        raw: Dict[str, List[float]] = defaultdict(list)
        skipped_invalid = 0
        negative_delta: Dict[str, int] = defaultdict(int)

        for execution in petri_net_log.executions:
            labeled = [s for s in execution.steps if not s.is_tau]
            for i in range(len(labeled) - 1):
                ts_current = labeled[i].attributes.get("time:timestamp")
                ts_next = labeled[i + 1].attributes.get("time:timestamp")
                activity = labeled[i].activity_name

                if ts_current is None or ts_next is None:
                    skipped_invalid += 1
                    continue

                delta = _delta_seconds(ts_current, ts_next)
                if delta >= 0:
                    raw[activity].append(delta)
                else:
                    negative_delta[activity] += 1
                    logger.debug(
                        "inter_event_pn: negative gap (%.1fs) for '%s', skipped",
                        delta, activity,
                    )

        if skipped_invalid:
            logger.warning(
                "estimate_from_inter_event_times_pn: skipped %d steps with missing timestamp",
                skipped_invalid,
            )
        if negative_delta:
            logger.warning(
                "estimate_from_inter_event_times_pn: %d negative inter-event gaps discarded — %s",
                sum(negative_delta.values()),
                dict(negative_delta),
            )

        result: Dict[str, ActionDurationStats] = {}
        for activity, durations in raw.items():
            if durations:
                result[activity] = _stats_to_duration(durations, source="inter_event")
                logger.debug(
                    "inter_event_pn: %s — n=%d, mean=%.1fs, eff=[%.1f, %.1f]",
                    activity,
                    result[activity].count,
                    result[activity].mean,
                    result[activity].effective_min,
                    result[activity].effective_max,
                )

        logger.info("estimate_from_inter_event_times_pn: found data for %d activities", len(result))
        return result

    # ------------------------------------------------------------------
    # Case 2 — external (user-supplied) durations
    # ------------------------------------------------------------------

    def from_external(
        self, durations: Dict[str, ExternalDuration]
    ) -> Dict[str, ActionDurationStats]:
        """
        Convert user-supplied duration bounds to ActionDurationStats.

        Only activities whose ExternalDuration passes validation are included.
        All statistical fields (mean, std_dev, observed_min, observed_max, count)
        are left as None because they are not meaningful for externally provided data.

        Args:
            durations: Mapping from activity name to ExternalDuration.  May cover
                       any subset of the activities present in the log.

        Returns:
            Mapping from activity name to ActionDurationStats, with source='external'.
        """
        result: Dict[str, ActionDurationStats] = {}
        for activity, ext in durations.items():
            try:
                # Trigger __post_init__ validation
                ExternalDuration(ext.min_duration, ext.max_duration)
            except ValueError as exc:
                logger.warning("Skipping external duration for '%s': %s", activity, exc)
                continue

            result[activity] = ActionDurationStats(
                effective_min=ext.min_duration,
                effective_max=ext.max_duration,
                source="external",
            )
            logger.debug(
                "external: %s — eff=[%.1f, %.1f]",
                activity,
                ext.min_duration,
                ext.max_duration,
            )

        logger.info("from_external: accepted %d / %d activities", len(result), len(durations))
        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def extract(
        self,
        external_durations: Optional[Dict[str, ExternalDuration]] = None,
        fallback_to_inter_event: bool = True,
        petri_net_log: Optional[PetriNetLog] = None,
    ) -> Dict[str, ActionDurationStats]:
        """
        Build a complete duration map by combining all three strategies.

        Priority per activity (highest to lowest):
          1. external_durations — if the activity is explicitly specified
          2. lifecycle           — if start/complete pairs exist in the log
          3. inter_event         — if fallback_to_inter_event is True

        When petri_net_log is provided, inter-event times are derived from it via
        estimate_from_inter_event_times_pn (already sanitized activity names, no
        lifecycle-event filtering needed). If petri_net_log is None and
        fallback_to_inter_event is True, a warning is logged and inter-event
        estimation is skipped.

        Activities not covered by any strategy are omitted from the result.

        Args:
            external_durations: Optional mapping of activity names to user-supplied
                                 bounds.  May cover any subset of activities.
            fallback_to_inter_event: When True, activities not covered by external
                                     or lifecycle data are estimated from inter-event
                                     timestamps.
            petri_net_log: Optional PetriNetLog; when provided, used for inter-event
                           estimation instead of the raw log.

        Returns:
            Mapping from sanitized activity name to ActionDurationStats.
        """
        lifecycle_stats = self.extract_from_lifecycle()

        inter_event_stats: Dict[str, ActionDurationStats] = {}
        if fallback_to_inter_event:
            if petri_net_log is not None:
                inter_event_stats = self.estimate_from_inter_event_times_pn(petri_net_log)
            else:
                logger.warning(
                    "extract: fallback_to_inter_event=True but no petri_net_log provided; "
                    "inter-event estimation skipped."
                )

        external_stats: Dict[str, ActionDurationStats] = {}
        if external_durations:
            external_stats = self.from_external(external_durations)

        # Merge with priority: external > lifecycle > inter_event
        all_activities = (
            set(lifecycle_stats)
            | set(inter_event_stats)
            | set(external_stats)
        )

        result: Dict[str, ActionDurationStats] = {}
        for activity in all_activities:
            if activity in external_stats:
                result[activity] = external_stats[activity]
            elif activity in lifecycle_stats:
                result[activity] = lifecycle_stats[activity]
            elif activity in inter_event_stats:
                result[activity] = inter_event_stats[activity]

        logger.info(
            "extract: resolved durations for %d activities "
            "(lifecycle=%d, inter_event=%d, external=%d)",
            len(result),
            sum(1 for s in result.values() if s.source == "lifecycle"),
            sum(1 for s in result.values() if s.source == "inter_event"),
            sum(1 for s in result.values() if s.source == "external"),
        )
        return result
