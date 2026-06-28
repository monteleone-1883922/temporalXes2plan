"""Unified event log validation for XES and CSV logs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import datetime
import pandas as pd

import core_utils as utils

logger = utils.get_logger(__name__)

# XES standard lifecycle values per IEEE 1849-2016
_STANDARD_LIFECYCLE_VALUES = frozenset({
    "start", "complete", "assign", "ate_abort", "withdraw",
    "suspend", "resume", "pi_abort", "schedule", "unknown",
})

# Minimum fraction of events that must have a timestamp for the log to be
# considered "timestamped". Below this threshold a warning is raised.
TIMESTAMP_MIN_COVERAGE = 1


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    infos: List[str] = field(default_factory=list)
    missing_timestamp: bool = False
    missing_lifecycle: bool = False


def validate_event_log(log: Any) -> ValidationResult:
    """Validate presence, nulls, and types of standard fields in a pm4py EventLog.

    Checks are grouped by severity:
    - errors   : blocking — pipeline must not start
    - warnings : require user confirmation (missing_timestamp only)
    - infos    : informational, no action required

    Args:
        log: A pm4py EventLog object.

    Returns:
        ValidationResult with categorised messages and shorthand flags.
    """
    result = ValidationResult()

    _check_case_id(log, result)
    _check_concept_name(log, result)
    _check_timestamp(log, result)
    _check_lifecycle(log, result)

    return result


def validate_partial_trace(trace: Any) -> List[str]:
    """Check mandatory fields on a single pm4py Trace for partial-trace replay.

    Checks presence and type for the two blocking fields. Timestamp and lifecycle
    are irrelevant for replay correctness and are not checked here.

    Args:
        trace: A pm4py Trace object.

    Returns:
        List of error messages (empty list = valid).
    """
    errors: List[str] = []

    # case:concept:name — stored as trace attribute "concept:name"
    case_id = trace.attributes.get("concept:name")
    if case_id is None or (isinstance(case_id, str) and not case_id.strip()):
        errors.append(
            "Trace is missing a case identifier (case:concept:name). "
            "Every trace must have a non-empty case ID."
        )
    elif isinstance(case_id, bool):
        errors.append(
            f"case:concept:name is a boolean ({case_id!r}). "
            "Case IDs must be strings or integers."
        )
    elif isinstance(case_id, float):
        errors.append(
            f"case:concept:name is a float ({case_id!r}). "
            "Case IDs must be strings or integers, not floats."
        )

    # concept:name on each event — presence and type
    for i, event in enumerate(trace):
        name = event.get("concept:name")
        if name is None or (isinstance(name, str) and not name.strip()):
            errors.append(
                f"Event {i} is missing 'concept:name' (activity name). "
                "Every event must have a non-empty activity name."
            )
            break
        if not isinstance(name, str):
            # covers bool (subclass of int), int, float
            type_label = "boolean" if isinstance(name, bool) else type(name).__name__
            errors.append(
                f"Event {i} has a non-string 'concept:name': {name!r} ({type_label}). "
                "Activity names must be strings."
            )
            break

    return errors


# ---------------------------------------------------------------------------
# Internal check functions
# ---------------------------------------------------------------------------

def _check_case_id(log: Any, result: ValidationResult) -> None:
    """Check case:concept:name across all traces."""
    total = len(log)
    if total == 0:
        result.errors.append("The log contains no traces.")
        return

    null_count = 0
    float_count = 0
    bool_count = 0

    for trace in log:
        val = trace.attributes.get("concept:name")
        if val is None or (isinstance(val, str) and not val.strip()):
            null_count += 1
        elif isinstance(val, float):
            float_count += 1
        elif isinstance(val, bool):
            bool_count += 1

    if null_count == total:
        result.errors.append(
            "case:concept:name is missing in all traces. "
            "Every trace must have a non-empty case identifier."
        )
    elif null_count > 0:
        result.errors.append(
            f"case:concept:name is null or empty in {null_count}/{total} traces. "
            "Every trace must have a non-empty case identifier."
        )
    elif bool_count == total:
        result.errors.append(
            "case:concept:name contains only boolean values "
            f"(e.g. {next(iter(log)).attributes.get('concept:name')!r}). "
            "Case IDs must be strings or integers, not booleans — "
            "check that the correct column is mapped to case_id."
        )
    elif float_count == total:
        result.errors.append(
            "case:concept:name contains only floating-point values "
            f"(e.g. {next(iter(log)).attributes.get('concept:name')}). "
            "Case IDs must be strings or integers, not floats — "
            "check that the correct column is mapped to case_id."
        )


def _check_concept_name(log: Any, result: ValidationResult) -> None:
    """Check concept:name (activity name) across all events."""
    total_events = 0
    null_count = 0
    bad_type_count = 0  # non-string: int, float, bool (bool is subclass of int)

    for trace in log:
        for event in trace:
            total_events += 1
            val = event.get("concept:name")
            if val is None or (isinstance(val, str) and not val.strip()):
                null_count += 1
            elif not isinstance(val, str):
                bad_type_count += 1

    if total_events == 0:
        result.errors.append("The log contains no events.")
        return

    if null_count == total_events:
        result.errors.append(
            "concept:name (activity name) is missing in all events. "
            "Every event must have a non-empty activity name."
        )
    elif null_count > 0:
        result.errors.append(
            f"concept:name is null or empty in {null_count}/{total_events} events. "
            "Every event must have a non-empty activity name."
        )
    elif bad_type_count == total_events:
        result.errors.append(
            "concept:name contains only non-string values (numeric or boolean). "
            "Activity names must be strings — "
            "check that the correct column is mapped to activity."
        )


def _check_timestamp(log: Any, result: ValidationResult) -> None:
    """Check time:timestamp coverage and type across all events."""


    total_events = 0
    present_count = 0
    bad_type_count = 0

    for trace in log:
        for event in trace:
            total_events += 1
            val = event.get("time:timestamp")
            if val is None:
                continue
            if isinstance(val, (datetime.datetime, datetime.date)):
                present_count += 1
            else:
                # Attempt string parse as a safety net (mainly relevant for CSV
                # paths where pm4py conversion may not have run yet)
                try:
                    pd.to_datetime(val)
                    present_count += 1
                except Exception:
                    bad_type_count += 1

    if total_events == 0:
        return

    if present_count == 0:
        result.warnings.append(
            "time:timestamp is missing in all events. "
            "Durative actions and the temporal planner (Optic) will not be available."
        )
        result.missing_timestamp = True
    elif bad_type_count > 0 and present_count == 0:
        result.warnings.append(
            f"time:timestamp is present but not parseable as a date in "
            f"{bad_type_count}/{total_events} events. "
            "Durative actions and the temporal planner (Optic) will not be available."
        )
        result.missing_timestamp = True
    else:
        coverage = present_count / total_events
        if coverage < TIMESTAMP_MIN_COVERAGE:
            result.warnings.append(
                f"time:timestamp is present in only {present_count}/{total_events} events "
                f"({coverage:.0%} coverage, minimum required: {TIMESTAMP_MIN_COVERAGE:.0%}). "
                "Durative actions and the temporal planner (Optic) will not be available."
            )
            result.missing_timestamp = True


def _check_lifecycle(log: Any, result: ValidationResult) -> None:
    """Check lifecycle:transition presence and value validity."""
    total_events = 0
    present_count = 0
    nonstandard: set = set()

    for trace in log:
        for event in trace:
            total_events += 1
            val = event.get("lifecycle:transition")
            if val is None:
                continue
            present_count += 1
            str_val = str(val).lower()
            if str_val not in _STANDARD_LIFECYCLE_VALUES:
                nonstandard.add(str(val))

    if total_events == 0:
        return

    if present_count == 0:
        result.infos.append(
            "lifecycle:transition not found — all events will be treated as 'complete'."
        )
        result.missing_lifecycle = True
    elif nonstandard:
        result.infos.append(
            f"lifecycle:transition contains non-standard values: "
            f"{sorted(nonstandard)}. "
            "Expected XES standard values: "
            f"{sorted(_STANDARD_LIFECYCLE_VALUES)}."
        )
