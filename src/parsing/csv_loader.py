"""CSV event log loading utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog

STANDARD_ALIASES: Dict[str, List[str]] = {
    "case_id": [
        "case:concept:name", "case_id", "caseid", "caseID", "case",
        "Case ID", "trace_id", "traceid", "TraceId", "Case_ID",
    ],
    "activity": [
        "concept:name", "activity", "Activity", "activity_name",
        "ActivityName", "task", "Task", "event", "Event",
    ],
    "timestamp": [
        "time:timestamp", "timestamp", "Timestamp", "datetime",
        "start_time", "StartTime", "time", "Date", "date",
    ],
    "lifecycle": [
        "lifecycle:transition", "lifecycle", "life_cycle",
        "transition", "event_type", "Lifecycle",
    ],
}

REQUIRED_FIELDS = ["case_id", "activity"]


def detect_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    """Auto-detect mapping from CSV columns to standard pm4py field names.

    Args:
        columns: List of column names from the CSV file.

    Returns:
        Dict mapping each standard field name to the detected column, or None
        if not found. Keys: case_id, activity, timestamp, lifecycle.
    """
    columns_lower = {col.lower(): col for col in columns}
    mapping: Dict[str, Optional[str]] = {}

    for field, aliases in STANDARD_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in columns:
                found = alias
                break
            if alias.lower() in columns_lower:
                found = columns_lower[alias.lower()]
                break
        mapping[field] = found

    return mapping


def csv_to_event_log(csv_path: str, mapping: Dict[str, Optional[str]]) -> EventLog:
    """Load a CSV file and convert it to a pm4py EventLog.

    Column existence is validated here (format-specific, pre-conversion).
    Type and null checks are performed by log_validator.validate_event_log()
    after conversion, unified with XES validation.

    Args:
        csv_path: Path to the CSV file.
        mapping: Dict mapping standard field names to CSV column names.
                 Required keys: case_id, activity, timestamp.
                 Optional key: lifecycle (None or empty string = not present).

    Returns:
        pm4py EventLog.

    Raises:
        ValueError: If required mapping fields are missing or columns not found.
    """
    for field in REQUIRED_FIELDS:
        if not mapping.get(field):
            raise ValueError(
                f"Required column mapping missing: '{field}'. "
                "Specify which CSV column maps to this field."
            )

    df = pd.read_csv(csv_path)
    existing_columns = set(df.columns)

    for field in REQUIRED_FIELDS:
        col = mapping[field]
        if col not in existing_columns:
            raise ValueError(
                f"Column '{col}' (mapped to '{field}') not found in CSV. "
                f"Available columns: {sorted(existing_columns)}"
            )

    rename_map: Dict[str, str] = {
        mapping["case_id"]: "case:concept:name",
        mapping["activity"]: "concept:name",
    }

    timestamp_col = mapping.get("timestamp") or None
    if timestamp_col and timestamp_col in existing_columns:
        rename_map[timestamp_col] = "time:timestamp"

    lifecycle_col = mapping.get("lifecycle") or None
    if lifecycle_col and lifecycle_col in existing_columns:
        rename_map[lifecycle_col] = "lifecycle:transition"

    df = df.rename(columns=rename_map)

    sort_cols = ["case:concept:name"]
    if "time:timestamp" in df.columns:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
        sort_cols.append("time:timestamp")

    df = df.sort_values(sort_cols)

    return pm4py.convert_dataframe_to_event_log(df)


def load_mapping_file(log_path: str) -> Dict[str, Optional[str]]:
    """Load the column mapping JSON file associated with a CSV log.

    Args:
        log_path: Path to the CSV log file.

    Returns:
        Mapping dict.

    Raises:
        FileNotFoundError: If the .mapping.json file does not exist.
    """
    mapping_path = Path(log_path).with_suffix(".mapping.json")
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"No column mapping found for '{log_path}'. "
            "A .mapping.json file is required alongside the CSV log. "
            "Run the pipeline once from the setup page to save the mapping."
        )
    with open(mapping_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
