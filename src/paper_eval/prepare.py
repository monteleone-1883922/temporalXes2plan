"""Phase "prepare": download a log and apply the preprocessing shared by every
method of the paper evaluation.

Preprocessing (applied identically to every log, so every method sees the
same events):
  - timestamps converted to UTC (naive timestamps are taken as UTC, the same
    convention as evaluation.trace_sampler._duration_seconds);
  - events of each trace stably sorted by timestamp (ties keep file order);
  - lifecycle:transition lower-cased, and only "start"/"complete" events kept.
    Other transitions (schedule, suspend, resume, ate_abort, withdraw, …)
    are dropped: parsing.log_processor keeps only "complete" events for
    discovery, while parsing.petri_net_log_builder pairs "start" with
    "complete" for durations and would treat any other transition as a
    completion — keeping only these two makes both views agree. Events
    without lifecycle are kept (treated as complete everywhere);
  - traces with fewer than min_trace_len control-flow events are dropped.
Start events are kept on purpose: they carry the activity durations.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
from pm4py.objects.log.obj import Event, EventLog, Trace

from core_utils import _load_log, write_traces_xes
from paper_eval.config import LogSpec

logger = logging.getLogger(__name__)

LIFECYCLE_KEY = "lifecycle:transition"
_KEPT_TRANSITIONS = {"start", "complete"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_log(spec: LogSpec, data_dir: Path, force: bool = False) -> Path:
    """Download spec's .xes.gz, verify its sha256 and decompress it.

    Returns the path of the raw (unprocessed) XES file,
    data_dir/raw/<file_name>. Skips the download when that file already
    exists, unless force.

    Raises:
        ValueError: if the downloaded archive does not match spec.sha256.
    """
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    xes_path = raw_dir / spec.file_name
    if xes_path.exists() and not force:
        return xes_path

    # Download into a temp dir (inside raw_dir), verify, then decompress: a
    # failed download or checksum never leaves a partial XES behind.
    with tempfile.TemporaryDirectory(dir=raw_dir) as tmp:
        gz_path = Path(tmp) / f"{spec.file_name}.gz"
        logger.info("Downloading %s from %s", spec.name, spec.url)
        response = requests.get(spec.url, timeout=300, stream=True)
        response.raise_for_status()
        with open(gz_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        actual = sha256_of(gz_path)
        if actual != spec.sha256:
            raise ValueError(
                f"Checksum mismatch for {spec.name}: expected {spec.sha256}, got {actual}"
            )
        with gzip.open(gz_path, "rb") as src, open(xes_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return xes_path


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def is_control_flow_event(event: Any) -> bool:
    """True for the events the control-flow view keeps: "complete" events,
    and events without lifecycle (expects lifecycles already lower-cased by
    preprocess)."""
    return event.get(LIFECYCLE_KEY, "complete") == "complete"


def preprocess_trace(trace: Trace) -> Trace:
    """Return a new trace: lifecycle normalized and filtered, timestamps in
    UTC, events stably sorted by timestamp."""
    events = []
    for original in trace:
        # Copy, so the input log is not modified.
        event = Event(dict(original))
        if LIFECYCLE_KEY in event:
            transition = str(event[LIFECYCLE_KEY]).lower()
            # Drop schedule/suspend/resume/... (see module docstring).
            if transition not in _KEPT_TRANSITIONS:
                continue
            event[LIFECYCLE_KEY] = transition
        ts = event.get("time:timestamp")
        if isinstance(ts, datetime):
            event["time:timestamp"] = to_utc(ts)
        events.append(event)
    # Python's sort is stable: equal timestamps keep their file order.
    events.sort(key=lambda e: e["time:timestamp"])
    new_trace = Trace(events, attributes=dict(trace.attributes))
    return new_trace


def preprocess(log: EventLog, min_trace_len: int) -> EventLog:
    """Apply the shared preprocessing to every trace (see module docstring).

    Traces without a timestamp on every event are dropped too: every
    temporal quantity of the evaluation needs them.
    """
    kept: List[Trace] = []
    for trace in log:
        # 1) drop traces with a missing timestamp, 2) normalize the rest,
        # 3) drop those that became too short after the lifecycle filter.
        if any(not isinstance(e.get("time:timestamp"), datetime) for e in trace):
            continue
        new_trace = preprocess_trace(trace)
        if sum(1 for e in new_trace if is_control_flow_event(e)) >= min_trace_len:
            kept.append(new_trace)
    return EventLog(kept, attributes=log.attributes)


def log_stats(log: EventLog) -> Dict[str, Any]:
    """Statistics of a preprocessed log, on its control-flow view."""
    # lengths/activities/variants count control-flow events only; lifecycle
    # and n_events count every event (including "start").
    lengths = [sum(1 for e in t if is_control_flow_event(e)) for t in log]
    activities = {e["concept:name"] for t in log for e in t if is_control_flow_event(e)}
    variants = {tuple(e["concept:name"] for e in t if is_control_flow_event(e)) for t in log}
    lifecycle = Counter(str(e.get(LIFECYCLE_KEY, "<none>")) for t in log for e in t)
    durations_days = sorted(
        (t[-1]["time:timestamp"] - t[0]["time:timestamp"]).total_seconds() / 86400 for t in log
    )
    n = len(lengths)
    return {
        "n_traces": n,
        "n_events": sum(len(t) for t in log),
        "n_control_flow_events": sum(lengths),
        "n_activities": len(activities),
        "n_variants": len(variants),
        "lifecycle": dict(lifecycle),
        "trace_length": {
            "min": min(lengths) if n else 0,
            "mean": sum(lengths) / n if n else 0.0,
            "max": max(lengths) if n else 0,
        },
        "case_duration_days": {
            "mean": sum(durations_days) / n if n else 0.0,
            "median": durations_days[n // 2] if n else 0.0,
            "max": durations_days[-1] if n else 0.0,
        },
    }


def prepare_log(spec: LogSpec, data_dir: Path, min_trace_len: int, force: bool = False) -> Path:
    """Download (if needed) and preprocess spec, writing
    data_dir/<name>.xes and data_dir/<name>.stats.json. Returns the XES path."""
    data_dir = Path(data_dir)
    out_path = data_dir / f"{spec.name}.xes"
    if out_path.exists() and not force:
        logger.info("[%s] prepared log already at %s — skipped", spec.name, out_path)
        return out_path

    raw_path = download_log(spec, data_dir, force=force)
    log = _load_log(str(raw_path), "xes", None)
    n_before = len(log)
    prepared = preprocess(log, min_trace_len)
    logger.info("[%s] %d traces kept of %d", spec.name, len(prepared), n_before)

    write_traces_xes(prepared, list(prepared), out_path)
    stats = log_stats(prepared)
    stats["n_traces_raw"] = n_before
    (data_dir / f"{spec.name}.stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out_path
