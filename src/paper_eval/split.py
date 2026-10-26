"""Phase "split": temporal 70/30 split of a prepared log, a temporal
validation set inside the training part (used by the TPE search), and the
deadline thresholds T_q.

All lengths and times are measured on the control-flow view of a trace
(its "complete" events, see control_flow_view), the same view discovery
works on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from paper_eval.prepare import is_control_flow_event, sha256_of


def case_id(trace: Any) -> str:
    return str(trace.attributes.get("concept:name"))


def control_flow_view(trace: Any) -> List[Any]:
    """Events of the trace that are activity executions ("complete" or
    without lifecycle), in order."""
    return [e for e in trace if is_control_flow_event(e)]


def case_duration_seconds(trace: Any) -> float:
    view = control_flow_view(trace)
    return (view[-1]["time:timestamp"] - view[0]["time:timestamp"]).total_seconds()


@dataclass
class SplitResult:
    """Case ids of each part plus the deadline thresholds.

    train_fit_ids + val_ids is the training set (TRAIN) the final network is
    built on; val_ids is the TPE search's held-out set; test_ids is never
    used before the evaluation.
    """

    train_fit_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    deadline_thresholds_s: Dict[str, float]
    params: Dict[str, Any]
    log_sha256: str
    split_hash: str

    @property
    def train_ids(self) -> List[str]:
        return self.train_fit_ids + self.val_ids

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "SplitResult":
        return cls(**data)


def _start_time(trace: Any) -> datetime:
    return control_flow_view(trace)[0]["time:timestamp"]


def chronological_case_ids(traces: Iterable[Any]) -> List[str]:
    """Case ids ordered by the timestamp of their first control-flow event
    (ties broken by case id, so the order is deterministic)."""
    return [case_id(t) for t in sorted(traces, key=lambda t: (_start_time(t), case_id(t)))]


def temporal_split_ids(
    traces: Sequence[Any], test_pct: float, val_pct: float,
) -> Dict[str, List[str]]:
    """First (1 - test_pct) of the cases by start time -> TRAIN, the rest ->
    TEST; inside TRAIN, the last val_pct -> VAL. Training cases still
    running when the test period starts are kept (decision D9)."""
    # E.g. 1000 cases, test_pct=0.3, val_pct=0.2: the 700 oldest are TRAIN
    # (560 fit + 140 val, val being the newest of them), the 300 newest TEST.
    ordered = chronological_case_ids(traces)
    n_train = round(len(ordered) * (1 - test_pct))
    train, test = ordered[:n_train], ordered[n_train:]
    n_fit = round(len(train) * (1 - val_pct))
    return {"train_fit": train[:n_fit], "val": train[n_fit:], "test": test}


def deadline_thresholds(train_traces: Iterable[Any], quantiles: Sequence[float]) -> Dict[str, float]:
    """T_q: the q-quantile of case duration (seconds) over the training
    traces, keyed "q50", "q75", … (linear interpolation, numpy default)."""
    durations = np.array([case_duration_seconds(t) for t in train_traces], dtype=float)
    return {f"q{round(q * 100)}": float(np.quantile(durations, q)) for q in quantiles}


def compute_split_hash(log_sha256: str, params: Dict[str, Any]) -> str:
    # Identifies a split by (input log bytes, split parameters): it names the
    # model directory, so a different log or parameters never reuse a model.
    payload = json.dumps({"log": log_sha256, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_split(
    log: Any, log_path: Path, test_pct: float, val_pct: float, quantiles: Sequence[float],
) -> SplitResult:
    traces = list(log)
    ids = temporal_split_ids(traces, test_pct, val_pct)
    # T_q are computed on TRAIN only (fit + val), never on TEST, so the
    # deadlines do not leak information from the test period.
    train_ids = set(ids["train_fit"]) | set(ids["val"])
    thresholds = deadline_thresholds((t for t in traces if case_id(t) in train_ids), quantiles)
    params = {"strategy": "temporal", "test_pct": test_pct, "val_pct": val_pct,
              "quantiles": list(quantiles)}
    log_sha = sha256_of(Path(log_path))
    return SplitResult(
        train_fit_ids=ids["train_fit"],
        val_ids=ids["val"],
        test_ids=ids["test"],
        deadline_thresholds_s=thresholds,
        params=params,
        log_sha256=log_sha,
        split_hash=compute_split_hash(log_sha, params),
    )


def write_split(split: SplitResult, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split.to_json(), indent=2), encoding="utf-8")


def read_split(path: Path) -> SplitResult:
    return SplitResult.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
