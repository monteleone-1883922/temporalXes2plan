import json
import logging
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set, Union

import pm4py

from parsing.csv_loader import csv_to_event_log

def get_logger(name: str) -> logging.Logger:
    """Return a logger that inherits its level and handlers from the root logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        logging.Logger instance with propagation enabled.
    """
    return logging.getLogger(name)


def _replace_special_chars(s: str) -> str:
    """Replace non-alphanumeric characters for PDDL name safety.

    Space becomes '_'; any other non-alphanumeric/non-underscore character
    becomes 'ascii{decimal_code}' (e.g. '#' → 'ascii35', '@' → 'ascii64').
    Consecutive underscores are collapsed and leading/trailing underscores
    are stripped.
    """
    parts = []
    for ch in s:
        if ch.isalpha() or ch.isdigit() or ch == '_':
            parts.append(ch)
        elif ch == ' ':
            parts.append('_')
        else:
            parts.append(f"ascii{ord(ch)}")
    result = ''.join(parts)
    result = re.sub(r'_+', '_', result)
    return result.strip('_')


def sanitize_name(name: str) -> str:
    """Sanitize an activity or attribute name for use in PDDL syntax.

    Strips 'case:' prefixes; spaces become '_'; other special characters
    become 'ascii{code}' to preserve uniqueness and avoid empty results.

    ESEMPIO:
        "case:concept:name" -> "conceptascii58name"
        "ER Registration"   -> "er_registration"
        "#"                 -> "ascii35"
    """
    if not name:
        return name
    if name.lower().startswith("case:"):
        name = name[5:]
    name = _replace_special_chars(name.strip().lower().replace(" ", "_")\
        .replace(":", "_")\
        .replace("-", "_")\
        .replace("(", "")\
        .replace(")", "")\
        .replace("/", "_")\
        .replace("\\", "_"))
    return name or "_"


def sanitize_value(attr_name: str, raw_value: Optional[str]) -> Optional[str]:
    """Produce a PDDL constant for an attribute value as {attr}_val_{value}.

    Args:
        attr_name: Already-sanitized attribute name (e.g. "costo").
        raw_value: Raw string value from the log (e.g. "alto", "lte_6_0", "#").

    Returns:
        A valid PDDL constant name, e.g. "costo_val_alto".

    ESEMPIO:
        sanitize_value("costo",    "alto")   -> "costo_val_alto"
        sanitize_value("expense",  "lte_6_0")-> "expense_val_lte_6_0"
        sanitize_value("dismissal","#")      -> "dismissal_val_ascii35"
        sanitize_value("amount",   "2")      -> "amount_val_2"
    """
    if raw_value is None:
        return None
    if raw_value.startswith(f"{attr_name}_val_"):
        return raw_value
    cleaned = _replace_special_chars(str(raw_value).strip().lower())
    cleaned = cleaned or "empty"
    return f"{attr_name}_val_{cleaned}"

def convert_interval_to_lte_gte(interval_str: str) -> str:
    """
    Convert numerical interval range notation into standard lte/gte string format.

    Args:
        interval_str: A string representing interval ranges like "(-inf-6.15]" or "(12.5-inf)".

    Returns:
        A PDDL-compatible string representation.

    ESEMPIO:
        "(-inf-6.15]" -> "lte_6_15"
        "(12.5-inf)" -> "gte_12_5"
        "(10.5-20.0]" -> "gte_10_5_lte_20_0"
    """
    interval = interval_str.strip('()[]')
    if interval.startswith('-inf-'):
        upper_part = interval[5:]  # Remove '-inf-'
        upper_fmt = upper_part.replace('.', '_').replace('-', 'neg')
        return f"lte_{upper_fmt}"
    elif interval.endswith('-inf'):
        lower_part = interval[:-4]  # Remove '-inf'
        lower_fmt = lower_part.replace('.', '_').replace('-', 'neg')
        return f"gte_{lower_fmt}"
    else:
        parts = interval.split('-')
        if len(parts) == 2:
            lower, upper = parts
            lower_fmt = lower.replace('.', '_').replace('-', 'neg')
            upper_fmt = upper.replace('.', '_').replace('-', 'neg')
            return f"gte_{lower_fmt}_lte_{upper_fmt}"
    
    return interval_str

#TODO is there a test for this or at least a review to see if can be improved
def discretize_value(
    attr: Union[str, None], 
    value: Union[float, str, None], 
    intervals: Optional[Dict[str, List[float]]] = None
) -> str:
    """
    Discretize a numeric or interval value based on defined threshold splits.

    Args:
        attr: The attribute name string.
        value: The raw numerical or interval value to discretize.
        intervals: A dictionary mapping attribute names to lists of floats representing thresholds.

    Returns:
        The discretized value string representation.

    ESEMPIO:
        value="(-inf-6.15]", intervals=None -> "lte_6_15"
        value=5.0, intervals={"crp": [10.0, 50.0]} -> "lte_10_0"
        value=25.0, intervals={"crp": [10.0, 50.0]} -> "gte_10_0_lte_50_0"
        value=60.0, intervals={"crp": [10.0, 50.0]} -> "gte_50_0"
    """
    if (isinstance(value, str) and
            (
                '-inf' in value or
                (
                    '-' in value and
                    not value.replace('-', '').replace('.', '').replace('_', '').isalnum()
                )
            )):
        return convert_interval_to_lte_gte(value)
    else:   # Numeric discretization
        if intervals is None or attr not in intervals or not intervals[attr] or value is None:
            return str(value)
        
        try:
            value = float(value)
            thresholds = sorted(intervals[attr])
            
            if value <= thresholds[0]:
                thresh_str = str(thresholds[0]).replace('.', '_').replace('-', 'neg')
                return f'lte_{thresh_str}'
            elif value >= thresholds[-1]:
                thresh_str = str(thresholds[-1]).replace('.', '_').replace('-', 'neg')
                return f'gte_{thresh_str}'
            else:
                for i in range(len(thresholds) - 1):
                    if thresholds[i] < value <= thresholds[i + 1]:
                        lower_str = str(thresholds[i]).replace('.', '_').replace('-', 'neg')
                        upper_str = str(thresholds[i + 1]).replace('.', '_').replace('-', 'neg')
                        return f'gte_{lower_str}_lte_{upper_str}'
            return str(value)
        except (ValueError, TypeError):
            return str(value)


_logger = get_logger(__name__)


def save_original_and_current(
    config_dir: str, data: Dict[str, Any]
) -> Tuple[bool, bool]:
    """Atomically write original.json and current.json if they do not exist.

    Args:
        config_dir: Directory for the configuration (e.g. data/sepsis/).
        data: Serialized Petri net dict to persist.

    Returns:
        Tuple (original_written, current_written) — True when the file was
        created, False when it already existed and was skipped.
    """
    os.makedirs(config_dir, exist_ok=True)

    written: List[bool] = []
    for filename in ("original.json", "current.json"):
        path = os.path.join(config_dir, filename)
        if os.path.exists(path):
            _logger.info("Skipping %s — already exists", path)
            written.append(False)
            continue
        fd, tmp_path = tempfile.mkstemp(dir=config_dir, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
            _logger.info("Written %s", path)
            written.append(True)
        except Exception:
            os.unlink(tmp_path)
            raise

    return (written[0], written[1])


# ---------------------------------------------------------------------------
# Stratified train/test split for event log evaluation and network-quality search
# ---------------------------------------------------------------------------
#
# Shared by evaluation.test_case_selector and network_search — kept here
# (rather than under evaluation/) because it is a consumer of neither: it
# only builds a train/test split of an event log, used both by plan
# evaluation and by the network-improvement search loop, neither of which
# should depend on the other.

@dataclass
class TrainTestSplit:
    """Output of a stratified train/test split.

    Attributes:
        train_path: Path to a temporary XES file containing the training traces.
            Deleted automatically when used as a context manager.
        test_cases: List of pm4py Trace objects for evaluation.
        n_train: Number of traces in the training set.
        n_test: Number of traces in the test set.
    """

    train_path: Path
    test_cases: List[Any]
    n_train: int
    n_test: int

    def __enter__(self) -> "TrainTestSplit":
        return self

    def __exit__(self, *_: Any) -> None:
        if self.train_path.exists():
            self.train_path.unlink()


def split(
    log_path: str,
    log_fmt: str = "xes",
    test_pct: float = 0.2,
    seed: int = 42,
    mapping: Optional[dict] = None,
) -> TrainTestSplit:
    """Load an event log and produce a general-purpose stratified train/test split.

    Test cases are selected via stratified sampling across trace-length deciles
    so the test set covers the full length distribution. Every trace is
    eligible as a test case regardless of length, and the test-set size is a
    plain rounded percentage with no lower/upper clamp — this is the
    general-purpose split (used e.g. by network_search). Callers that need
    clamping and a minimum test-trace length for their own purpose (e.g.
    evaluation's Q1/Q2/Q3 prefix queries, which need a minimum number of
    events to produce a meaningful prefix) should build that on top of this
    function's building blocks (_load_log, _stratified_sample_indices,
    _build_train_test_split) rather than passing extra parameters here — see
    evaluation.test_case_selector.split for that specialized version.

    The number of test traces is computed as:
        n = round(total_traces * test_pct)

    Args:
        log_path: Path to the XES or CSV event log.
        log_fmt: Format of the log file — "xes" or "csv".
        test_pct: Fraction of total traces to use as test set (e.g. 0.2 = 20%).
        seed: Random seed for reproducibility.
        mapping: Column mapping required when log_fmt="csv".  Keys: case_id,
            activity, timestamp (optional), lifecycle (optional).

    Returns:
        TrainTestSplit with train_path, test_cases, n_train, n_test.
    """
    log = _load_log(log_path, log_fmt, mapping)
    traces = list(log)

    n_test_cases = round(len(traces) * test_pct)
    rng = random.Random(seed)
    test_indices = _stratified_sample_indices(traces, n_test_cases, rng)

    return _build_train_test_split(log, traces, test_indices)


def _build_train_test_split(log: Any, traces: List[Any], test_indices: List[int]) -> TrainTestSplit:
    """Shared by every split() variant: given the full trace list and the
    already-selected test indices, write the remaining traces to a temporary
    XES file and assemble the TrainTestSplit.

    Args:
        log: The pm4py EventLog loaded from disk (only its .attributes are used).
        traces: list(log) — every trace, in original order.
        test_indices: Indices into traces selected for the test set.

    Returns:
        TrainTestSplit with train_path, test_cases, n_train, n_test.
    """
    test_set = set(test_indices)
    test_cases = [traces[i] for i in test_indices]
    train_traces = [traces[i] for i in range(len(traces)) if i not in test_set]

    train_log = pm4py.objects.log.obj.EventLog(
        train_traces,
        attributes=log.attributes,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".xes", delete=False)
    tmp.close()
    train_path = Path(tmp.name)
    pm4py.write_xes(train_log, str(train_path))

    return TrainTestSplit(
        train_path=train_path,
        test_cases=test_cases,
        n_train=len(train_traces),
        n_test=len(test_cases),
    )


def _load_log(log_path: str, log_fmt: str, mapping: Optional[dict]) -> Any:
    """Load an XES or CSV event log via pm4py.

    Args:
        log_path: File path.
        log_fmt: "xes" or "csv".
        mapping: Column mapping for CSV files.

    Returns:
        pm4py EventLog object.
    """
    if log_fmt.lower() == "csv":
        return csv_to_event_log(log_path, mapping or {})
    return pm4py.objects.log.importer.xes.importer.apply(log_path)


def _stratified_sample_indices(
    traces: List[Any],
    n_test_cases: int,
    rng: random.Random,
) -> List[int]:
    """Select test-set indices via proportional stratified sampling on trace length deciles.

    Traces are binned into up to 10 decile buckets sorted by length.  Each
    bucket contributes a number of test cases proportional to its size, so the
    test set reflects the actual length distribution of the candidate pool.
    Leftover quota from rounding is assigned to the largest buckets first.

    Args:
        traces: Candidate traces (already filtered for minimum length).
        n_test_cases: Target total number of test traces.
        rng: Seeded Random instance for reproducibility.

    Returns:
        Sorted list of selected trace indices (into the *traces* list).
    """
    n_total = len(traces)
    if n_total == 0:
        return []

    n_select = min(n_test_cases, n_total)

    lengths = [len(t) for t in traces]
    indexed = sorted(enumerate(lengths), key=lambda x: x[1])

    n_buckets = min(10, n_total)
    buckets: List[List[int]] = [[] for _ in range(n_buckets)]
    for rank, (orig_idx, _) in enumerate(indexed):
        bucket_idx = min(rank * n_buckets // n_total, n_buckets - 1)
        buckets[bucket_idx].append(orig_idx)

    # Proportional allocation: each bucket gets floor(n_select * size / n_total).
    raw_alloc = [n_select * len(b) / n_total for b in buckets]
    alloc = [int(a) for a in raw_alloc]
    remainder = n_select - sum(alloc)

    # Distribute leftover slots to buckets with largest fractional parts.
    fracs = sorted(
        range(n_buckets),
        key=lambda i: raw_alloc[i] - alloc[i],
        reverse=True,
    )
    for i in range(remainder):
        alloc[fracs[i]] += 1

    selected: List[int] = []
    for bucket, take in zip(buckets, alloc):
        take = min(take, len(bucket))
        if take > 0:
            selected.extend(rng.sample(bucket, take))

    return sorted(selected)
