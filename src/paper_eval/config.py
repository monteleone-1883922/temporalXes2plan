"""Configuration for the paper evaluation suite (see TIMING_PLAN.md).

Logs come from ERamaM/ProcessMiningDatasets — the same XES versions used by
the Rama-Maneiro et al. (IEEE TSC 2023) benchmark — pinned to a commit and
checked against a sha256 so every run uses byte-identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ERAMAM_COMMIT = "b6f4acbca242b036c21384835922a4a931974937"
_ERAMAM_RAW = f"https://raw.githubusercontent.com/ERamaM/ProcessMiningDatasets/{_ERAMAM_COMMIT}/XES"


@dataclass(frozen=True)
class LogSpec:
    """One event log of the evaluation.

    Attributes:
        name: Short id used on the CLI and in output paths.
        file_name: File name (without .gz) in the ERamaM XES folder.
        sha256: Checksum of the downloaded .xes.gz file.
        has_lifecycle: True when the log carries start/complete pairs, so
            activity durations come from lifecycle instead of inter-event gaps.
        ignore_attributes: Event attributes removed from the model for this
            log, on top of AnalysisConfig's default ignored set.
        keep_attributes: Event attributes the two automatic filters (noise,
            unused) must not remove. An attribute cannot be in both lists.
        status_as_activity: True when lifecycle:transition holds the sub-status of
            a status change (BPIC 2013 incidents: "In Progress", "Awaiting Assignment",
            ...), not start/complete. Each event is then an instantaneous change of
            status: the activity becomes "<concept:name>-<lifecycle:transition>" and the
            event a complete, so durations are inter-event times (the time spent in a
            status is the time to the next change). The BPIC 2013 closed-problems log
            comes already in this form.
        local: True for a log that is not downloaded but copied by hand into
            <data_dir>/raw/<file_name> (the learning-factory log). sha256 is then the
            checksum of that .xes file itself, not of a .gz archive.
    """

    name: str
    file_name: str
    sha256: str
    has_lifecycle: bool
    ignore_attributes: Tuple[str, ...] = ()
    keep_attributes: Tuple[str, ...] = ()
    status_as_activity: bool = False
    local: bool = False

    @property
    def url(self) -> str:
        return f"{_ERAMAM_RAW}/{self.file_name}.gz"


LOGS: Dict[str, LogSpec] = {
    spec.name: spec
    for spec in [
        # Diagnose has 146 distinct values, so the noise filter (N2) would drop it, but
        # it is a real clinical attribute that the decision trees do read: keep it.
        LogSpec("sepsis", "SEPSIS.xes",
                "709c52340306415952811b9b9c5dc6bcc8f8d47d583eba39df9a538459dc543a", False,
                keep_attributes=("Diagnose",)),
        LogSpec("road_fines", "road_fine.xes",
                "efa992309da64516f0e9f58e02c0c9d6377ded03db4c393713151bf09fd12097", False),
        LogSpec("bpic12_a", "BPI_Challenge_2012_A.xes",
                "1906e7368de8da2f1a8c500b3d65d27f66cec48799a6235aff0cabe60e190f4d", False),
        LogSpec("bpic12_w", "BPI_Challenge_2012_W.xes",
                "069d7aa1a850cc3f7562cf75872cda7d6987c7402e0a1d2c7bec8f4dbad9a87c", True),
        LogSpec("bpic13_cp", "BPI_Challenge_2013_closed_problems.xes",
                "584c634dff73b336ab813dc39680762e0f3f0214d286817024b4c5e8028147eb", False),
        LogSpec("bpic13_i", "bpi_challenge_2013_incidents.xes",
                "6ec2136c607648608655802486df97a866f351668751b7ecbca51e1f8715d522", False,
                status_as_activity=True),
        LogSpec("bpic17", "BPI_2017.xes",
                "183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249", True),
        LogSpec("helpdesk", "Helpdesk.xes",
                "daa36a75933e4c032a4f21ce6e4f583563f65cb79ce0d3976b8b2c17c98a0a79", False),
        # The learning-factory log (RQ2 runs only on it). Provided by hand. Its event
        # attributes are copies of the activity and the resource, the BPMN element id and
        # simulator ids/costs, none of them a decision of the process: all left out, so
        # the net is control flow plus durations.
        LogSpec("factory", "fabbrica_2vgr_normalizzato.xes",
                "9011045a56c206acbff80a4998cc23239efac2a0bef82eb7699a670f8ba5f856", True,
                ignore_attributes=("Activity", "Resource", "elementId", "resourceCost", "resourceId"),
                local=True),
    ]
}

# Inclusive (low, high) prefix-length ranges; high=None means open-ended.
Bucket = Tuple[int, Optional[int]]

# No ranges: every prefix length k is its own bucket ("2", "3", ..., "37"), decided on
# 2026-11-09. A log can still group lengths with buckets_by_log (see bucket_label).
DEFAULT_BUCKETS: List[Bucket] = []


@dataclass
class PaperEvalConfig:
    """Parameters shared by every phase of the paper evaluation.

    buckets_by_log overrides DEFAULT_BUCKETS for a single log, so buckets can
    be tuned once the test-trace length distribution of each log is known.
    """

    test_pct: float = 0.3  # last 30% of cases (by start time) -> TEST
    val_pct: float = 0.2  # last 20% of TRAIN -> VAL, scores the TPE trials
    # Take the TRAIN cases still running when the test period starts out of TRAIN
    # (paper §5.1, "after removing cases that span the split point"). Off: decided on
    # 2026-11-09, because on some logs it removes most of the training cases (BPIC13: 596 of
    # 695 closed problems, 3,836 of 5,280 incidents). Part of the split hash.
    drop_spanning_train: bool = False
    min_trace_len: int = 3  # traces with fewer control-flow events are dropped
    k_min: int = 2  # shortest prefix generated
    # No longer used for the goals (the deadline is now min(variant mean - elapsed,
    # real remaining time), see prefixes.py), but part of the split hash: removing
    # it would rename every model folder and redo the TPE, so it stays for now.
    deadline_quantiles: Tuple[float, ...] = (0.5, 0.75, 0.9)
    # Deadline rule of the "deadline" goal (see prefixes.py). False: min(variant
    # mean - elapsed, real remaining time). True: always variant mean - elapsed
    # (real remaining time only when the variant is not in TRAIN). Not part of
    # the split hash: it only changes the prefixes phase.
    deadline_variant_only: bool = False
    buckets_by_log: Dict[str, List[Bucket]] = field(default_factory=dict)
    seed: int = 42  # TPE seed
    search_n_trials: int = 30  # TPE trials
    # Rung 0 of the TPE search: trials where one transition's effect groups pass
    # this are rejected before they can exhaust memory (None = no limit).
    max_effect_groups: Optional[int] = 4096
    # The first event of a case is input data, not modelled behaviour: its
    # attributes seed the initial state and its effects are not mined (see
    # AnalysisConfig.initial_event_as_input).
    initial_event_as_input: bool = True
    # The two automatic attribute filters (see AnalysisConfig): noise (statistics
    # of each attribute) and unused (written but never read by a guard).
    noise_filter: bool = True
    unused_attribute_filter: bool = True
    # Phase "plan" (OPTIC on every instance). The values are starting points to
    # be reviewed per log before the definitive runs; *_by_log override them for one log.
    plan_timeout_s: int = 60  # wall-clock limit of one OPTIC run
    plan_memory_mb: int = 4000  # virtual memory limit of one OPTIC run (ulimit -v)
    plan_timeout_by_log: Dict[str, int] = field(default_factory=dict)
    plan_memory_by_log: Dict[str, int] = field(default_factory=dict)
    # Phase "rq2" (OPTIC optimising total-time, runs only on the factory log). Provisional,
    # like the plan_* values: the run limit is decided later.
    rq2_timeout_s: int = 300
    rq2_memory_mb: int = 4000
    # Every action of the PDDL domain costs exactly 1 (instead of the -log(probability)
    # costs of the paper's §4.3 encoding): what the RQ1 tests use. It changes domain.pddl,
    # so it is recorded in model_meta.json and a model built with the other value is refused.
    unit_costs: bool = False
    # POWL, as in the paper (§4.1): sound models by construction and fewer silent
    # transitions than the Inductive Miner, which keeps the planning domain small.
    discovery_algorithm: str = "powl"
    coverage_percentage: float = 0.001  # variant coverage filter, as in Pipeline
    data_dir: Path = _REPO_ROOT / "data" / "paper_eval"  # prepared logs
    results_dir: Path = _REPO_ROOT / "results" / "paper_eval"  # split/model/prefixes

    def buckets_for(self, log_name: str) -> List[Bucket]:
        return self.buckets_by_log.get(log_name, DEFAULT_BUCKETS)

    def timeout_for(self, log_name: str) -> int:
        """OPTIC wall-clock limit (seconds) for this log."""
        return self.plan_timeout_by_log.get(log_name, self.plan_timeout_s)

    def memory_for(self, log_name: str) -> int:
        """OPTIC virtual memory limit (MB) for this log."""
        return self.plan_memory_by_log.get(log_name, self.plan_memory_mb)


def bucket_label(k: int, buckets: List[Bucket]) -> Optional[str]:
    """Label of the bucket containing prefix length k, or None when no bucket covers k.

    With no buckets at all (DEFAULT_BUCKETS) every length is its own bucket: 5 -> "5".
    With ranges, e.g. [(2, 3), (11, 15), (21, None)]: 3 -> "2-3", 13 -> "11-15",
    30 -> "21+", 5 -> None; the first matching range wins."""
    if not buckets:
        return str(k)
    for low, high in buckets:
        if k >= low and (high is None or k <= high):
            if high is None:
                return f"{low}+"
            return str(low) if low == high else f"{low}-{high}"
    return None
