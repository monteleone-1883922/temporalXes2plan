"""Phase "prefixes": every prefix k = k_min … n-1 of every TEST trace, its
ground truth, its goals, and the deduplicated instances every method is
evaluated on.

Goals per prefix:
  - "completion": predict the suffix and the remaining time;
  - "deadline_q<NN>": will the case end within T_q? The deadline handed to
    a method is d = T_q - elapsed (time already spent at the cut), the label
    is case_duration <= T_q. When d <= 0 the deadline has already expired
    at the cut: the row is flagged expired and reported separately.

Deduplication (decision D1): two occurrences are the same instance when
they have the same prefix activity sequence, the same initial state of the
planning problem (marking + discretized attributes, from replaying the
prefix on the TRAIN model) and the same goal (for deadline goals, the same
deadline at the resolution it is written into the PDDL problem). Each
instance counts once in the metrics: a method's score on an instance is the
mean over the ground truths of the cases it groups.

Times are in seconds; *_days fields are the same value in days.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pm4py.objects.log.obj import Trace

from evaluation.trace_sampler import prefix_sample_from_outcome, PrefixSample
from paper_eval.config import Bucket, bucket_label
from paper_eval.split import case_id, control_flow_view
from replay import EvaluationReplayOutcome

logger = logging.getLogger(__name__)

DAY_S = 86400.0
COMPLETION_GOAL = "completion"




@dataclass
class PrefixRecord:
    """One (prefix, goal) of a TEST trace: what the methods are given
    (prefix, initial state, deadline) and the ground truth to compare with.

    A prefix has one record per goal, all sharing the same prefix fields.
    Times are in seconds unless the name says days.
    """

    # Identity of the prefix.
    case_id: str
    k: int  # prefix length (number of control-flow events)
    n: int  # length of the whole trace
    bucket: Optional[str]  # label of the bucket containing k (None if none does)
    prefix_acts: List[str]
    # Ground truth from the real trace.
    true_suffix_acts: List[str]
    elapsed_s: float  # time spent when the prefix is cut
    true_rt_s: float  # real remaining time
    true_rt_days: float
    case_duration_s: float
    # Replay of the trace on the TRAIN model.
    replay_ok: bool  # False: replay failed, the state fields below are None
    replay_error: Optional[str]
    init_places: Optional[List[str]] = None  # marking at the cut
    init_effects: Optional[List[Dict[str, Any]]] = None  # [{"attribute", "value"}, ...]
    reached_end: Optional[bool] = None  # the real suffix can be walked to the end
    is_replayable: Optional[bool] = None  # ... and ends in the real final state
    min_modeled_duration_s: Optional[float] = None
    max_modeled_duration_s: Optional[float] = None
    reached_within_time: Optional[bool] = None
    # Goal. Default = "completion": no deadline, no label.
    goal: str = COMPLETION_GOAL
    deadline_s: Optional[float] = None  # T_q - elapsed (deadline goals only)
    label_feasible: Optional[bool] = None  # case_duration <= T_q (deadline goals only)
    expired: bool = False  # deadline already past at the cut
    # Set by group_instances: the instance this record belongs to.
    group_id: Optional[int] = None

    def set_initial_state(self, sample: PrefixSample, k: int, outcomes: dict[int, EvaluationReplayOutcome]):
        self.init_places = sample.init_places
        self.init_effects = sample.init_effects
        self.reached_end = outcomes[k].reached_end
        self.is_replayable = sample.is_replayable
        self.min_modeled_duration_s = sample.min_modeled_duration_s
        self.max_modeled_duration_s = sample.max_modeled_duration_s
        self.reached_within_time = sample.reached_within_time

    def instance_key(self, duration_scale_factor: float) -> Tuple:
        """Dedup key: prefix activities + initial planning state + goal."""
        # Lists/dicts become sorted tuples so the key is hashable and independent
        # of ordering. None (replay failed) is a valid key part: two prefixes with
        # the same activities and no init state still group together.
        places = tuple(sorted(self.init_places)) if self.init_places is not None else None
        # init_effects = [{"attribute": "crp", "value": "high"}, ...]
        effects = (
            tuple(sorted((e["attribute"], str(e["value"])) for e in self.init_effects))
            if self.init_effects is not None else None
        )
        # Deadline at PDDL resolution; None for the completion goal.
        deadline = (
            pddl_deadline_key(self.deadline_s, duration_scale_factor)
            if self.deadline_s is not None else None
        )
        return tuple(self.prefix_acts), places, effects, self.goal, deadline


@dataclass
class CaseGroundTruth:
    """What really happened in ONE test case, for the prefix/goal of the
    EvaluationInstance that contains it.

    Several cases can collapse into the same instance (same prefix, same
    initial state, same goal): the methods are run once on the instance, and
    each case contributes its own ground truth here, against which the
    prediction is compared. E.g. cases c1 (suffix [C], remaining time 7d)
    and c2 (suffix [D], 9d) with the same prefix [A, B] give one instance
    with two CaseGroundTruth; a predicted remaining time of 8d is scored
    against both and the scores are averaged.
    """

    case_id: str
    true_suffix_acts: List[str]
    true_rt_s: float
    true_rt_days: float
    elapsed_s: float
    case_duration_s: float
    deadline_s: Optional[float]
    label_feasible: Optional[bool]
    reached_end: Optional[bool]
    is_replayable: Optional[bool]
    reached_within_time: Optional[bool]

    @classmethod
    def from_record(cls, record: PrefixRecord) -> "CaseGroundTruth":
        return cls(**{f.name: getattr(record, f.name) for f in dataclasses.fields(cls)})


@dataclass
class EvaluationInstance:
    """One input on which every method is evaluated, after deduplication.

    It is a group of PrefixRecords with the same prefix activities, the same
    initial planning state and the same goal (see instance_key). The fields
    up to `deadline_s` are what a method is given and are identical for all
    the cases of the group; `cases` holds the ground truth of each case in
    the group, one CaseGroundTruth per case. A method's score on the
    instance is the mean of its scores over `cases`.
    """

    group_id: int
    k: int
    bucket: Optional[str]
    prefix_acts: List[str]
    goal: str
    init_places: Optional[List[str]]
    init_effects: Optional[List[Dict[str, Any]]]
    replay_ok: bool
    expired: bool
    deadline_s: Optional[float]  # at PDDL resolution (see pddl_deadline_key)
    cases: List[CaseGroundTruth] = field(default_factory=list)


def _seconds_between(a: Any, b: Any) -> float:
    return (b["time:timestamp"] - a["time:timestamp"]).total_seconds()


def pddl_deadline_key(deadline_s: float, duration_scale_factor: float) -> float:
    """The deadline as problem_builder writes it into the PDDL problem
    (`at {deadline:.1f}` after dividing by duration_scale_factor), so two
    deadlines that produce the same PDDL problem share the same key."""
    return round(deadline_s / duration_scale_factor, 1)


def prefix_records_for_trace(
    trace: Any,
    replayer: Any,
    thresholds_s: Dict[str, float],
    buckets: List[Bucket],
    k_min: int,
) -> List[PrefixRecord]:
    """One record per (prefix, goal) of trace. Replay failures are recorded
    (replay_ok=False) instead of dropping the prefix: the instance still
    exists for the methods that do not need the model."""
    # --- 1. Which prefixes exist -------------------------------------------
    # Work on the control-flow view (activity executions only), so that k and
    # n count the same events the model was discovered from.
    view = control_flow_view(trace)
    n = len(view)
    # A prefix of k events needs a non-empty suffix, hence k <= n-1.
    # E.g. n=5, k_min=2 -> ks=[2, 3, 4]; a trace with n <= k_min has none.
    ks = list(range(k_min, n))
    if not ks:
        return []
    cid = case_id(trace)
    # The replayer expects a Trace object: rebuild one from the filtered
    # events, keeping the case attributes (concept:name, ...).
    view_trace = Trace(view, attributes=dict(trace.attributes))

    # --- 2. Replay on the TRAIN model --------------------------------------
    # One call for all cut points: the alignment of the whole trace (the
    # expensive part) runs once, then each k only slices it.
    # outcomes = {k: EvaluationReplayOutcome} with the marking and attribute
    # values the model has after the first k events, and whether the rest of
    # the trace can be walked to the end.
    try:
        outcomes = replayer.replay_all_splits(view_trace, ks)
        replay_error = None
    except Exception as exc:  # replay failures must not stop the run
        # Keep the prefixes anyway (outcomes empty): methods that do not use
        # the model (baselines) can still be evaluated on them.
        logger.warning("[%s] replay failed: %s", cid, exc)
        outcomes, replay_error = {}, str(exc)

    # --- 3. Quantities shared by every prefix of this trace ----------------
    activities = [e["concept:name"] for e in view]
    # Total case duration, first to last event of the view (seconds).
    case_duration_s = _seconds_between(view[0], view[-1])
    records: List[PrefixRecord] = []
    for k in ks:
        # Time already spent when the prefix is cut: first event -> k-th event
        # (view[k-1], since the prefix is view[:k]).
        elapsed_s = _seconds_between(view[0], view[k - 1])
        # --- 4. The "completion" record of this (trace, k) ------------------
        # Ground truth (true_suffix_acts, true_rt_*) comes from the real trace;
        # replay_ok says whether the init state below could be computed.
        # The state and goal fields keep their defaults (None / completion).
        record = PrefixRecord(
            case_id=cid,
            k=k,
            n=n,
            bucket=bucket_label(k, buckets),
            prefix_acts=activities[:k],
            true_suffix_acts=activities[k:],
            elapsed_s=elapsed_s,
            true_rt_s=case_duration_s - elapsed_s,
            true_rt_days=(case_duration_s - elapsed_s) / DAY_S,
            case_duration_s=case_duration_s,
            replay_ok=k in outcomes,
            replay_error=replay_error,
        )
        if k in outcomes:
            # --- 5. Initial state of the planning problem ----------------
            # prefix_sample_from_outcome turns the replay outcome of cut k
            # into: init_places (marking), init_effects (discretized
            # attribute values), and replayability/duration checks of the
            # real suffix on the model (see trace_sampler.PrefixSample).
            # If the replay failed, these stay None.
            sample = prefix_sample_from_outcome(view, k, outcomes[k])
            record.set_initial_state(sample, k, outcomes)

        # --- 6. One record per goal, all copying the prefix fields -------
        # completion: no deadline, no feasible/unfeasible label.
        records.append(record)
        # deadline goals, one per threshold T_q (name e.g. "q50"):
        #   deadline handed to the methods = T_q - elapsed
        #   label = did the real case finish within T_q?
        #   expired = the deadline is already past at the cut (<= 0)
        # E.g. T_q=5d, elapsed=1d -> deadline 4d; case lasted 8d -> label False.
        for name, t_q in thresholds_s.items():
            deadline_s = t_q - elapsed_s
            records.append(dataclasses.replace(
                record,
                goal=f"deadline_{name}",
                deadline_s=deadline_s,
                label_feasible=case_duration_s <= t_q,
                expired=deadline_s <= 0,
            ))
    return records



def group_instances(
    records: List[PrefixRecord], duration_scale_factor: float,
) -> List[EvaluationInstance]:
    """Assign record.group_id in place and return one instance per group.

    The instance keeps the fields shared by all its cases (identical by
    construction of the key) and, in `cases`, the ground truth of each case;
    its deadline is the PDDL-resolution value the cases share.
    """
    groups: Dict[Tuple, EvaluationInstance] = {}
    for record in records:
        key = record.instance_key(duration_scale_factor)
        # First record with this key: create the instance, taking the shared
        # fields from it (the first case decides them).
        if key not in groups:
            groups[key] = EvaluationInstance(
                group_id=len(groups),
                k=record.k,
                bucket=record.bucket,
                prefix_acts=record.prefix_acts,
                goal=record.goal,
                init_places=record.init_places,
                init_effects=record.init_effects,
                replay_ok=record.replay_ok,
                expired=record.expired,
                deadline_s=key[4] * duration_scale_factor if key[4] is not None else None,
            )
        # Every record (new or duplicate) adds its case's ground truth to the
        # instance and is linked back to it through group_id.
        instance = groups[key]
        record.group_id = instance.group_id
        instance.cases.append(CaseGroundTruth.from_record(record))
    return list(groups.values())


def prefix_stats(records: List[PrefixRecord], instances: List[EvaluationInstance]) -> Dict[str, Any]:
    """Volume and replayability summary, overall and per bucket."""
    def summarize(recs: List[PrefixRecord], insts: List[EvaluationInstance]) -> Dict[str, Any]:
        # Prefix counts use the completion records only: each prefix has
        # exactly one, while it has one record per deadline goal too.
        completion = [r for r in recs if r.goal == COMPLETION_GOAL]
        by_goal = Counter(i.goal for i in insts)
        expired = Counter(r.goal for r in recs if r.expired)
        return {
            "n_prefixes": len(completion),
            "n_replay_failed": sum(1 for r in completion if not r.replay_ok),
            "n_tail_not_replayable": sum(1 for r in completion if r.replay_ok and not r.reached_end),
            "n_instances_by_goal": dict(by_goal),
            "n_expired_by_goal": dict(expired),
        }

    # Same summary on the whole set and on each bucket's slice.
    recs_by_bucket: Dict[Optional[str], List[PrefixRecord]] = defaultdict(list)
    for r in records:
        recs_by_bucket[r.bucket].append(r)
    insts_by_bucket: Dict[Optional[str], List[EvaluationInstance]] = defaultdict(list)
    for i in instances:
        insts_by_bucket[i.bucket].append(i)
    return {
        "overall": summarize(records, instances),
        "by_bucket": {
            str(b): summarize(recs_by_bucket[b], insts_by_bucket[b]) for b in recs_by_bucket
        },
    }


def generate(
    test_traces: Iterable[Any],
    replayer: Any,
    thresholds_s: Dict[str, float],
    buckets: List[Bucket],
    k_min: int,
    duration_scale_factor: float,
) -> Tuple[List[PrefixRecord], List[EvaluationInstance]]:
    """All records of all TEST traces, plus the deduplicated instances.
    duration_scale_factor comes from the model's domain (see pddl_deadline_key)."""
    records: List[PrefixRecord] = []
    for trace in test_traces:
        records.extend(prefix_records_for_trace(trace, replayer, thresholds_s, buckets, k_min))
    instances = group_instances(records, duration_scale_factor)
    return records, instances


def write_jsonl(rows: Sequence[Any], path: Path) -> None:
    """One JSON object per line; dataclass rows are written as their fields."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            data = dataclasses.asdict(row) if dataclasses.is_dataclass(row) else row
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Rows as plain dicts (the files are for analysis, not for this module)."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
