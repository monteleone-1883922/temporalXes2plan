"""Phase 1 of the trace replayer: resolve one trace (full or partial) against
the Petri net into an ordered TraceExecution (FiringStep sequence, tau
transitions included), tolerating low fitness instead of dropping the trace.

This is a dedicated, thin wrapper — parsing/petri_net_log_builder.py is used
by the mining pipeline and is deliberately left untouched (it always drops
low-fitness traces, correct for training data but wrong here: the trace being
replayed is the one asked for, it cannot be discarded). Reuses
PetriNetLogBuilder's private alignment/execution-building methods directly
(_align_trace, _align_trace_via_alignment, _build_execution) since they are
already engine-agnostic and don't depend on the fitness-threshold logic being
bypassed here — see claude_plans/trace_replayer_implementation_plan.md §3.1.
"""
from typing import Any, Dict, List, Set, Tuple

import pm4py
from pm4py import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from pm4py.objects.powl.obj import Transition

from models import AnalysisConfig, TraceExecution
from parsing.petri_net_log_builder import _ALIGNMENT_VARIANTS, PetriNetLogBuilder


def align_single_trace(
    trace: Any,
    petrinet: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    trans_inputs: Dict[Transition, Set[PetriNet.Place]],
    trans_outputs: Dict[Transition, Set[PetriNet.Place]],
    silent_transitions: Dict[Transition, str],
    config: AnalysisConfig,
    is_partial: bool = False,
) -> Tuple[TraceExecution, List[str]]:
    """Resolve tau/marking ambiguity for one trace via config.replay_engine.

    Never drops the trace on low fitness (unlike PetriNetLogBuilder) — only
    warns. The fitness score itself may legitimately be low for a genuinely
    partial trace (it never reaches final_marking), which is expected and
    reflected in the warning message, not treated as a replay failure: Phase 2
    (PDDL-level replay) is the only layer that decides executability here.

    Args:
        trace: A single pm4py Trace (full or partial).
        petrinet, initial_marking, final_marking, trans_inputs, trans_outputs,
            silent_transitions: Same structural inputs as PetriNetLogBuilder.
        config: AnalysisConfig — only replay_engine/replay_alignment_variant/
            replay_min_fitness (for the warning threshold) are read.
        is_partial: True when the trace is known to not reach final_marking
            (casi 2/3) — only changes the wording of the low-fitness warning.

    Returns:
        (execution, warnings) — execution is the resolved FiringStep sequence
        (tau included), warnings is a list of human-readable strings (empty
        when fitness is at/above config.replay_min_fitness).
    """
    # A one-trace EventLog: every pm4py conformance-checking function below
    # operates on a log, not a bare trace, and PetriNetLogBuilder's own
    # methods (_align_trace, _build_execution, ...) are written the same way.
    builder = PetriNetLogBuilder(
        petrinet, initial_marking, final_marking,
        trans_inputs, trans_outputs, silent_transitions, config,
    )
    mini_log = EventLog()
    mini_log.append(trace)

    # Lifecycle (start/complete) annotation, same as PetriNetLogBuilder.build():
    # embeds each complete event's __duration_seconds__ directly so
    # _build_execution can extract it into FiringStep.duration_seconds — needed
    # for replay_evaluation_split's optional deadline check. Without this,
    # duration_seconds would silently stay None for every step. mini_log[0] is
    # reassigned to `trace` so every call below (conformance checking AND the
    # alignment/build_execution calls further down) sees the annotated,
    # complete-only events -- not the original, possibly still-lifecycle-paired
    # trace.
    if builder._has_lifecycle_start_events(mini_log):
        mini_log = builder._filter_and_annotate_lifecycle(mini_log)
        trace = mini_log[0]

    warnings: List[str] = []

    # Which pm4py conformance-checking function runs is entirely driven by
    # config.replay_engine — same dispatch PetriNetLogBuilder uses for the
    # mining pipeline (_resolve_alignment), just without ever dropping the
    # trace on low fitness (see this function's docstring).
    if config.replay_engine == "alignments":
        variant = _ALIGNMENT_VARIANTS.get(config.replay_alignment_variant)
        if variant is None:
            raise ValueError(
                f"Unknown replay_alignment_variant "
                f"{config.replay_alignment_variant!r}; expected one of "
                f"{sorted(_ALIGNMENT_VARIANTS)}"
            )
        result = pm4py.conformance_diagnostics_alignments(
            mini_log, petrinet, initial_marking, final_marking,
            variant_str=variant, ret_tuple_as_trans_desc=True,
        )[0]
        fitness = result.get("fitness", 0.0)
        aligned = builder._align_trace_via_alignment(trace, result.get("alignment", []))
    else:
        result = pm4py.conformance_diagnostics_token_based_replay(
            mini_log, petrinet, initial_marking, final_marking,
            opt_parameters={"return_object_names": False},
        )[0]
        fitness = result.get("trace_fitness", 0.0)
        aligned = builder._align_trace(trace, result.get("activated_transitions", []))

    if fitness < config.replay_min_fitness:
        reason = (
            "expected for a partial trace (it does not reach the final marking)"
            if is_partial else
            "the trace may deviate from the model"
        )
        warnings.append(
            f"Trace fitness {fitness:.3f} is below the configured threshold "
            f"{config.replay_min_fitness} ({reason}). Replaying anyway — "
            f"Phase 2 (PDDL-level replay) decides executability, not fitness."
        )

    execution = builder._build_execution(trace, aligned)
    return execution, warnings
