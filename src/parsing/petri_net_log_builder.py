from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import pm4py
from pm4py import PetriNet, Marking
from pm4py.objects.powl.obj import Transition

import core_utils as utils
from models import AnalysisConfig, FiringStep, PetriNetLog, TraceExecution

logger = utils.get_logger(__name__)


class PetriNetLogBuilder:
    """
    Builds a PetriNetLog by aligning a pm4py EventLog with token-based replay.

    Each log trace is mapped to a TraceExecution: an ordered sequence of
    FiringSteps where every activated transition (including tau) becomes one step.
    Labeled transitions are paired with the corresponding log event via lockstep
    alignment; tau transitions receive empty attributes {}.

    The marking is simulated step by step so each FiringStep records the exact
    input places that held tokens (from_places) at the moment of firing.

    Traces whose replay fitness falls below config.replay_min_fitness are excluded.

    trans_inputs and trans_outputs must be pre-built by StructureAnalyzer.build_arc_maps()
    and passed in — the builder does not derive them from edges itself.

    Extensibility for optimal alignment (Approach B): override _align_trace to use
    pm4py.conformance_diagnostics_alignments instead of the lockstep assumption.
    The rest of the builder (marking simulation, FiringStep construction) is unchanged.
    """

    def __init__(
        self,
        petrinet: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        trans_inputs: DefaultDict[Transition, Set[PetriNet.Place]],
        trans_outputs: DefaultDict[Transition, Set[PetriNet.Place]],
        silent_transitions: Dict[Transition, str],
        config: AnalysisConfig,
    ) -> None:
        self.petrinet = petrinet
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self._trans_inputs = trans_inputs
        self._trans_outputs = trans_outputs
        self.silent_transitions = silent_transitions
        self.config = config

    def build(self, log: Any) -> PetriNetLog:
        """
        Build a PetriNetLog from a pm4py EventLog.

        Runs token-based replay, filters traces below the fitness threshold, aligns
        each accepted trace with its replay result, and builds TraceExecution objects.

        Args:
            log: A pm4py EventLog.

        Returns:
            PetriNetLog with one TraceExecution per accepted trace.
        """
        self._warn_if_lifecycle_log(log)
        pairs = self._replay_log(log)
        executions = [self._build_execution(trace, result) for trace, result in pairs]
        return PetriNetLog(
            executions=executions,
            net=self.petrinet,
            initial_marking=self.initial_marking,
            final_marking=self.final_marking,
        )

    def _warn_if_lifecycle_log(self, log: Any) -> None:
        """
        Emit a single warning if the log contains lifecycle start events.

        The lockstep alignment in _align_trace assumes one event per labeled
        transition. A full-lifecycle log (start + complete per activity) causes
        start events to be consumed for labeled transitions, misaligning attributes
        and timestamps for all subsequent steps.

        Pass a complete-only log, e.g. filtered with:
            pm4py.filter_event_attribute_values(log, 'lifecycle:transition', ['complete'])

        Args:
            log: A pm4py EventLog.
        """
        for trace in log:
            for event in trace:
                if event.get("lifecycle:transition", "").lower() == "start":
                    logger.warning(
                        "PetriNetLogBuilder: lifecycle start events detected. "
                        "Pass the complete-only log to avoid attribute misalignment. "
                        "Filter with pm4py.filter_event_attribute_values(log, "
                        "'lifecycle:transition', ['complete'])"
                    )
                    return

    def _replay_log(self, log: Any) -> List[Tuple[Any, Dict]]:
        """
        Run token-based replay and return (trace, replay_result) pairs for traces
        that meet the minimum fitness threshold.

        Args:
            log: A pm4py EventLog.

        Returns:
            List of (original_trace, replay_result) pairs above the fitness threshold.
        """
        logger.info("Running token-based replay to build PetriNetLog...")
        replayed = pm4py.conformance_diagnostics_token_based_replay(
            log, self.petrinet, self.initial_marking, self.final_marking,
            opt_parameters={"return_object_names": False},
        )
        total = len(replayed)
        skipped = 0
        pairs = []
        for trace, result in zip(log, replayed):
            fitness = result.get("trace_fitness", 0.0)
            if fitness < self.config.replay_min_fitness:
                skipped += 1
            else:
                pairs.append((trace, result))
        if skipped > 0:
            skip_pct = 100.0 * skipped / total if total > 0 else 0.0
            logger.warning(
                f"PetriNetLog: skipped {skipped}/{total} traces ({skip_pct:.1f}%) "
                f"with fitness < {self.config.replay_min_fitness}"
            )
        else:
            logger.info(f"PetriNetLog: all {total} traces accepted.")
        return pairs

    def _build_execution(self, trace: Any, replay_result: Dict) -> TraceExecution:
        """
        Build a TraceExecution from a single trace and its replay result.

        Aligns the trace events with activated_transitions via lockstep, then
        simulates the token marking to determine from_places for each firing.

        Args:
            trace: A pm4py Trace (iterable of Event dicts).
            replay_result: Token replay result dict with 'activated_transitions'.

        Returns:
            TraceExecution with one FiringStep per activated transition.
        """
        trace_id = trace.attributes.get("concept:name", "unknown")
        activated_transitions = replay_result.get("activated_transitions", [])
        aligned = self._align_trace(trace, activated_transitions)

        steps: List[FiringStep] = []
        marking: Dict[PetriNet.Place, int] = dict(self.initial_marking)

        for transition, attributes in aligned:
            from_places = {
                p for p in self._trans_inputs[transition]
                if marking.get(p, 0) > 0
            }

            for p in self._trans_inputs[transition]:
                tokens = marking.get(p, 0)
                if tokens == 1:
                    del marking[p]
                elif tokens > 1:
                    marking[p] = tokens - 1

            for p in self._trans_outputs[transition]:
                marking[p] = marking.get(p, 0) + 1

            is_tau = transition.label is None
            activity_name = (
                utils.sanitize_name(transition.label)
                if not is_tau
                else self.silent_transitions.get(transition, f"tau_unknown_{id(transition)}")
            )

            steps.append(FiringStep(
                transition=transition,
                activity_name=activity_name,
                is_tau=is_tau,
                from_places=from_places,
                attributes=attributes,
            ))

        return TraceExecution(trace_id=trace_id, steps=steps)

    def _align_trace(
        self,
        trace: Any,
        activated_transitions: List[Transition],
    ) -> List[Tuple[Transition, Dict[str, Any]]]:
        """
        Lockstep alignment between replay transitions and log events.

        Labeled transitions consume the next log event and carry its attributes.
        Tau transitions receive empty attributes {}.

        This method is the single extension point for Approach B: override it to
        use pm4py.conformance_diagnostics_alignments (move_both / move_model /
        move_log) for exact alignment on low-fitness traces.

        Args:
            trace: A pm4py Trace (iterable of Event dicts).
            activated_transitions: Ordered list of fired transitions from replay.

        Returns:
            List of (transition, attributes) pairs, one per activated transition.
        """
        event_iter = iter(trace)
        result: List[Tuple[Transition, Dict[str, Any]]] = []
        for transition in activated_transitions:
            if transition.label is not None:
                try:
                    attrs = dict(next(event_iter))
                except StopIteration:
                    attrs = {}
            else:
                attrs = {}
            result.append((transition, attrs))
        return result
