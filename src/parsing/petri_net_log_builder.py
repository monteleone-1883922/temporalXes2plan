from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import pm4py
from pm4py import PetriNet, Marking
from pm4py.algo.conformance.alignments.petri_net import algorithm as pn_alignments
from pm4py.objects.powl.obj import Transition

import core_utils as utils
from models import AnalysisConfig, FiringStep, PetriNetLog, TraceExecution

logger = utils.get_logger(__name__)

# config.replay_alignment_variant -> pm4py Variants enum member, used only when
# config.replay_engine == "alignments".
_ALIGNMENT_VARIANTS = {
    "state_equation_a_star": pn_alignments.Variants.VERSION_STATE_EQUATION_A_STAR,
    "dijkstra_no_heuristics": pn_alignments.Variants.VERSION_DIJKSTRA_NO_HEURISTICS,
    "dijkstra_less_memory": pn_alignments.Variants.VERSION_DIJKSTRA_LESS_MEMORY,
    "discounted_a_star": pn_alignments.Variants.VERSION_DISCOUNTED_A_STAR,
}


class PetriNetLogBuilder:
    """
    Builds a PetriNetLog by aligning a pm4py EventLog with the model, via either
    token-based replay or optimal alignments (config.replay_engine).

    Each log trace is mapped to a TraceExecution: an ordered sequence of
    FiringSteps where every activated transition (including tau) becomes one step.
    Labeled transitions are paired with the corresponding log event; tau
    transitions receive empty attributes {}.

    - "token_based_replay" (default): greedy replay, fast, well suited to whole-log
      replay. Transitions are paired to log events via lockstep alignment
      (_align_trace).
    - "alignments": pm4py.conformance_diagnostics_alignments, an optimal
      A*/Dijkstra search (config.replay_alignment_variant selects the variant).
      Slower, but resolves each move exactly — including distinguishing between
      multiple silent transitions, which lockstep alignment cannot do — via
      ret_tuple_as_trans_desc=True (_align_trace_via_alignment).

    The marking is simulated step by step so each FiringStep records the exact
    input places that held tokens (from_places) at the moment of firing.

    Lifecycle-aware: if the log contains start/complete event pairs, the builder
    automatically filters to complete-only events for replay and injects the
    start→complete duration into FiringStep.duration_seconds for each labeled step.
    The duration is embedded directly on each complete event before replay, so no
    external mapping structures are needed. Passing a complete-only log is also
    supported — duration_seconds will be None for all steps.

    Traces whose replay fitness falls below config.replay_min_fitness are excluded.

    trans_inputs and trans_outputs are available directly from PetriNetModel
    (built by ModelDiscoverer.discover()) and must be passed in at construction.
    """

    def __init__(
        self,
        petrinet: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        trans_inputs: Dict[Transition, Set[PetriNet.Place]],
        trans_outputs: Dict[Transition, Set[PetriNet.Place]],
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

        Automatically detects lifecycle logs (start + complete events). When
        detected, start events are removed and start→complete durations are
        embedded directly on each complete event before replay. Fitness filtering
        is applied after this step, so only accepted traces carry durations.

        Args:
            log: A pm4py EventLog. May be full lifecycle (start + complete) or
                 complete-only. Both forms are handled transparently.

        Returns:
            PetriNetLog with one TraceExecution per accepted trace.
        """
        replay_log = log
        if self._has_lifecycle_start_events(log):
            logger.info(
                "PetriNetLogBuilder: lifecycle log detected — "
                "filtering to complete events and annotating durations."
            )
            replay_log = self._filter_and_annotate_lifecycle(log)

        pairs = self._resolve_alignment(replay_log)
        executions = [self._build_execution(trace, aligned) for trace, aligned in pairs]
        return PetriNetLog(
            executions=executions,
            net=self.petrinet,
            initial_marking=self.initial_marking,
            final_marking=self.final_marking,
        )

    @staticmethod
    def _has_lifecycle_start_events(log: Any) -> bool:
        """Return True if any event in the log carries lifecycle:transition == 'start'."""
        for trace in log:
            for event in trace:
                if event.get("lifecycle:transition", "").lower() == "start":
                    logger.debug("lifecycle: detected start event in log")
                    return True
        return False

    def _filter_and_annotate_lifecycle(self, log: Any) -> Any:
        """
        Single-pass filter and duration annotation for lifecycle logs.

        For each trace, scans events in order:
        - start events: the timestamp is recorded in a pending queue per activity
          and the event is excluded from the output.
        - complete events (or events without lifecycle:transition): included in
          the output. If a matching start timestamp exists for the same activity,
          the start→complete delta is attached directly on the event copy as
          __duration_seconds__. Negative deltas are discarded.
        - Events without a lifecycle attribute are treated as complete.

        Each complete event in the returned log carries its own duration as an
        attribute, so no external mapping structure is needed. The original log
        is never modified — a shallow copy of each event dict is made.

        Args:
            log: A pm4py EventLog containing lifecycle start/complete events.

        Returns:
            A new EventLog with start events removed and __duration_seconds__
            injected on complete events where a matching start was found.
        """
        filtered = type(log)()
        filtered.attributes.update(log.attributes)
        negative_count = 0

        for trace in log:
            new_trace = type(trace)()
            new_trace.attributes.update(trace.attributes)
            pending: Dict[str, List[Any]] = defaultdict(list)

            for event in trace:
                lifecycle = event.get("lifecycle:transition", "complete").lower()
                activity = event.get("concept:name", "")
                timestamp = event.get("time:timestamp")

                if lifecycle == "start":
                    if activity and timestamp is not None:
                        pending[activity].append(timestamp)
                    continue  # start events are never included in the filtered log

                # Complete event (or event without lifecycle attribute)
                annotated = dict(event)
                if activity and timestamp is not None and pending[activity]:
                    start_ts = pending[activity].pop(0)
                    delta = self._delta_seconds(start_ts, timestamp)
                    if delta >= 0:
                        annotated["__duration_seconds__"] = delta
                    else:
                        negative_count += 1
                        logger.debug(
                            "lifecycle: negative delta (%.1fs) for '%s', skipped",
                            delta, activity,
                        )
                new_trace.append(annotated)

            filtered.append(new_trace)

        if negative_count:
            logger.warning(
                "PetriNetLogBuilder: %d negative start→complete deltas discarded.",
                negative_count,
            )

        return filtered

    def _resolve_alignment(
        self, log: Any
    ) -> List[Tuple[Any, List[Tuple[Transition, Dict[str, Any]]]]]:
        """
        Dispatch to the configured replay engine (config.replay_engine) and return,
        for each accepted trace, its (trace, aligned) pair — aligned already being
        the final List[(Transition, attributes)] sequence, engine-agnostic.
        """
        if self.config.replay_engine == "alignments":
            return self._replay_log_alignments(log)
        return self._replay_log_token_based(log)

    def _replay_log_token_based(
        self, log: Any
    ) -> List[Tuple[Any, List[Tuple[Transition, Dict[str, Any]]]]]:
        """
        Run token-based replay and return (trace, aligned) pairs for traces
        that meet the minimum fitness threshold.

        Args:
            log: A pm4py EventLog.

        Returns:
            List of (original_trace, aligned) pairs above the fitness threshold,
            aligned being the lockstep-paired List[(Transition, attributes)].
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
                activated_transitions = result.get("activated_transitions", [])
                pairs.append((trace, self._align_trace(trace, activated_transitions)))
        if skipped > 0:
            skip_pct = 100.0 * skipped / total if total > 0 else 0.0
            logger.warning(
                f"PetriNetLog: skipped {skipped}/{total} traces ({skip_pct:.1f}%) "
                f"with fitness < {self.config.replay_min_fitness}"
            )
        else:
            logger.info(f"PetriNetLog: all {total} traces accepted.")
        return pairs

    def _replay_log_alignments(
        self, log: Any
    ) -> List[Tuple[Any, List[Tuple[Transition, Dict[str, Any]]]]]:
        """
        Run optimal alignments (config.replay_alignment_variant) and return
        (trace, aligned) pairs for traces that meet the minimum fitness threshold.

        Unlike token-based replay, alignments resolve each move exactly (including
        which silent transition fired, via ret_tuple_as_trans_desc=True), so no
        separate lockstep pass is needed — _align_trace_via_alignment converts the
        alignment directly into the final List[(Transition, attributes)] sequence.

        Args:
            log: A pm4py EventLog.

        Returns:
            List of (original_trace, aligned) pairs above the fitness threshold.
        """
        variant = _ALIGNMENT_VARIANTS.get(self.config.replay_alignment_variant)
        if variant is None:
            raise ValueError(
                f"Unknown replay_alignment_variant "
                f"{self.config.replay_alignment_variant!r}; expected one of "
                f"{sorted(_ALIGNMENT_VARIANTS)}"
            )

        logger.info(
            "Running alignments (%s) to build PetriNetLog...",
            self.config.replay_alignment_variant,
        )
        aligned_results = pm4py.conformance_diagnostics_alignments(
            log, self.petrinet, self.initial_marking, self.final_marking,
            variant_str=variant, ret_tuple_as_trans_desc=True,
        )
        total = len(aligned_results)
        skipped = 0
        pairs = []
        for trace, result in zip(log, aligned_results):
            fitness = result.get("fitness", 0.0)
            if fitness < self.config.replay_min_fitness:
                skipped += 1
            else:
                pairs.append(
                    (trace, self._align_trace_via_alignment(trace, result.get("alignment", [])))
                )
        if skipped > 0:
            skip_pct = 100.0 * skipped / total if total > 0 else 0.0
            logger.warning(
                f"PetriNetLog: skipped {skipped}/{total} traces ({skip_pct:.1f}%) "
                f"with fitness < {self.config.replay_min_fitness}"
            )
        else:
            logger.info(f"PetriNetLog: all {total} traces accepted.")
        return pairs

    def _align_trace_via_alignment(
        self,
        trace: Any,
        alignment: List[Tuple[Tuple[Any, Any], Tuple[Any, Any]]],
    ) -> List[Tuple[Transition, Dict[str, Any]]]:
        """
        Convert a pm4py alignment (obtained with ret_tuple_as_trans_desc=True) into
        the same List[(Transition, attributes)] shape produced by _align_trace.

        Each move is ((event_repr, transition_name), (activity_or_>>, transition_label_or_>>)).
        - transition_name in (None, '>>'): move-on-log (a log event with no
          corresponding model transition) — a deviation, already reflected in the
          trace's fitness. No FiringStep is produced and the event iterator is not
          advanced (the event itself is the unmatched one, not an extra to skip).
        - otherwise: a model move. If event_repr is not in (None, '>>') it is a sync
          move — the next log event is consumed and its attributes carried, exactly
          like _align_trace. Otherwise it's a move-on-model (including tau) and
          receives empty attributes {}, without consuming the event iterator.

        Args:
            trace: A pm4py Trace (iterable of Event dicts).
            alignment: The 'alignment' list from conformance_diagnostics_alignments.

        Returns:
            List of (transition, attributes) pairs, one per model move.
        """
        # Built from _trans_inputs/_trans_outputs (not self.petrinet.transitions):
        # these dicts are the source of truth for "all transitions" everywhere
        # else in this class (see _build_execution), and callers are only
        # required to pass a populated PetriNet object, not populated maps.
        transitions_by_name = {
            t.name: t for t in set(self._trans_inputs) | set(self._trans_outputs)
        }
        event_iter = iter(trace)
        result: List[Tuple[Transition, Dict[str, Any]]] = []
        for (event_repr, transition_name), _ in alignment:
            if transition_name in (None, ">>"):
                continue
            transition = transitions_by_name[transition_name]
            if event_repr not in (None, ">>"):
                attrs = dict(next(event_iter))
            else:
                attrs = {}
            result.append((transition, attrs))
        return result

    def _build_execution(
        self, trace: Any, aligned: List[Tuple[Transition, Dict[str, Any]]]
    ) -> TraceExecution:
        """
        Build a TraceExecution from a single trace and its already-resolved
        alignment (transition, attributes) sequence.

        Simulates the token marking to determine from_places for each firing.
        If a complete event carries a __duration_seconds__ attribute (injected by
        _filter_and_annotate_lifecycle), it is extracted into FiringStep.duration_seconds
        and removed from the step's attributes dict.

        Args:
            trace: A pm4py Trace (iterable of Event dicts).
            aligned: Ordered List[(Transition, attributes)], one per activated
                transition — produced by either _align_trace (token-based replay)
                or _align_trace_via_alignment (alignments).

        Returns:
            TraceExecution with one FiringStep per activated transition.
        """
        trace_id = trace.attributes.get("concept:name", "unknown")

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
                transition.label
                if not is_tau
                else self.silent_transitions.get(transition, f"tau_unknown_{id(transition)}")
            )

            # Extract the lifecycle duration embedded by _filter_and_annotate_lifecycle.
            # attributes is already a copy (dict(event) in _align_trace), so pop is safe.
            duration_seconds: Optional[float] = attributes.pop("__duration_seconds__", None)

            steps.append(FiringStep(
                transition=transition,
                activity_name=activity_name,
                is_tau=is_tau,
                from_places=from_places,
                attributes=attributes,
                duration_seconds=duration_seconds,
            ))

        return TraceExecution(trace_id=trace_id, steps=steps)

    def _align_trace(
        self,
        trace: Any,
        activated_transitions: List[Transition],
    ) -> List[Tuple[Transition, Dict[str, Any]]]:
        """
        Lockstep alignment between replay transitions and log events.

        Labeled transitions consume the next log event and carry its attributes
        as a fresh copy (dict). Tau transitions receive empty attributes {}.

        Used only for the token_based_replay engine; see
        _align_trace_via_alignment for the alignments engine's counterpart.

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

    @staticmethod
    def _delta_seconds(ts_from: Any, ts_to: Any) -> float:
        """
        Compute (ts_to - ts_from).total_seconds(), handling mixed timezone awareness.

        Args:
            ts_from: Start timestamp.
            ts_to: End timestamp.

        Returns:
            Duration in seconds.
        """
        try:
            return (ts_to - ts_from).total_seconds()
        except TypeError:
            if hasattr(ts_from, "replace"):
                ts_from = ts_from.replace(tzinfo=None)
            if hasattr(ts_to, "replace"):
                ts_to = ts_to.replace(tzinfo=None)
            return (ts_to - ts_from).total_seconds()
