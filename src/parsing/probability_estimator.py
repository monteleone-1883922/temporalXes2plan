from collections import defaultdict
from typing import Dict, List, Set, Any

import pm4py
from pm4py import PetriNet, Marking
from pm4py.objects.powl.obj import Transition

import core_utils as utils
from models import AnalysisConfig

logger = utils.get_logger(__name__)


class ProbabilityEstimator:
    """
    Computes XOR-split branch probabilities via token-based replay against the Petri net.

    Token replay assigns each trace to an exact execution path through the net,
    eliminating the ambiguity of raw directly-follows counting.  Traces whose
    replay fitness falls below config.replay_min_fitness are excluded from the
    branch counts and logged.
    """

    def __init__(
        self,
        petrinet: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        edges: Set[PetriNet.Arc],
        silent_transitions: Dict[Transition, str],
        config: AnalysisConfig,
    ) -> None:
        """
        Initialize ProbabilityEstimator.

        Args:
            petrinet: The discovered Petri net.
            initial_marking: Initial token marking of the Petri net.
            final_marking: Final token marking of the Petri net.
            edges: Set of arcs in the Petri net (used to build input/output place maps).
            silent_transitions: Mapping from Transition objects to their tau names.
            config: Analysis configuration (replay_min_fitness threshold, etc.).
        """
        self.petrinet = petrinet
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.silent_transitions = silent_transitions
        self.config = config

        # Pre-build input/output place maps for marking simulation
        self._trans_inputs: Dict[Transition, Set[PetriNet.Place]] = defaultdict(set)
        self._trans_outputs: Dict[Transition, Set[PetriNet.Place]] = defaultdict(set)
        for arc in edges:
            if isinstance(arc.source, PetriNet.Place) and isinstance(arc.target, PetriNet.Transition):
                self._trans_inputs[arc.target].add(arc.source)
            elif isinstance(arc.source, PetriNet.Transition) and isinstance(arc.target, PetriNet.Place):
                self._trans_outputs[arc.source].add(arc.target)

    def compute_decision_points_probabilities(
        self,
        full_log: Any,
        decision_points: Dict[PetriNet.Place, List[Transition]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute branch probabilities for the given XOR-split decision points via
        token-based replay.

        Args:
            full_log: The full event log to replay.
            decision_points: XOR-split places mapped to their outgoing transitions,
                as returned by StructureAnalyzer.identify_xor_splits().

        Returns:
            Dictionary mapping place names to branch probabilities:
            PlaceName -> BranchActivityName -> Probability

            ESEMPIO DI OUTPUT RITORNATO:
            {
                "place_XOR_1": {
                    "ER_Triage": 0.40,
                    "ER_Sepsis_Triage": 0.60
                },
                "place_XOR_2": {
                    "Admission_ICU": 0.15,
                    "Release_A": 0.85
                }
            }
        """
        replay_results = self._replay_log(full_log)
        branch_counts = self._count_branches_from_replay(replay_results, decision_points)
        return self._normalize_branch_counts(branch_counts, decision_points)

    def _replay_log(self, full_log: Any) -> List[Dict]:
        """
        Replay the event log against the Petri net and return only traces that meet
        the minimum fitness threshold.

        Traces below config.replay_min_fitness are skipped; a summary warning is
        logged at the end showing the percentage of skipped traces.

        Args:
            full_log: The event log to replay.

        Returns:
            List of replay result dicts for traces that passed the fitness filter.
            Each dict contains at least 'activated_transitions' and 'trace_fitness'.
        """
        logger.info("Running token-based replay to estimate XOR-split probabilities...")
        # return_object_names=False → activated_transitions contains Transition
        # objects instead of name strings, required for marking simulation.
        replayed = pm4py.conformance_diagnostics_token_based_replay(
            full_log, self.petrinet, self.initial_marking, self.final_marking,
            opt_parameters={"return_object_names": False},
        )

        total = len(replayed)
        skipped = 0
        filtered = []

        for i, trace_result in enumerate(replayed):
            fitness = trace_result.get("trace_fitness", 0.0)
            if fitness < self.config.replay_min_fitness:
                logger.debug(
                    f"Skipping trace {i}: fitness={fitness:.3f} < threshold={self.config.replay_min_fitness}"
                )
                skipped += 1
            else:
                filtered.append(trace_result)

        if skipped > 0:
            skip_pct = 100.0 * skipped / total if total > 0 else 0.0
            logger.warning(
                f"Token replay: skipped {skipped}/{total} traces ({skip_pct:.1f}%) "
                f"with fitness < {self.config.replay_min_fitness}"
            )
        else:
            logger.info(f"Token replay: all {total} traces passed fitness threshold.")

        return filtered

    def _count_branches_from_replay(
        self,
        replay_results: List[Dict],
        decision_points: Dict[PetriNet.Place, List[Transition]],
    ) -> Dict[PetriNet.Place, Dict[Transition, int]]:
        """
        Count XOR-split branch activations by simulating the token marking over
        each replayed trace.

        For each trace we start from initial_marking and process activated_transitions
        in order.  Before firing each transition T we check which of its input places
        are XOR-split places that currently hold a token: only those get a +1 for
        branch T.  We then update the marking (consume inputs, produce outputs) before
        moving to the next transition.

        This is the only correct approach: it ties each branch count to the actual
        token state at the moment of firing, so overlapping XOR-split structures and
        loops are handled without ambiguity.

        Args:
            replay_results: Filtered replay results from _replay_log().
                activated_transitions must contain Transition objects
                (requires return_object_names=False in the replay call).
            decision_points: XOR-split places mapped to their outgoing transitions.

        Returns:
            Nested dict: Place -> Transition -> activation count.
        """
        xor_places: Set[PetriNet.Place] = set(decision_points.keys())
        branch_counts: Dict[PetriNet.Place, Dict[Transition, int]] = {
            place: defaultdict(int) for place in decision_points
        }

        for trace_result in replay_results:
            marking: Dict[PetriNet.Place, int] = dict(self.initial_marking)

            for transition in trace_result.get("activated_transitions", []):
                # Step 1 — count: which XOR-split places have a token and lead to T?
                for place in self._trans_inputs[transition]:
                    if place in xor_places and marking.get(place, 0) > 0:
                        branch_counts[place][transition] += 1

                # Step 2 — fire: consume one token from each input place
                for place in self._trans_inputs[transition]:
                    tokens = marking.get(place, 0)
                    if tokens == 1:
                        del marking[place]
                    elif tokens > 1:
                        marking[place] = tokens - 1

                # Step 3 — fire: produce one token in each output place
                for place in self._trans_outputs[transition]:
                    marking[place] = marking.get(place, 0) + 1

        return branch_counts

    def _normalize_branch_counts(
        self,
        branch_counts: Dict[PetriNet.Place, Dict[Transition, int]],
        decision_points: Dict[PetriNet.Place, List[Transition]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Normalize branch activation counts into probabilities.

        Branches with zero total count fall back to equal probabilities and a
        warning is logged.

        Args:
            branch_counts: Activation counts per place and transition.
            decision_points: XOR-split places mapped to their outgoing transitions.

        Returns:
            Dictionary mapping place names to branch probabilities.
        """
        result: Dict[str, Dict[str, float]] = {}

        for place, transitions in decision_points.items():
            counts = branch_counts[place]
            total = sum(counts.values())

            if total > 0:
                place_probs = {
                    self._get_activity_name_for_transition(t): round(counts[t] / total, 2)
                    for t in transitions
                }
            else:
                branch_names = [self._get_activity_name_for_transition(t) for t in transitions]
                logger.warning(
                    f"No replay data for XOR-split '{place.name}' "
                    f"(branches: {branch_names}). Assigning equal probabilities."
                )
                equal_prob = round(1.0 / len(transitions), 2) if transitions else 0.0
                place_probs = {
                    self._get_activity_name_for_transition(t): equal_prob
                    for t in transitions
                }

            if place_probs:
                result[place.name] = place_probs

        return result

    def _get_activity_name_for_transition(self, transition: Transition) -> str:
        """
        Resolves the label for a transition (returns its sanitized label or assigned tau name).

        Args:
            transition: The Transition object to check.

        Returns:
            Sanitized label name, or the pre-assigned unique silent transition ID name.
            E.g. "ER_Triage" or "tau_1"
        """
        if transition.label is not None:
            return utils.sanitize_name(transition.label)
        fallback_name = f"tau_unknown_{id(transition)}"
        return self.silent_transitions.get(transition, fallback_name)
