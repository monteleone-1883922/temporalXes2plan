"""
Tests for parsing.petri_net_log_builder.PetriNetLogBuilder.

_align_trace and _build_execution are tested with synthetic Petri nets and
manually constructed trace/replay inputs — no real event log or pm4py discovery
algorithm is involved.

build() is tested with unittest.mock.patch to replace pm4py's conformance
checking with controlled fake results.
"""
import pytest
from typing import Dict, Set, TypedDict
from unittest.mock import patch

from pm4py import PetriNet, Marking

from tests.helpers import _transition, _place, _arc, make_event, make_trace, make_log
from parsing.petri_net_log_builder import PetriNetLogBuilder
from parsing.structure_analyzer import StructureAnalyzer
from models import AnalysisConfig, FiringStep, PetriNetLog, TraceExecution


# ---------------------------------------------------------------------------
# Net helpers
# ---------------------------------------------------------------------------

class _SeqNet(TypedDict):
    t_a: PetriNet.Transition
    t_b: PetriNet.Transition
    p_in: PetriNet.Place
    p_mid: PetriNet.Place
    p_out: PetriNet.Place
    arcs: Set[PetriNet.Arc]


class _TauSeqNet(TypedDict):
    t_a: PetriNet.Transition
    t_tau: PetriNet.Transition
    t_b: PetriNet.Transition
    p_in: PetriNet.Place
    p_mid1: PetriNet.Place
    p_mid2: PetriNet.Place
    p_out: PetriNet.Place
    arcs: Set[PetriNet.Arc]
    silent: Dict[PetriNet.Transition, str]


def _seq_net() -> _SeqNet:
    """
    Minimal sequence net:
        p_in -> t_a -> p_mid -> t_b -> p_out
    """
    t_a = _transition("t_a", "A")
    t_b = _transition("t_b", "B")
    p_in  = _place("p_in")
    p_mid = _place("p_mid")
    p_out = _place("p_out")
    arcs: Set[PetriNet.Arc] = {
        _arc(p_in,  t_a), _arc(t_a, p_mid),
        _arc(p_mid, t_b), _arc(t_b, p_out),
    }
    return _SeqNet(t_a=t_a, t_b=t_b, p_in=p_in, p_mid=p_mid, p_out=p_out, arcs=arcs)


def _tau_seq_net() -> _TauSeqNet:
    """
    Sequence net with a silent transition in the middle:
        p_in -> t_a -> p_mid1 -> t_tau -> p_mid2 -> t_b -> p_out
    """
    t_a   = _transition("t_a",   "A")
    t_tau = _transition("t_tau", None)
    t_b   = _transition("t_b",   "B")
    p_in   = _place("p_in")
    p_mid1 = _place("p_mid1")
    p_mid2 = _place("p_mid2")
    p_out  = _place("p_out")
    arcs: Set[PetriNet.Arc] = {
        _arc(p_in,   t_a),   _arc(t_a,   p_mid1),
        _arc(p_mid1, t_tau), _arc(t_tau, p_mid2),
        _arc(p_mid2, t_b),   _arc(t_b,   p_out),
    }
    return _TauSeqNet(
        t_a=t_a, t_tau=t_tau, t_b=t_b,
        p_in=p_in, p_mid1=p_mid1, p_mid2=p_mid2, p_out=p_out,
        arcs=arcs, silent={t_tau: "tau_1"},
    )


def _make_builder(
    arcs: Set[PetriNet.Arc],
    initial_marking: Marking,
    silent_transitions: Dict = None,
    config: AnalysisConfig = None,
) -> PetriNetLogBuilder:
    silent = silent_transitions or {}
    trans_inputs, trans_outputs = StructureAnalyzer(arcs, set(), silent).build_arc_maps()
    return PetriNetLogBuilder(
        petrinet=PetriNet("test"),
        initial_marking=initial_marking,
        final_marking=Marking(),
        trans_inputs=trans_inputs,
        trans_outputs=trans_outputs,
        silent_transitions=silent,
        config=config or AnalysisConfig(),
    )


def _fake_replay(transitions, fitness: float = 1.0) -> Dict:
    return {"trace_fitness": fitness, "activated_transitions": transitions}


# ===========================================================================
# _align_trace — lockstep alignment
# ===========================================================================

class TestAlignTrace:
    def test_labeled_transitions_get_event_attributes(self):
        net = _seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}))
        trace = make_trace(make_event("A", crp=2.1), make_event("B", crp=8.3))

        aligned = builder._align_trace(trace, [net["t_a"], net["t_b"]])

        assert aligned[0][1]["crp"] == 2.1
        assert aligned[1][1]["crp"] == 8.3

    def test_tau_transition_gets_empty_attributes(self):
        net = _tau_seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}), silent_transitions=net["silent"])
        trace = make_trace(make_event("A"), make_event("B"))

        aligned = builder._align_trace(trace, [net["t_a"], net["t_tau"], net["t_b"]])

        assert aligned[1][1] == {}

    def test_log_events_consumed_in_order_skipping_taus(self):
        net = _tau_seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}), silent_transitions=net["silent"])
        trace = make_trace(make_event("A", x=1), make_event("B", x=2))

        aligned = builder._align_trace(trace, [net["t_a"], net["t_tau"], net["t_b"]])

        assert aligned[0][0] is net["t_a"]
        assert aligned[0][1]["x"] == 1
        assert aligned[2][0] is net["t_b"]
        assert aligned[2][1]["x"] == 2

    def test_result_length_equals_number_of_activated_transitions(self):
        net = _tau_seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}), silent_transitions=net["silent"])
        trace = make_trace(make_event("A"), make_event("B"))

        aligned = builder._align_trace(trace, [net["t_a"], net["t_tau"], net["t_b"]])

        assert len(aligned) == 3


# ===========================================================================
# _build_execution — from_places, to_places, marking simulation
# ===========================================================================

class TestBuildExecution:
    def test_first_step_from_places_contains_initial_place(self):
        net = _seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}))
        trace = make_trace(make_event("A"), make_event("B"))

        execution = builder._build_execution(trace, _fake_replay([net["t_a"], net["t_b"]]))

        assert net["p_in"] in execution.steps[0].from_places

    def test_second_step_from_places_contains_intermediate_place(self):
        """Marking is updated after t_a fires, so t_b sees p_mid."""
        net = _seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}))
        trace = make_trace(make_event("A"), make_event("B"))

        execution = builder._build_execution(trace, _fake_replay([net["t_a"], net["t_b"]]))

        assert net["p_mid"] in execution.steps[1].from_places

    def test_labeled_step_carries_event_attributes(self):
        net = _seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}))
        trace = make_trace(make_event("A", crp=3.5), make_event("B", crp=9.1))

        execution = builder._build_execution(trace, _fake_replay([net["t_a"], net["t_b"]]))

        assert execution.steps[0].attributes["crp"] == 3.5
        assert execution.steps[1].attributes["crp"] == 9.1

    def test_tau_step_has_empty_attributes_and_is_tau_flag(self):
        net = _tau_seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}), silent_transitions=net["silent"])
        trace = make_trace(make_event("A"), make_event("B"))

        execution = builder._build_execution(
            trace, _fake_replay([net["t_a"], net["t_tau"], net["t_b"]])
        )
        tau_step = execution.steps[1]

        assert tau_step.is_tau is True
        assert tau_step.attributes == {}

    def test_tau_step_activity_name_comes_from_silent_transitions_map(self):
        net = _tau_seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}), silent_transitions=net["silent"])
        trace = make_trace(make_event("A"), make_event("B"))

        execution = builder._build_execution(
            trace, _fake_replay([net["t_a"], net["t_tau"], net["t_b"]])
        )

        assert execution.steps[1].activity_name == "tau_1"

    def test_labeled_step_is_not_tau(self):
        net = _seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}))
        trace = make_trace(make_event("A"), make_event("B"))

        execution = builder._build_execution(trace, _fake_replay([net["t_a"], net["t_b"]]))

        assert execution.steps[0].is_tau is False
        assert execution.steps[1].is_tau is False

    def test_trace_id_extracted_from_trace_attributes(self):
        net = _seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}))
        trace = make_trace(make_event("A"), make_event("B"), case_id="case_xyz")

        execution = builder._build_execution(trace, _fake_replay([net["t_a"], net["t_b"]]))

        assert execution.trace_id == "case_xyz"

    def test_step_count_matches_activated_transitions(self):
        net = _tau_seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}), silent_transitions=net["silent"])
        trace = make_trace(make_event("A"), make_event("B"))

        execution = builder._build_execution(
            trace, _fake_replay([net["t_a"], net["t_tau"], net["t_b"]])
        )

        assert len(execution.steps) == 3

    def test_marking_resets_between_traces(self):
        """Two independent executions must each start from initial_marking."""
        net = _seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}))
        trace = make_trace(make_event("A"), make_event("B"))

        ex1 = builder._build_execution(trace, _fake_replay([net["t_a"], net["t_b"]]))
        ex2 = builder._build_execution(trace, _fake_replay([net["t_a"], net["t_b"]]))

        # Both should start from p_in, not from the state left by ex1
        assert net["p_in"] in ex1.steps[0].from_places
        assert net["p_in"] in ex2.steps[0].from_places


# ===========================================================================
# TraceExecution — navigation helpers
# ===========================================================================

class TestNavigation:
    def _execution(self) -> tuple:
        net = _seq_net()
        builder = _make_builder(net["arcs"], Marking({net["p_in"]: 1}))
        trace = make_trace(make_event("A"), make_event("B"))
        ex = builder._build_execution(trace, _fake_replay([net["t_a"], net["t_b"]]))
        return ex, net

    def test_steps_for_transition_returns_matching_step(self):
        ex, net = self._execution()

        result = ex.steps_for_transition(net["t_a"])

        assert len(result) == 1
        assert result[0].transition is net["t_a"]

    def test_steps_for_transition_returns_empty_for_absent_transition(self):
        ex, net = self._execution()
        t_other = _transition("t_x", "X")

        assert ex.steps_for_transition(t_other) == []

    def test_steps_through_place_returns_consuming_step(self):
        """p_mid is the input place of t_b — only the consuming step must appear."""
        ex, net = self._execution()

        steps = ex.steps_through_place(net["p_mid"])
        transitions = {s.transition for s in steps}

        assert net["t_b"] in transitions
        assert net["t_a"] not in transitions

    def test_steps_through_place_returns_empty_for_unvisited_place(self):
        ex, net = self._execution()
        p_other = _place("p_other")

        assert ex.steps_through_place(p_other) == []


# ===========================================================================
# build() — integration with mocked replay
# ===========================================================================

class TestBuild:
    def _builder(self, config=None):
        net = _seq_net()
        return _make_builder(net["arcs"], Marking({net["p_in"]: 1}), config=config), net

    def _log(self, n: int = 1):
        return make_log(*[
            make_trace(make_event("A"), make_event("B"), case_id=f"c{i}")
            for i in range(n)
        ])

    def test_build_returns_petri_net_log_instance(self):
        builder, net = self._builder()
        with patch("pm4py.conformance_diagnostics_token_based_replay",
                   return_value=[_fake_replay([net["t_a"], net["t_b"]])]):
            result = builder.build(self._log())
        assert isinstance(result, PetriNetLog)

    def test_one_execution_per_accepted_trace(self):
        builder, net = self._builder()
        fake = [_fake_replay([net["t_a"], net["t_b"]])] * 3
        with patch("pm4py.conformance_diagnostics_token_based_replay", return_value=fake):
            result = builder.build(self._log(3))
        assert len(result.executions) == 3

    def test_low_fitness_trace_is_excluded(self):
        builder, net = self._builder()
        fake = [
            _fake_replay([net["t_a"], net["t_b"]], fitness=1.0),
            _fake_replay([net["t_a"], net["t_b"]], fitness=0.2),
        ]
        with patch("pm4py.conformance_diagnostics_token_based_replay", return_value=fake):
            result = builder.build(self._log(2))
        assert len(result.executions) == 1

    def test_custom_fitness_threshold_is_respected(self):
        config = AnalysisConfig(replay_min_fitness=0.5)
        builder, net = self._builder(config=config)
        fake = [
            _fake_replay([net["t_a"], net["t_b"]], fitness=1.0),
            _fake_replay([net["t_a"], net["t_b"]], fitness=0.6),
            _fake_replay([net["t_a"], net["t_b"]], fitness=0.3),
        ]
        with patch("pm4py.conformance_diagnostics_token_based_replay", return_value=fake):
            result = builder.build(self._log(3))
        assert len(result.executions) == 2

    def test_petri_net_log_stores_net_reference(self):
        builder, net = self._builder()
        with patch("pm4py.conformance_diagnostics_token_based_replay",
                   return_value=[_fake_replay([net["t_a"], net["t_b"]])]):
            result = builder.build(self._log())
        assert result.net is builder.petrinet
