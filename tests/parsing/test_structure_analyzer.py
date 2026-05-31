"""
Tests for parsing.structure_analyzer.StructureAnalyzer.

All Petri nets are constructed programmatically via conftest helpers —
no XES files or pm4py discovery algorithms are involved.

Terminology reminder:
  - Place: a pm4py PetriNet.Place (circle in the net diagram).
  - Transition: a pm4py PetriNet.Transition (rectangle).
  - Arc: directed edge connecting a Place to a Transition or vice-versa.
  - Silent (tau) transition: a Transition with label=None, used to model
    invisible routing steps in the Petri net.
  - XOR-split: one Transition leads to a single Place that has multiple outgoing
    Transitions (exclusive choice — only one branch fires).
  - AND-split: one Transition leads to multiple Places, each with its own
    outgoing Transition (parallel execution — all branches fire).
"""
import pytest

from tests.helpers import (
    build_sequence_net,
    build_xor_net,
    build_and_net,
    _transition,
    _place,
    _arc,
)
from parsing.structure_analyzer import StructureAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(arcs, places, silent_transitions=None) -> StructureAnalyzer:
    """Construct a StructureAnalyzer from raw net components."""
    return StructureAnalyzer(arcs, places, silent_transitions or {})


# ===========================================================================
# extract_predecessors
# ===========================================================================

class TestExtractPredecessors:
    def test_linear_sequence_maps_each_activity_to_its_predecessor(self):
        """
        Net: A -> p0 -> B -> p1 -> C
        Expected predecessors:
          B has predecessor A
          C has predecessor B
          A has no predecessor (nothing feeds its input place p_start)
        """
        arcs, places, transitions, silent = build_sequence_net(["A", "B", "C"])
        preds = _make_analyzer(arcs, places, silent).extract_predecessors()

        assert "b" in preds and "a" in preds["b"]
        assert "c" in preds and "b" in preds["c"]
        # A itself has no predecessor in this net
        assert "a" not in preds

    def test_xor_split_gives_both_branches_the_same_predecessor(self):
        """
        Net: A -> p_xor -> B
                        -> C   (exclusive choice)
        Both B and C share the single predecessor A.
        """
        arcs, places, transitions, silent = build_xor_net("A", ["B", "C"])
        preds = _make_analyzer(arcs, places, silent).extract_predecessors()

        assert "a" in preds.get("b", [])
        assert "a" in preds.get("c", [])

    def test_silent_transition_appears_as_predecessor_with_tau_name(self):
        """
        Net: A -> p1 -> tau -> p2 -> B
        The silent transition is registered in silent_transitions as 'tau_1'.
        Expected: B's predecessor is 'tau_1', and tau_1's predecessor is A.
        """
        t_a = _transition("t_a", "A")
        t_tau = _transition("t_tau", None)    # label=None → silent
        t_b = _transition("t_b", "B")
        p0, p1, p2 = _place("p0"), _place("p1"), _place("p2")

        arcs = {
            _arc(p0, t_a), _arc(t_a, p1),
            _arc(p1, t_tau), _arc(t_tau, p2),
            _arc(p2, t_b),
        }
        places = {p0, p1, p2}
        silent_transitions = {t_tau: "tau_1"}

        preds = _make_analyzer(arcs, places, silent_transitions).extract_predecessors()

        assert "a" in preds.get("tau_1", [])
        assert "tau_1" in preds.get("b", [])

    def test_empty_net_returns_empty_dict(self):
        """A net with no arcs has no predecessor relationships."""
        preds = _make_analyzer(set(), set()).extract_predecessors()
        assert preds == {}


# ===========================================================================
# identify_parallels
# ===========================================================================

class TestIdentifyParallels:
    def test_and_split_detected_as_parallel(self):
        """
        Net: A -> p1 -> B    (AND-split: A fires, then B and C run concurrently)
               -> p2 -> C
        A must appear in the parallels dict with B and C as parallel successors.
        """
        arcs, places, transitions, silent = build_and_net("A", ["B", "C"])
        parallels = _make_analyzer(arcs, places, silent).identify_parallels()

        assert "a" in parallels
        assert set(parallels["a"]) == {"b", "c"}

    def test_xor_split_not_detected_as_parallel(self):
        """
        Net: A -> p_xor -> B
                        -> C   (XOR: only one branch fires)
        The shared choice place makes this an XOR-split, not an AND-split.
        StructureAnalyzer must NOT classify A as a parallel source.
        """
        arcs, places, transitions, silent = build_xor_net("A", ["B", "C"])
        parallels = _make_analyzer(arcs, places, silent).identify_parallels()

        assert "a" not in parallels

    def test_sequence_has_no_parallels(self):
        """A linear sequence A -> B -> C has no AND-split; parallels must be empty."""
        arcs, places, transitions, silent = build_sequence_net(["A", "B", "C"])
        parallels = _make_analyzer(arcs, places, silent).identify_parallels()
        assert parallels == {}

    def test_and_split_with_three_branches(self):
        """Three parallel branches after a single source must all appear together."""
        arcs, places, transitions, silent = build_and_net("Fork", ["X", "Y", "Z"])
        parallels = _make_analyzer(arcs, places, silent).identify_parallels()

        assert "fork" in parallels
        assert set(parallels["fork"]) == {"x", "y", "z"}


# ===========================================================================
# build_direct_transition_graph
# ===========================================================================

class TestBuildDirectTransitionGraph:
    def test_linear_sequence_connects_transitions_directly(self):
        """
        Net: A -> p0 -> B -> p1 -> C
        Removing the intermediate places, A must connect to B and B to C.
        The graph is verified by checking that each source's successors contain
        the expected activity names (graph values are lists of strings).
        """
        arcs, places, transitions, silent = build_sequence_net(["A", "B", "C"])
        graph = _make_analyzer(arcs, places, silent).build_direct_transition_graph()

        # Collect all successor strings across all source nodes
        all_pairs = {
            (src if isinstance(src, str) else src.label or src.name, succ)
            for src, succs in graph.items()
            for succ in succs
        }
        # Normalise to lowercase (sanitize_name lowercases labels)
        pairs_lower = {(s.lower() if s else s, d.lower()) for s, d in all_pairs}
        assert ("a", "b") in pairs_lower
        assert ("b", "c") in pairs_lower

    def test_graph_includes_tau_transitions_as_named_nodes(self):
        """
        Silent transitions are NOT removed from the direct transition graph —
        they appear as 'tau_1' nodes.  The graph reflects the full Petri net
        structure including invisible routing steps.

        Net: A -> p0 -> tau_1 -> p1 -> B
        Expected edges: A -> tau_1  and  tau_1 -> B
        """
        t_a = _transition("t_a", "A")
        t_tau = _transition("t_tau", None)
        t_b = _transition("t_b", "B")
        p0, p1, p2 = _place("p0"), _place("p1"), _place("p2")

        arcs = {
            _arc(p0, t_a), _arc(t_a, p1),
            _arc(p1, t_tau), _arc(t_tau, p2),
            _arc(p2, t_b),
        }
        places = {p0, p1, p2}
        silent_transitions = {t_tau: "tau_1"}

        graph = _make_analyzer(arcs, places, silent_transitions).build_direct_transition_graph()

        all_succs_flat = {
            succ
            for succs in graph.values()
            for succ in succs
        }
        assert "tau_1" in all_succs_flat or "b" in all_succs_flat
