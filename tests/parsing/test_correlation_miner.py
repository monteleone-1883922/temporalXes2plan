"""
Tests for parsing.correlation_miner.CorrelationMiner.

All logs are constructed in-memory via conftest helpers — no XES files are loaded.

Two methods are tested:
  - discover_attribute_activity_relationships: finds which attribute values
    reliably precede specific activities (used for PDDL preconditions).
  - discover_activity_attribute_effects: finds how activities change attribute
    values between consecutive events (used for PDDL effects).

Key design constraint: CorrelationMiner only considers transitions that are
structurally valid in the Petri net, checked via the `predecessors` dict.
Tests must therefore build matching predecessor mappings alongside the logs.
"""
import pytest

from tests.helpers import make_event, make_trace, make_log
from parsing.correlation_miner import CorrelationMiner, RelationshipMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _miner(log, predecessors, intervals=None) -> CorrelationMiner:
    """Construct a CorrelationMiner with the given log, predecessor map, and intervals."""
    return CorrelationMiner(log, predecessors, intervals or {})


# ===========================================================================
# discover_attribute_activity_relationships
# ===========================================================================

class TestDiscoverAttributeActivityRelationships:
    def test_discriminative_attribute_value_is_captured(self):
        """
        Setup: 5 traces where 'triage(fever=yes)' always leads to 'administer_drug',
               5 traces where 'triage(fever=no)' always leads to 'discharge'.
        The 'fever' attribute has two values that perfectly predict the next activity.
        Both relationships must appear in the result with probability ≈ 1.0.

        predecessors: administer_drug ← triage, discharge ← triage
        """
        yes_traces = [
            make_trace(
                make_event("triage", offset=0, fever="yes"),
                make_event("administer_drug", offset=60),
                case_id=f"yes_{i}",
            )
            for i in range(5)
        ]
        no_traces = [
            make_trace(
                make_event("triage", offset=0, fever="no"),
                make_event("discharge", offset=60),
                case_id=f"no_{i}",
            )
            for i in range(5)
        ]
        log = make_log(*yes_traces, *no_traces)
        predecessors = {
            "administer_drug": ["triage"],
            "discharge": ["triage"],
        }
        result = _miner(log, predecessors).discover_attribute_activity_relationships()

        assert "fever" in result
        assert "yes" in result["fever"]
        assert "administer_drug" in result["fever"]["yes"]
        rel = result["fever"]["yes"]["administer_drug"]
        assert isinstance(rel, RelationshipMetrics)
        assert rel.probability == pytest.approx(1.0)

    def test_ubiquitous_attribute_is_filtered_out(self):
        """
        An attribute value that precedes ALL activities with equal probability has
        no discriminative power.  The miner filters it out via the
        STRONG_PROBABILITY_THRESHOLD / common_activities check.

        Setup: 'triage(temp=normal)' leads equally to 'drug' and 'discharge'.
        Neither transition of 'temp' can be attributed to a single activity.
        """
        traces = [
            make_trace(
                make_event("triage", offset=0, temp="normal"),
                make_event("drug", offset=60),
                case_id=f"d_{i}",
            )
            for i in range(5)
        ] + [
            make_trace(
                make_event("triage", offset=0, temp="normal"),
                make_event("discharge", offset=60),
                case_id=f"dis_{i}",
            )
            for i in range(5)
        ]
        log = make_log(*traces)
        predecessors = {"drug": ["triage"], "discharge": ["triage"]}
        result = _miner(log, predecessors).discover_attribute_activity_relationships()

        # 'temp' has only one value ('normal') → filtered as non-discriminative
        assert "temp" not in result

    def test_structurally_invalid_transition_is_ignored(self):
        """
        A consecutive pair (A → B) not present in the predecessors dict must be
        ignored, even if attribute data exists on A.
        This ensures only Petri-net-valid transitions influence preconditions.
        """
        log = make_log(
            make_trace(
                make_event("rogue", offset=0, attr_x="high"),
                make_event("target", offset=60),
            )
        )
        # 'rogue' is NOT listed as a predecessor of 'target'
        predecessors = {"target": ["some_other_activity"]}
        result = _miner(log, predecessors).discover_attribute_activity_relationships()

        # No valid structural transition → no relationships to mine
        assert not any(
            "target" in vals
            for attr_values in result.values()
            for vals in attr_values.values()
        )

    def test_result_contains_relationship_metrics_objects(self):
        """Every leaf value in the result dict must be a RelationshipMetrics instance."""
        log = make_log(
            *[
                make_trace(
                    make_event("A", offset=0, status="ok"),
                    make_event("B", offset=30),
                    case_id=f"c{i}",
                )
                for i in range(3)
            ],
            *[
                make_trace(
                    make_event("A", offset=0, status="error"),
                    make_event("C", offset=30),
                    case_id=f"e{i}",
                )
                for i in range(3)
            ],
        )
        predecessors = {"b": ["a"], "c": ["a"]}
        result = _miner(log, predecessors).discover_attribute_activity_relationships()

        for attr_values in result.values():
            for val_activities in attr_values.values():
                for rel in val_activities.values():
                    assert isinstance(rel, RelationshipMetrics)
                    assert 0.0 <= rel.probability <= 1.0
                    assert rel.count >= 1


# ===========================================================================
# discover_activity_attribute_effects
# ===========================================================================

class TestDiscoverActivityAttributeEffects:
    def test_value_transition_attributed_to_preceding_activity(self):
        """
        When an attribute changes value between two consecutive events,
        the change is recorded against the PRECEDING activity (the one that
        was executing when the value changed).

        Setup: 'treatment' is always preceded by 'triage(fever=high)' and
               'treatment' itself has 'fever=low'.
        Expected: effects['triage']['fever']['high']['low'] with probability > 0.
        """
        log = make_log(
            *[
                make_trace(
                    make_event("triage",    offset=0,  fever="high"),
                    make_event("treatment", offset=60, fever="low"),
                    case_id=f"c{i}",
                )
                for i in range(5)
            ]
        )
        predecessors = {"treatment": ["triage"]}
        result = _miner(log, predecessors).discover_activity_attribute_effects()

        assert "triage" in result
        assert "fever" in result["triage"]
        assert "high" in result["triage"]["fever"]
        rel = result["triage"]["fever"]["high"]["low"]
        assert isinstance(rel, RelationshipMetrics)
        assert rel.probability == pytest.approx(1.0)

    def test_new_attribute_introduction_marked_as_new(self):
        """
        When an attribute appears for the first time in a trace, it is recorded
        as a 'NEW' introduction attributed to the activity that first carries it.

        Setup: 'diagnosis' is not present at 'registration' but appears first
               at 'triage(diagnosis=sepsis)'.
        Expected: effects['triage']['diagnosis']['NEW']['sepsis'] with count > 0.
        """
        log = make_log(
            *[
                make_trace(
                    make_event("registration", offset=0),              # no diagnosis
                    make_event("triage", offset=30, diagnosis="sepsis"),
                    case_id=f"c{i}",
                )
                for i in range(4)
            ]
        )
        predecessors = {"triage": ["registration"]}
        result = _miner(log, predecessors).discover_activity_attribute_effects()

        assert "triage" in result
        assert "diagnosis" in result["triage"]
        assert "NEW" in result["triage"]["diagnosis"]
        assert "sepsis" in result["triage"]["diagnosis"]["NEW"]

    def test_unchanged_attribute_not_recorded_as_effect(self):
        """
        If an attribute has the same value in two consecutive events, no effect
        entry must be created (no-change transitions are meaningless for PDDL).
        """
        log = make_log(
            make_trace(
                make_event("step1", offset=0, stage="A"),
                make_event("step2", offset=30, stage="A"),   # same value — no change
            )
        )
        predecessors = {"step2": ["step1"]}
        result = _miner(log, predecessors).discover_activity_attribute_effects()

        # 'stage' must not appear as a value-transition effect
        if "step1" in result:
            assert "stage" not in result["step1"] or "a" not in result["step1"].get("stage", {})

    def test_effect_metrics_have_valid_probability(self):
        """All RelationshipMetrics in the effects result must have 0 <= probability <= 1."""
        log = make_log(
            *[
                make_trace(
                    make_event("A", offset=0, x="old"),
                    make_event("B", offset=60, x="new"),
                    case_id=f"c{i}",
                )
                for i in range(3)
            ]
        )
        predecessors = {"b": ["a"]}
        result = _miner(log, predecessors).discover_activity_attribute_effects()

        def _check(node):
            if isinstance(node, RelationshipMetrics):
                assert 0.0 <= node.probability <= 1.0
            elif isinstance(node, dict):
                for v in node.values():
                    _check(v)

        _check(result)
