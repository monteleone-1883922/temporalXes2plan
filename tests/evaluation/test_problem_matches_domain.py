"""The problem may only use the predicates the domain declares (OPTIC stops on a type error).

A boolean attribute has a (attr_false) predicate in the domain only when some guard negates it
(PDDL has no negation of a fact otherwise: false is the absence of (attr_true)). The initial
state of a problem must follow the same rule. Before it did not, and with a Sepsis model every
problem was rejected ("Predicate diagnosticblood_false not found")."""
import re

from encoding.domain_builder import build_domain_with_variant_map
from encoding.prepared_input import (
    GraphEdge, GraphNode, PreparedDomainInput, PreparedEffectGroup, PreparedTransition,
)
from evaluation.eval_api import EvalAPI
from models import AttributeCatalogEntry, Guard


def _prepared() -> PreparedDomainInput:
    """start --A--> mid --B--> end. A sets urgent and paid; B needs NOT urgent (so urgent is
    negated somewhere and gets (urgent_false)); nothing negates paid."""
    a = PreparedTransition(
        activity_name="A", input_places=["p_start"], preconditions=[],
        effect_groups=[PreparedEffectGroup(
            assignments=[("urgent", "True"), ("paid", "True")], guard=[], probability=1.0)],
    )
    b = PreparedTransition(
        activity_name="B", input_places=["p_mid"],
        preconditions=[[Guard(attribute="urgent", value=None, negated=True)]], effect_groups=[],
    )
    return PreparedDomainInput(
        transitions={"A": a, "B": b}, xor_branches={},
        attribute_catalog={
            "urgent": AttributeCatalogEntry("boolean", set()),
            "paid": AttributeCatalogEntry("boolean", set()),
        },
        nodes=[GraphNode("p_start", "place", "p_start"), GraphNode("t_a", "transition", "A"),
               GraphNode("p_mid", "place", "p_mid"), GraphNode("t_b", "transition", "B"),
               GraphNode("p_end", "place", "p_end")],
        edges=[GraphEdge("p_start", "t_a"), GraphEdge("t_a", "p_mid"),
               GraphEdge("p_mid", "t_b"), GraphEdge("t_b", "p_end")],
    )


def _problem(init_effects) -> tuple:
    prepared = _prepared()
    domain, _ = build_domain_with_variant_map(prepared, domain_name="d", use_durative=True, has_deadline=True)
    serialized = prepared.to_dict()
    serialized["metadata"] = {"end_place": "p_end"}
    problem = EvalAPI().build_problem(
        serialized=serialized, init_places=["p_start"], goal_sop=[[]], init_effects=init_effects,
        require_completion=True, domain_name="d", temporal=True,
    )
    return str(domain), problem


def _init_predicates(problem: str):
    init = problem.split("(:init")[1].split("(:goal")[0]
    return set(re.findall(r"\(([a-z_0-9]+)[ )]", init)) - {"and", "at", "not", "="}


def test_false_boolean_that_no_guard_negates_is_not_written_in_the_initial_state():
    domain, problem = _problem([{"attribute": "urgent", "value": "False"},
                                {"attribute": "paid", "value": "False"}])
    assert "(paid_false)" not in problem          # the domain does not declare it
    assert "(urgent_false)" in problem            # declared: urgent is negated by B's guard
    assert "(paid_false)" not in domain and "(urgent_false)" in domain


def test_true_booleans_are_always_written():
    domain, problem = _problem([{"attribute": "urgent", "value": "True"},
                                {"attribute": "paid", "value": "True"}])
    assert "(urgent_true)" in problem and "(paid_true)" in problem


def _declared_predicates(domain: str):
    block = domain.split("(:predicates")[1].split("(:durative-action")[0].split("(:action")[0]
    return set(re.findall(r"\(([a-z_0-9]+)", block))


def test_every_predicate_of_the_initial_state_is_declared_by_the_domain():
    for value in ("False", "True"):
        domain, problem = _problem([{"attribute": "urgent", "value": value},
                                    {"attribute": "paid", "value": value}])
        # deadline_ok comes with the deadline domain (not built here: no durations in this toy model)
        undeclared = _init_predicates(problem) - _declared_predicates(domain) - {"deadline_ok"}
        assert not undeclared, undeclared
