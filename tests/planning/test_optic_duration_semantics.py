"""Integration test pinning OPTIC's semantics for duration inequalities.

The Q2 ground truth (trace_sampler.PrefixSample.reached_within_time) and
the Q2 precheck (query_builder.compute_min_time_to_end) both rely on this
behaviour: a durative action declared with
(and (>= ?duration MIN) (<= ?duration MAX)) gets a ?duration chosen by
OPTIC inside [MIN, MAX] (it is a variable of OPTIC's Simple Temporal
Network), so a sequence of actions meets a TIL deadline iff the sum of the
MINIMUM durations fits -- not the maximums.

The domain below uses the same encoding the project generates
(encoding/pddl_model.py's durative actions with (over all (deadline_ok)),
encoding/problem_builder.py's (at D (not (deadline_ok))) TIL):
two sequential actions a in [1, 5] and b in [2, 4], i.e. sum of minimums 3
and sum of maximums 9.

Requires the compiled OPTIC binary (optic/release/optic/optic-clp); run
with `pytest -m integration`.
"""
from pathlib import Path

import pytest

import planning.optic as optic_module

DOMAIN = """(define (domain duration_semantics)
  (:requirements :strips :typing :durative-actions :duration-inequalities :timed-initial-literals)
  (:predicates (p0) (p1) (p2) (deadline_ok))
  (:durative-action a
    :parameters ()
    :duration (and (>= ?duration 1) (<= ?duration 5))
    :condition (and (at start (p0)) (over all (deadline_ok)))
    :effect (and (at start (not (p0))) (at end (p1))))
  (:durative-action b
    :parameters ()
    :duration (and (>= ?duration 2) (<= ?duration 4))
    :condition (and (at start (p1)) (over all (deadline_ok)))
    :effect (and (at start (not (p1))) (at end (p2))))
)
"""

PROBLEM = """(define (problem duration_semantics_d) (:domain duration_semantics)
  (:init (p0) (deadline_ok) (at {deadline} (not (deadline_ok))))
  (:goal (p2)))
"""


def _run(tmp_path: Path, deadline: float) -> dict:
    (tmp_path / "domain.pddl").write_text(DOMAIN)
    (tmp_path / "problem.pddl").write_text(PROBLEM.format(deadline=deadline))
    return optic_module.run(tmp_path, stop_at_first=True, timeout=60)


@pytest.mark.integration
@pytest.mark.skipif(not optic_module.is_available(), reason="OPTIC binary not compiled")
class TestOpticDurationSemantics:
    def test_feasible_when_min_sum_fits_even_if_max_sum_exceeds_deadline(self, tmp_path):
        # sum(min) = 3 <= 6 < 9 = sum(max): OPTIC must find a plan.
        result = _run(tmp_path, deadline=6.0)
        assert result["success"] is True
        assert result["solvability"] == "solved"

    def test_unsolvable_when_min_sum_exceeds_deadline(self, tmp_path):
        # sum(min) = 3 > 2.5: no admissible assignment of durations exists.
        result = _run(tmp_path, deadline=2.5)
        assert result["success"] is False
