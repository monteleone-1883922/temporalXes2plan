"""Wrap EvalAPI.run_planner() with linear backoff and failure pointer files.

Always uses OPTIC as planner.  Deterministic failures are returned immediately
without retry.  After all attempts are exhausted, a failure pointer JSON is
written to ``failure_dir/<query_id>.json`` for later inspection.

Example::

    from evaluation.planner_runner_with_retry import RetryConfig, run_with_retry

    cfg = RetryConfig(max_attempts=3, base_delay_s=2.0, timeout_s=60, memory_mb=4000)
    result, attempts = run_with_retry(domain, problem, api, cfg, failure_dir, query_id)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evaluation.eval_api import EvalAPI, PlanResult

logger = logging.getLogger(__name__)

_NON_RETRYABLE = {
    "unsolvable_structural", "unsolvable_resource", "unsolvable_parse",
    "timeout", "out_of_memory",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    """Parameters for the planner retry loop.

    Attributes:
        max_attempts: Maximum number of planner invocations per query.
        base_delay_s: Base delay in seconds; delay for attempt i = i * base_delay_s.
        timeout_s: Per-attempt OPTIC timeout in seconds.
        memory_mb: Memory cap for OPTIC in MB.
    """

    max_attempts: int = 3
    base_delay_s: float = 2.0
    timeout_s: int = 60
    memory_mb: int = 4000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_with_retry(
    domain_text: str,
    problem_text: str,
    api: EvalAPI,
    retry_cfg: RetryConfig,
    failure_dir: Path,
    query_id: str,
    pddl_dir: Optional[Path] = None,
) -> tuple[PlanResult, int]:
    """Run OPTIC with linear backoff, returning the result and attempt count.

    Deterministic failures (solvability in ``_NON_RETRYABLE``) are returned
    immediately without sleeping or incrementing the retry counter further.
    After all attempts are exhausted on retryable errors, a failure pointer
    file is written and the last result is returned.

    Args:
        domain_text: PDDL domain string.
        problem_text: PDDL problem string.
        api: EvalAPI instance used to invoke the planner.
        retry_cfg: Retry configuration (attempts, delays, timeout, memory).
        failure_dir: Directory to write failure pointer files on exhaustion.
        query_id: Unique query identifier used as the failure pointer filename.

    Returns:
        Tuple of (PlanResult, number_of_attempts_made).
    """
    last_result: Optional[PlanResult] = None

    for attempt in range(1, retry_cfg.max_attempts + 1):
        logger.info("[retry] %s attempt %d/%d", query_id, attempt, retry_cfg.max_attempts)
        result = api.run_planner(
            domain_text,
            problem_text,
            planner="optic",
            timeout=retry_cfg.timeout_s,
            memory_mb=retry_cfg.memory_mb,
        )
        last_result = result

        if result.success:
            logger.info("[retry] %s → solved (attempt %d)", query_id, attempt)
            return result, attempt

        if result.solvability in _NON_RETRYABLE:
            failure_path = _write_failure_pointer(
                failure_dir, query_id, problem_text, result, pddl_dir
            )
            logger.warning(
                "[retry] %s → %s (non-retryable, attempt %d): %s — see %s",
                query_id, result.solvability, attempt,
                result.error or "(no error detail)",
                failure_path,
            )
            return result, attempt

        if attempt < retry_cfg.max_attempts:
            delay = attempt * retry_cfg.base_delay_s
            logger.warning(
                "[retry] %s attempt %d → %s, retry in %.1fs",
                query_id, attempt, result.solvability, delay,
            )
            time.sleep(delay)

    logger.error("[retry] %s exhausted %d attempts — writing failure pointer.", query_id, retry_cfg.max_attempts)
    failure_path = _write_failure_pointer(failure_dir, query_id, problem_text, last_result, pddl_dir)
    logger.error("[retry] %s failure log: %s", query_id, failure_path)
    return last_result, retry_cfg.max_attempts


# ---------------------------------------------------------------------------
# Failure pointer
# ---------------------------------------------------------------------------

def _write_failure_pointer(
    failure_dir: Path,
    query_id: str,
    problem_text: str,
    last_result: PlanResult,
    pddl_dir: Optional[Path] = None,
) -> Path:
    """Write a JSON failure pointer file and problem.pddl for post-hoc debugging.

    Args:
        failure_dir: Directory to write the JSON file into (created if absent).
        query_id: Used as the filename stem.
        problem_text: PDDL problem string for the failed query.
        last_result: PlanResult from the final attempt.
        pddl_dir: If provided, problem.pddl is written to pddl_dir/<query_id>/problem.pddl.
    """
    failure_dir.mkdir(parents=True, exist_ok=True)

    if pddl_dir is not None:
        query_pddl_dir = pddl_dir / query_id
        query_pddl_dir.mkdir(parents=True, exist_ok=True)
        (query_pddl_dir / "problem.pddl").write_text(problem_text, encoding="utf-8")

    pointer = {
        "query_id": query_id,
        "last_solvability": last_result.solvability,
        "last_error": last_result.error,
        "last_stdout": last_result.raw_stdout,
        "last_stderr": last_result.raw_stderr,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    path = failure_dir / f"{query_id}.json"
    path.write_text(json.dumps(pointer, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
