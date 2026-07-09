"""Internal evaluation API — programmatic access to pipeline steps without HTTP or GUI.

Designed for use in evaluation/benchmarking scripts.  Each method corresponds to
one logical pipeline step and can be called independently or via run_pipeline().

Example::

    from evaluation.eval_api import EvalAPI

    api = EvalAPI()

    # Full pipeline in one shot
    result = api.run_pipeline("logs/sample.xes", use_durative=True)
    print(result.domain_text[:200])

    # Incremental: parse → build_domain → build_problem → run_planner
    pr      = api.parse("logs/sample.xes")
    domain  = api.build_domain(pr, domain_name="clinic")
    problem = api.build_problem(result.serialized, init_places=["p_start"],
                                goal_sop=[[{"attribute": "status",
                                            "predicate": "=", "value": "done"}]])
    plan    = api.run_planner(domain, problem)
    print(plan.success, plan.plan_steps)
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models import AnalysisConfig, ParseResult
from parsing.log_validator import ValidationResult, validate_event_log
from parsing.xes_parser import Parser
from parsing.csv_loader import csv_to_event_log
from encoding.domain_builder import DomainBuilder
from encoding.domain_builder import build_domain_with_variant_map as _build_domain_with_variant_map
from encoding.prepared_input import PreparedDomainInput, VariantInfo
from encoding.problem_builder import ProblemBuilder
from planning.planner_runner import PlannerResult, run_planner as _run_planner
from pipeline import Pipeline, BuildResult
from network_search.scoring import ScoreWeights
import pm4py

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Output of a full XES → PDDL pipeline run."""

    parse_result: ParseResult
    domain_text: str
    serialized: Dict[str, Any]


@dataclass
class PlanResult:
    """Outcome of a planner invocation."""

    success: bool
    plan_steps: List[str]
    cost: Optional[float]
    duration_s: Optional[float]
    solvability: str
    error: Optional[str]
    raw_stdout: str
    raw_stderr: str
    planner: str


# ---------------------------------------------------------------------------
# Main API class
# ---------------------------------------------------------------------------

class EvalAPI:
    """Facade for programmatic access to all pipeline stages.

    All methods are stateless — instantiate once and call freely.
    """

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def build_network(
        self,
        log_path: str,
        domain_name: str = "test_process",
        discovery_algorithm: str = "inductive",
        coverage_percentage: float = 0.001,
        use_durative: bool = False,
        use_costs: bool = False,
        use_activity_classifier: bool = False,
        config: Optional[AnalysisConfig] = None,
        search: bool = False,
        search_n_trials: int = 30,
        search_test_pct: float = 0.2,
        search_seed: int = 42,
        search_weights: Optional[ScoreWeights] = None,
        snapshot_dir: Optional[str] = None,
        discretizer_cache_path: Optional[Path] = None,
        force_rediscretize: bool = False,
    ) -> BuildResult:
        """Build the PDDL domain for an event log — delegates entirely to
        Pipeline.build_network() (the same optimizer + parse + encode logic
        used by the CLI/web pipeline), so evaluation never re-implements the
        search: the domain is built exactly once per log, optimizer or not,
        and every query/test run afterwards reuses the same BuildResult.
        """
        return Pipeline().build_network(
            log_path=log_path,
            domain_name=domain_name,
            discovery_algorithm=discovery_algorithm,
            coverage_percentage=coverage_percentage,
            use_durative=use_durative,
            use_costs=use_costs,
            use_activity_classifier=use_activity_classifier,
            config=config,
            search=search,
            search_n_trials=search_n_trials,
            search_test_pct=search_test_pct,
            search_seed=search_seed,
            search_weights=search_weights,
            snapshot_dir=snapshot_dir,
            discretizer_cache_path=discretizer_cache_path,
            force_rediscretize=force_rediscretize,
        )


    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def validate_log(
        self,
        log_path: str,
        mapping: Optional[Dict[str, Optional[str]]] = None,
    ) -> ValidationResult:
        """Load an event log and validate its mandatory fields.

        Args:
            log_path: Path to the XES or CSV event log.
            mapping: Column mapping required for CSV files. Keys: case_id,
                activity, timestamp (optional), lifecycle (optional).

        Returns:
            ValidationResult with errors, warnings, infos, and boolean flags.
        """
        if log_path.lower().endswith(".csv"):
            if mapping is None:
                raise ValueError("Column mapping is required for CSV logs.")
            log = csv_to_event_log(log_path, mapping)
        else:

            log = pm4py.objects.log.importer.xes.importer.apply(log_path)
        return validate_event_log(log)

    def serialize(self, prepared: PreparedDomainInput, parse_result: ParseResult) -> Dict[str, Any]:
        """Serialize a PreparedDomainInput into the current.json-shaped UI dict.

        Same shape produced by pipeline.py's own serialization step — the
        PreparedDomainInput used to build the domain is the source of truth,
        plus the start/end place metadata it has no reason to carry itself.
        """
        data = prepared.to_dict()
        data["metadata"] = {
            "start_place": parse_result.start_place,
            "end_place": parse_result.end_place,
        }
        return data


    def build_problem(
        self,
        serialized: Dict[str, Any],
        init_places: List[str],
        goal_sop: List[List[Dict[str, Any]]],
        init_effects: Optional[List[Dict[str, Any]]] = None,
        metric: Optional[str] = None,
        cost_weight: float = 0.001,
        require_completion: bool = False,
        deadline: Optional[float] = None,
        domain_name: str = "test_process",
        problem_name: str = "test_problem",
            temporal: bool = False,
    ) -> str:
        """Build a PDDL problem string.

        Args:
            serialized: Dict in current.json format (used for attribute_catalog
                and end_place).
            init_places: List of place IDs that hold tokens in the initial state.
            goal_sop: Goal in SOP form — list of AND-clauses, each a list of
                condition dicts with keys attribute, predicate ("=" or "<>"),
                value.
            init_effects: List of {"attribute": str, "value": str} dicts
                representing attribute values in the initial state.
            metric: "minimize_cost" | "minimize_time" | "minimize_weighted" | None.
            cost_weight: Scaling factor α for "minimize_weighted" metric.
            require_completion: Append end-place marking to every goal clause.
            deadline: Deadline in seconds (TIL in PDDL).
            domain_name: Must match the domain name used in build_domain().
            problem_name: PDDL problem name.

        Returns:
            PDDL problem as a string.
        """
        attribute_catalog = serialized.get("attribute_catalog", {})
        end_place = serialized.get("metadata", {}).get("end_place")
        return ProblemBuilder().build(
            problem_name=problem_name,
            domain_name=domain_name,
            init_places=init_places,
            init_effects=init_effects or [],
            goal_sop=goal_sop,
            attribute_catalog=attribute_catalog,
            metric=metric,
            cost_weight=cost_weight,
            require_completion=require_completion,
            end_place=end_place,
            deadline=deadline,
            temporal=temporal
        )

    def run_planner(
        self,
        domain_text: str,
        problem_text: str,
        planner: str = "fast_downward",
        **options: Any,
    ) -> PlanResult:
        """Run a planner on domain + problem PDDL strings.

        Writes both files to a temporary directory, invokes the planner, and
        cleans up on exit.

        Args:
            domain_text: PDDL domain string.
            problem_text: PDDL problem string.
            planner: "fast_downward" or "optic".
            **options: Planner-specific options.
                For fast_downward: search, timeout, memory_mb.
                For optic: stop_at_first, ignore_costs, timeout, memory_mb.

        Returns:
            PlanResult with outcome details.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pddl_dir = Path(tmp)
            (pddl_dir / "domain.pddl").write_text(domain_text, encoding="utf-8")
            (pddl_dir / "problem.pddl").write_text(problem_text, encoding="utf-8")
            raw: PlannerResult = _run_planner(pddl_dir, planner, options)

        return PlanResult(
            success=raw.success,
            plan_steps=raw.plan_actions,
            cost=raw.metrics.get("cost"),
            duration_s=raw.metrics.get("duration"),
            solvability=raw.solvability,
            error=None if raw.success else raw.message,
            raw_stdout=raw.stdout,
            raw_stderr=raw.stderr,
            planner=raw.planner,
        )
