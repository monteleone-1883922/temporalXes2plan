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
from typing import Any, Dict, List, Optional

from models import AnalysisConfig, ParseResult
from parsing.log_validator import ValidationResult, validate_event_log
from parsing.xes_parser import Parser
from parsing.csv_loader import csv_to_event_log
from parsing.partial_trace_replayer import PartialTraceReplayer
from encoding.domain_builder import DomainBuilder
from encoding.pddl_writer import PDDLWriter
from encoding.problem_builder import ProblemBuilder
from planning.planner_runner import PlannerResult, run_planner as _run_planner
from web.serializer import serialize_parse_result


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


@dataclass
class ReplayResult:
    """Outcome of a partial-trace replay."""

    init_places: List[str]
    init_effects: List[Dict[str, Any]]
    replayed_activities: List[str]
    n_events: int
    warnings: List[str]


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

    def run_pipeline(
        self,
        log_path: str,
        config: Optional[AnalysisConfig] = None,
        domain_name: str = "process",
        discovery_algorithm: str = "inductive",
        coverage_percentage: float = 0.001,
        use_durative: bool = False,
        use_costs: bool = False,
        use_activity_classifier: bool = False,
    ) -> PipelineResult:
        """Parse an event log and encode it into a PDDL domain.

        Does NOT write any files to disk.

        Args:
            log_path: Path to the XES or CSV event log.
            config: Optional AnalysisConfig; defaults are used if None.
            domain_name: PDDL domain name.
            discovery_algorithm: Process discovery algorithm ("inductive" or "heuristic").
            coverage_percentage: Minimum variant coverage for log filtering.
            use_durative: Encode durative actions (requires timestamps).
            use_costs: Include action costs in the PDDL domain.
            use_activity_classifier: Use concept:name + lifecycle:transition as classifier.

        Returns:
            PipelineResult with parse_result, domain_text, and serialized JSON.
        """
        parse_result = self.parse(
            log_path,
            config=config,
            coverage_percentage=coverage_percentage,
            discovery_algorithm=discovery_algorithm,
            use_activity_classifier=use_activity_classifier,
        )
        domain_text = self.build_domain(
            parse_result,
            domain_name=domain_name,
            use_durative=use_durative,
            use_costs=use_costs,
        )
        serialized = serialize_parse_result(parse_result)
        return PipelineResult(
            parse_result=parse_result,
            domain_text=domain_text,
            serialized=serialized,
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
            import pm4py
            log = csv_to_event_log(log_path, mapping)
        else:
            import pm4py
            log = pm4py.read_xes(log_path)
        return validate_event_log(log)

    def parse(
        self,
        log_path: str,
        config: Optional[AnalysisConfig] = None,
        coverage_percentage: float = 0.001,
        discovery_algorithm: str = "inductive",
        use_activity_classifier: bool = False,
    ) -> ParseResult:
        """Run the parsing stage only (Petri net discovery + mining).

        Args:
            log_path: Path to the XES or CSV event log.
            config: Optional AnalysisConfig; defaults are used if None.
            coverage_percentage: Minimum variant coverage for log filtering.
            discovery_algorithm: "inductive" or "heuristic".
            use_activity_classifier: Use concept:name + lifecycle:transition.

        Returns:
            ParseResult with Petri net model, transitions, attribute catalog, etc.
        """
        parser = Parser(
            log_path=log_path,
            coverage_percentage=coverage_percentage,
            discovery_algorithm=discovery_algorithm,
            use_activity_classifier=use_activity_classifier,
            config=config,
        )
        return parser.parse_result

    def build_domain(
        self,
        parse_result: ParseResult,
        domain_name: str = "process",
        use_durative: bool = False,
        use_costs: bool = False,
    ) -> str:
        """Encode a ParseResult into a PDDL domain string.

        Args:
            parse_result: Output of parse().
            domain_name: PDDL domain name.
            use_durative: Encode durative actions.
            use_costs: Include action costs.

        Returns:
            PDDL domain as a string.
        """
        domain, _ = DomainBuilder().build_with_registry(
            parse_result,
            domain_name=domain_name,
            use_durative=use_durative,
            use_costs=use_costs,
        )
        return PDDLWriter()._render_domain(domain)


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
        domain_name: str = "process",
        problem_name: str = "problem",
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

    def replay_trace(
        self,
        trace_bytes: bytes,
        serialized: Dict[str, Any],
        fmt: str = "xes",
        mapping: Optional[Dict[str, Optional[str]]] = None,
    ) -> ReplayResult:
        """Replay a partial trace against a serialized Petri net.

        Args:
            trace_bytes: Raw bytes of the XES or CSV trace file.
            serialized: Dict in current.json format.
            fmt: "xes" or "csv".
            mapping: Column mapping required when fmt="csv".

        Returns:
            ReplayResult with derived init state.

        Raises:
            PartialTraceError: If the trace cannot be replayed.
        """
        raw = PartialTraceReplayer().replay(
            file_bytes=trace_bytes,
            current_data=serialized,
            fmt=fmt,
            mapping=mapping,
        )
        return ReplayResult(
            init_places=raw["init_places"],
            init_effects=raw["init_effects"],
            replayed_activities=raw["replayed_activities"],
            n_events=raw["n_events"],
            warnings=raw["warnings"],
        )
