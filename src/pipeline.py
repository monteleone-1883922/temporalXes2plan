"""End-to-end pipeline coordinator: XES event log → PDDL domain + UI JSON files.

Orchestrates:
  0. Config search  — (optional, search=True) network_search.runner.find_best_config()
                       picks the AnalysisConfig used by steps 1-4, see docs/network_improvement_loop.md
  1. Parsing        — Parser (xes_parser) produces a ParseResult
  2. Encoding       — DomainBuilder.build_prepared_input() + build_domain_with_variant_map()
                       build the shared PreparedDomainInput and the PDDLDomain from it
  3. Serialization  — PreparedDomainInput.to_dict() (+ start/end place metadata)
                       produces the UI JSON dict — the same object used to build the domain
  4. Persistence    — core_utils.save_original_and_current() writes original.json / current.json
  5. PDDL write     — PDDLDomain.write() writes the .pddl file
"""

import os
from pathlib import Path
from typing import Optional

import core_utils as utils
from models import AnalysisConfig
from parsing.xes_parser import Parser
from encoding.domain_builder import DomainBuilder, build_domain_with_variant_map
from network_search.runner import find_best_config
from network_search.scoring import ScoreWeights

logger = utils.get_logger(__name__)


class Pipeline:
    """Coordinates the full XES → PDDL + UI JSON pipeline."""


    def run(
        self,
        log_path: str,
        data_dir: str,
        pddl_output_dir: str,
        domain_name: Optional[str] = None,
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
    ) -> Path:
        """Run the full pipeline for a single XES log file.

        Args:
            log_path: Absolute or relative path to the XES event log.
            data_dir: Root directory for UI data (e.g. ``data/``).
                      A sub-directory named after the log stem is created inside.
            pddl_output_dir: Directory where the PDDL domain file is written.
            domain_name: Name embedded in the PDDL domain header.
                         Defaults to the sanitized log file stem.
            discovery_algorithm: Petri net discovery algorithm.
                                 One of ``alpha``, ``inductive``, ``heuristics``, ``ilp``.
            coverage_percentage: Minimum cumulative variant coverage for log filtering.
            use_durative: Encode durative actions when True.
            use_costs: Include action cost effects and functions section when True.
            use_activity_classifier: Combine concept:name + lifecycle:transition as label.
            config: Analysis configuration; defaults to AnalysisConfig(). Ignored
                when search=True (network_search picks the config instead).
            search: When True, run network_search.runner.find_best_config() to
                pick the AnalysisConfig instead of using `config` directly — see
                docs/network_improvement_loop.md/docs/network_improvement_loop_plan.md §0.
                Every candidate is scored on a held-out split of log_path, but the
                final network below is still built on the complete log_path with
                the winning config, not on that split.
            search_n_trials: Number of candidate configs to evaluate when search=True.
            search_test_pct: Fraction of traces held out to score candidates when search=True.
            search_seed: Random seed for the search's train/test split.
            search_weights: ScoreWeights for the search; defaults to ScoreWeights()
                (placeholder values, docs/network_improvement_loop.md §8).

        Returns:
            Path to the written PDDL domain file.
        """
        log_path = os.path.abspath(log_path)
        stem = utils.sanitize_name(Path(log_path).stem)
        if not domain_name:
            domain_name = stem

        config_dir = os.path.join(data_dir, stem)
        pddl_path = Path(pddl_output_dir) / "domain.pddl"

        if search:
            logger.info("Step 0/5: Searching for the best AnalysisConfig (%d trials)", search_n_trials)
            original_activity_names = utils.activity_names(log_path)
            with utils.split(log_path, test_pct=search_test_pct, seed=search_seed) as split_result:
                config, _trial_records = find_best_config(
                    training_log_path=str(split_result.train_path),
                    test_traces=split_result.test_cases,
                    n_trials=search_n_trials,
                    original_activity_names=original_activity_names,
                    weights=search_weights,
                    coverage_percentage=coverage_percentage,
                    discovery_algorithm=discovery_algorithm,
                )
        else:
            config = config or AnalysisConfig()

        # Route per-config debug output to data/<stem>/analysis_debug/
        config.snapshot_dir = os.path.join(config_dir, "analysis_debug")

        logger.info("=== Pipeline start: %s ===", stem)

        # 1 — Parse
        logger.info("Step 1/5: Parsing %s", log_path)
        parser = Parser(
            log_path=log_path,
            coverage_percentage=coverage_percentage,
            discovery_algorithm=discovery_algorithm,
            use_activity_classifier=use_activity_classifier,
            config=config,
        )
        parse_result = parser.parse_result

        # 2 — Encode
        logger.info("Step 2/4: Building PDDL domain")
        builder = DomainBuilder()
        prepared = builder.build_prepared_input(parse_result, config=config)
        domain, variant_map = build_domain_with_variant_map(
            prepared,
            config=config,
            domain_name=domain_name,
            use_durative=use_durative,
            use_costs=use_costs,
        )

        # 3 — Serialize (same PreparedDomainInput used to build the domain,
        # plus the start/end place metadata the GUI needs and that
        # PreparedDomainInput itself has no reason to carry)
        logger.info("Step 3/4: Serializing to UI JSON")
        ui_data = prepared.to_dict()
        ui_data["metadata"] = {
            "start_place": parse_result.start_place,
            "end_place": parse_result.end_place,
        }

        # 4 — Persist JSON (original + current, skipped if already exist)
        logger.info("Step 4/4: Saving JSON to %s", config_dir)
        orig_written, curr_written = utils.save_original_and_current(config_dir, ui_data)
        if orig_written:
            logger.info("original.json created")
        else:
            logger.info("original.json already existed — skipped")
        if curr_written:
            logger.info("current.json created")
        else:
            logger.info("current.json already existed — skipped")

        # Write PDDL
        logger.info("Writing PDDL domain to %s", pddl_path)
        domain.write(pddl_path)

        logger.info("=== Pipeline complete: %s ===", stem)
        return pddl_path
