"""End-to-end pipeline coordinator: XES event log → PDDL domain + UI JSON files.

Orchestrates:
  1. Parsing        — Parser (xes_parser) produces a ParseResult
  2. Encoding       — DomainBuilder.build() builds the PDDLDomain
  3. Serialization  — serialize_parse_result() produces the UI JSON dict
  4. Persistence    — save_original_and_current() writes original.json / current.json
  5. PDDL write     — PDDLWriter.write_domain() writes the .pddl file
"""

import os
from pathlib import Path
from typing import Optional

import core_utils as utils
from models import AnalysisConfig
from parsing.xes_parser import Parser
from encoding.domain_builder import DomainBuilder
from encoding.graph_updater import save_original_and_current, update_parse_result
from encoding.pddl_writer import PDDLWriter
from web.serializer import serialize_parse_result

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
        use_activity_classifier: bool = False,
        config: Optional[AnalysisConfig] = None,
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
            use_activity_classifier: Combine concept:name + lifecycle:transition as label.
            config: Analysis configuration; defaults to AnalysisConfig().

        Returns:
            Path to the written PDDL domain file.
        """
        log_path = os.path.abspath(log_path)
        stem = utils.sanitize_name(Path(log_path).stem)
        if not domain_name:
            domain_name = stem

        config_dir = os.path.join(data_dir, stem)
        pddl_path = Path(pddl_output_dir) / f"{stem}_domain.pddl"

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
        domain, registry = builder.build_with_registry(
            parse_result=parse_result,
            domain_name=domain_name,
            use_durative=use_durative,
            config=config,
        )
        update_parse_result(registry, parse_result)

        # 3 — Serialize
        logger.info("Step 3/4: Serializing to UI JSON")
        ui_data = serialize_parse_result(parse_result)

        # 4 — Persist JSON (original + current, skipped if already exist)
        logger.info("Step 4/4: Saving JSON to %s", config_dir)
        orig_written, curr_written = save_original_and_current(config_dir, ui_data)
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
        PDDLWriter().write_domain(domain, pddl_path)

        logger.info("=== Pipeline complete: %s ===", stem)
        return pddl_path
