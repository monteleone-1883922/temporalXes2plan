"""Entry point for the temporalXes2plan pipeline.

Usage:
    python -m main --log path/to/log.xes [options]
"""

import argparse
import os
import sys

import core_utils as utils
from pipeline import Pipeline

logger = utils.get_logger(__name__)

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_DEFAULT_PDDL_DIR = os.path.join(os.path.dirname(__file__), "..", "pddl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the XES → PDDL encoding pipeline."
    )
    parser.add_argument(
        "--log",
        required=True,
        metavar="PATH",
        help="Path to the XES event log file.",
    )
    parser.add_argument(
        "--data-dir",
        default=_DEFAULT_DATA_DIR,
        metavar="DIR",
        help="Root directory for UI JSON output (default: ../data/).",
    )
    parser.add_argument(
        "--pddl-out",
        default=_DEFAULT_PDDL_DIR,
        metavar="DIR",
        help="Directory for the PDDL domain file (default: ../pddl/).",
    )
    parser.add_argument(
        "--domain-name",
        default=None,
        metavar="NAME",
        help="Domain name in the PDDL header (default: log file stem).",
    )
    parser.add_argument(
        "--algorithm",
        default="inductive",
        choices=["alpha", "inductive", "heuristics", "ilp", "powl"],
        help="Petri net discovery algorithm (default: inductive).",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.001,
        metavar="FLOAT",
        help="Minimum cumulative variant coverage for log filtering (default: 0.001).",
    )
    parser.add_argument(
        "--durative",
        action="store_true",
        help="Encode timed transitions as durative actions.",
    )
    parser.add_argument(
        "--activity-classifier",
        action="store_true",
        help="Use concept:name + lifecycle:transition as activity label.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    log_path = os.path.abspath(args.log)
    if not os.path.isfile(log_path):
        logger.error("Log file not found: %s", log_path)
        sys.exit(1)

    data_dir = os.path.abspath(args.data_dir)
    pddl_out = os.path.abspath(args.pddl_out)

    pddl_path = Pipeline().run(
        log_path=str(log_path),
        data_dir=str(data_dir),
        pddl_output_dir=str(pddl_out),
        domain_name=args.domain_name,
        discovery_algorithm=args.algorithm,
        coverage_percentage=args.coverage,
        use_durative=args.durative,
        use_activity_classifier=args.activity_classifier,
    )

    print(f"Domain written to: {pddl_path}")
