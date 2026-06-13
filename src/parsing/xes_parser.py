"""XES parsing pipeline — facade coordinating all analysis modules.

Orchestrates the full pipeline from raw XES log to encoder-ready ParseResult:
  1. Log loading and filtering        (LogProcessor)
  2. Petri net discovery              (ModelDiscoverer)
  3. Numeric attribute discretization (Discretizer)
  4. Token replay                     (PetriNetLogBuilder)
  5. Single-pass log preprocessing    (LogPreprocessor)
  6. Probability estimation           (ProbabilityEstimator)
  7. XOR split screening              (DecisionMiner.screen_xor_splits)
  8. Effect screening                 (DecisionMiner.screen_effects)
  9. XOR split DT mining              (DecisionMiner.mine_xor_splits)
  10. Effect DT mining                (DecisionMiner.mine_effects)

Steps 4–10 store intermediate results as instance attributes so that
downstream steps (pruning, filtering, ParseResult assembly) can access them.
"""

import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import pm4py

import core_utils as utils
from models import (
    AnalysisConfig,
    AttributeEffect,
    PetriNetModel,
    PreprocessedLog,
    TransitionEffects,
    TransitionScreening,
    XorSplitGuards,
    XorSplitScreening,
    XorSplitStats,
)
from .discretizer import Discretizer
from .log_preprocessor import LogPreprocessor
from .log_processor import LogProcessor
from .model_discoverer import ModelDiscoverer
from .petri_net_log_builder import PetriNetLogBuilder
from .probability_estimator import ProbabilityEstimator
from .decision_mining import DecisionMiner

SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

logger = utils.get_logger(__name__)


class Parser:
    """Facade orchestrating the XES → encoder-ready analysis pipeline.

    After construction all intermediate results are available as instance
    attributes.  Steps 5–10 (filter, assemble ParseResult) are implemented
    separately so that the pipeline can be paused for inspection between
    stages.
    """

    def __init__(
        self,
        log_path: str,
        coverage_percentage: float,
        discovery_algorithm: str = 'inductive',
        use_activity_classifier: bool = False,
        config: Optional[AnalysisConfig] = None,
    ) -> None:
        """Run the full analysis pipeline up to and including DT mining.

        Args:
            log_path: Path to the XES event log file.
            coverage_percentage: Minimum cumulative variant coverage for
                log filtering (e.g. 0.8 keeps traces covering 80% of cases).
            discovery_algorithm: Petri net discovery algorithm name.
                One of 'alpha', 'inductive', 'heuristics', 'ilp'.
            use_activity_classifier: Combine concept:name + lifecycle:transition
                as activity label when True.
            config: Analysis configuration; defaults to AnalysisConfig() if None.
        """
        self.config = config or AnalysisConfig()

        # --- Step 1: log loading and filtering ---
        logger.info("Loading and filtering log: %s", log_path)
        processor = LogProcessor(log_path, use_activity_classifier)
        self.log, self.full_log, _ = processor.load_and_filter_log(coverage_percentage)
        train_log, _ = processor.split_train_test(self.log)
        self.log = train_log
        self.attributes, self.attribute_categories = processor.initialize_attributes(
            self.full_log
        )
        logger.info("Log loaded: %d training traces, %d attributes",
                    len(self.log), len(self.attributes))

        # --- Step 2: Petri net discovery ---
        logger.info("Starting Petri net discovery [%s]", discovery_algorithm)
        self.petri_net_model: PetriNetModel = ModelDiscoverer(
            discovery_algorithm
        ).discover(self.log)
        pnm = self.petri_net_model
        logger.info("Petri net discovered: %d places, %d transitions, %d XOR splits",
                    len(pnm.petrinet.places), len(pnm.petrinet.transitions),
                    len(pnm.xor_splits))

        # --- Step 3: numeric attribute discretization ---
        numeric_attrs = [
            a for a, t in self.attribute_categories.items() if t == 'numerical'
        ]
        logger.info("Starting discretization for %d numerical attributes", len(numeric_attrs))
        discretizer = Discretizer(self.config)
        discretizer.fit(self.full_log, numeric_attrs)
        self.discretizer = discretizer
        logger.info("Discretization complete: %d attributes discretized",
                    len(discretizer.boundaries))

        # --- Step 4: token replay → PetriNetLog ---
        logger.info("Starting token replay")
        builder = PetriNetLogBuilder(
            petrinet=pnm.petrinet,
            initial_marking=pnm.initial_marking,
            final_marking=pnm.final_marking,
            trans_inputs=pnm.trans_inputs,
            trans_outputs=pnm.trans_outputs,
            silent_transitions=pnm.silent_transitions,
            config=self.config,
        )
        self.pn_log = builder.build(self.log)
        logger.info("Token replay complete: %d traces accepted", len(self.pn_log.executions))

        # --- Step 5: single-pass log preprocessing ---
        logger.info("Starting log preprocessing")
        self.preprocessed_log: PreprocessedLog = LogPreprocessor(
            config=self.config,
            discretizer=discretizer,
        ).preprocess(self.pn_log, pnm.xor_splits)
        logger.info("Log preprocessing complete")

        # --- Step 6: probability estimation ---
        logger.info("Starting probability estimation")
        estimator = ProbabilityEstimator(
            silent_transitions=pnm.silent_transitions,
            config=self.config,
        )
        self.xor_stats: Dict[str, XorSplitStats] = estimator.compute_xor_probabilities(
            self.preprocessed_log, pnm.xor_splits
        )
        self.attribute_effects: Dict[str, AttributeEffect] = (
            estimator.compute_attribute_effect_probabilities(self.preprocessed_log)
        )
        logger.info("Probability estimation complete")

        # --- Steps 7–10: screening and DT mining ---
        miner = DecisionMiner(
            silent_transitions=pnm.silent_transitions,
            config=self.config,
        )

        logger.info("Starting XOR split screening")
        self.xor_screening: Dict[str, XorSplitScreening] = miner.screen_xor_splits(
            pnm.xor_splits, self.xor_stats
        )
        logger.info("XOR split screening complete")

        logger.info("Starting effect screening")
        self.effect_screening: Dict[str, TransitionScreening] = miner.screen_effects(
            self.attribute_effects
        )
        logger.info("Effect screening complete")

        logger.info("Starting XOR split DT mining")
        self.xor_guards: Dict[str, XorSplitGuards] = miner.mine_xor_splits(
            self.preprocessed_log, self.xor_screening
        )
        logger.info("XOR split DT mining complete: %d guards trained", len(self.xor_guards))

        logger.info("Starting effect DT mining")
        self.transition_effects: Dict[str, TransitionEffects] = miner.mine_effects(
            self.preprocessed_log, self.effect_screening
        )
        logger.info("Effect DT mining complete")

        # --- Snapshot: persist pre-pruning state to disk ---
        logger.info("Saving pre-pruning snapshot")
        self._save_snapshot()
        logger.info("Snapshot saved to %s", self.config.snapshot_dir)

        logger.info("Pipeline complete")

    def _save_snapshot(self) -> None:
        """Persist the pre-pruning state (PetriNet + screening) to disk.

        Writes three files to config.snapshot_dir:
          - petri_net.pnml     : original discovered Petri net (PNML format)
          - xor_screening.json : XOR split screening decisions
          - effect_screening.json : effect screening decisions

        The directory is created if it does not exist.  Existing files are
        overwritten silently so that re-runs always reflect the latest log.
        """
        out = self.config.snapshot_dir
        os.makedirs(out, exist_ok=True)

        pnm = self.petri_net_model
        pm4py.write_pnml(
            pnm.petrinet,
            pnm.initial_marking,
            pnm.final_marking,
            os.path.join(out, "petri_net.pnml"),
        )

        xor_data = {place: scr.to_dict() for place, scr in self.xor_screening.items()}
        _write_json(xor_data, os.path.join(out, "xor_screening.json"))

        effect_data = {act: scr.to_dict() for act, scr in self.effect_screening.items()}
        _write_json(effect_data, os.path.join(out, "effect_screening.json"))


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _write_json(data: Any, path: str) -> None:
    """Write data as indented JSON to path."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
