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

import dataclasses
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Set, List

import numpy as np
from pm4py import PetriNet

import core_utils as utils
from models import (
    ActionDurationStats,
    AnalysisConfig,
    AttributeCatalogEntry,
    AttributeEffect,
    EffectGuards,
    EffectInfo,
    Guard,
    ParseResult,
    PetriNetModel,
    PreprocessedLog,
    TransitionEffects,
    TransitionInfo,
    TransitionScreening,
    XorBranchInfo,
    XorSplitGuards,
    XorSplitScreening,
    XorSplitStats,
)
from .decision_mining import DecisionMiner
from .discretizer import Discretizer
from .effect_correlation_analyzer import EffectCorrelationAnalyzer
from .log_preprocessor import LogPreprocessor
from .log_processor import LogProcessor
from .model_discoverer import ModelDiscoverer
from .petri_net_log_builder import PetriNetLogBuilder
from .probability_estimator import ProbabilityEstimator
from .temporal_extractor import ExternalDuration, TemporalExtractor

SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

logger = utils.get_logger(__name__)

# Minimum probability for unobserved or low-probability XOR branches.
# Prevents -log(0) = infinity in PDDL cost calculations.
#TODO: add as config param
_PROB_FLOOR: float = 1e-3


@dataclasses.dataclass
class PreloadedLogAndNet:
    """Output of steps 1-2 (log loading/filtering + Petri net discovery) —
    the only two Parser steps that don't depend on AnalysisConfig, and so
    can be computed once and reused across trials that share the same
    log_path/coverage_percentage/discovery_algorithm/use_activity_classifier
    (see network_search/runner.py::find_best_config)."""

    log: Any
    full_log: Any
    full_lifecycle_log: Any
    attributes: Any
    attribute_categories: Any
    petri_net_model: PetriNetModel


def load_and_discover(
    log_path: str,
    coverage_percentage: float,
    discovery_algorithm: str = 'inductive',
    use_activity_classifier: bool = False,
) -> PreloadedLogAndNet:
    """Steps 1-2 of Parser.__init__, extracted verbatim so callers that run
    many trials against the same log/discovery settings (but varying
    AnalysisConfig) can compute this once — see PreloadedLogAndNet."""
    # --- Step 1: log loading and filtering ---
    logger.info("Loading and filtering log: %s", log_path)
    processor = LogProcessor(log_path, use_activity_classifier)
    log, full_log, full_lifecycle_log = processor.load_and_filter_log(coverage_percentage)
    attributes, attribute_categories = processor.initialize_attributes(full_log)
    logger.info("Log loaded: %d traces, %d attributes", len(log), len(attributes))

    # --- Step 2: Petri net discovery ---
    logger.info("Starting Petri net discovery [%s]", discovery_algorithm)
    petri_net_model: PetriNetModel = ModelDiscoverer(discovery_algorithm).discover(log)
    logger.info("Petri net discovered: %d places, %d transitions, %d XOR splits",
                len(petri_net_model.petrinet.places), len(petri_net_model.petrinet.transitions),
                len(petri_net_model.xor_splits))

    return PreloadedLogAndNet(
        log=log,
        full_log=full_log,
        full_lifecycle_log=full_lifecycle_log,
        attributes=attributes,
        attribute_categories=attribute_categories,
        petri_net_model=petri_net_model,
    )


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
        external_durations: Optional[Dict[str, ExternalDuration]] = None,
        discretizer_cache_path: Optional[Path] = None,
        force_rediscretize: bool = False,
        preloaded: Optional[PreloadedLogAndNet] = None,
    ) -> None:
        """Run the full analysis pipeline up to and including DT mining.

        Args:
            log_path: Path to the XES event log file.
            coverage_percentage: Minimum cumulative variant coverage for
                log filtering (e.g. 0.8 keeps traces covering 80% of cases).
            discovery_algorithm: Petri net discovery algorithm name.
                One of 'alpha', 'inductive', 'heuristics', 'ilp', 'powl'.
            use_activity_classifier: Combine concept:name + lifecycle:transition
                as activity label when True.
            config: Analysis configuration; defaults to AnalysisConfig() if None.
            external_durations: Optional user-supplied duration bounds per
                activity.  These take priority over log-derived estimates.
                Activities not listed fall back to lifecycle or inter-event data.
            preloaded: Result of load_and_discover(log_path, coverage_percentage,
                discovery_algorithm, use_activity_classifier) computed ahead of
                time — when given, steps 1-2 are skipped entirely and their
                results are taken from here instead. Callers that run many
                trials against the same log/discovery settings should compute
                this once and pass it to every Parser() call.
        """
        self.config = config or AnalysisConfig()

        if preloaded is None:
            preloaded = load_and_discover(
                log_path, coverage_percentage, discovery_algorithm, use_activity_classifier
            )
        self.log = preloaded.log
        self.full_log = preloaded.full_log
        self.full_lifecycle_log = preloaded.full_lifecycle_log
        self.attributes = preloaded.attributes
        self.attribute_categories = preloaded.attribute_categories
        self.petri_net_model = preloaded.petri_net_model
        pnm = self.petri_net_model

        # --- Step 3: numeric attribute discretization ---
        numeric_attrs = [
            a for a, t in self.attribute_categories.items() if t == 'numerical'
        ]
        logger.info("Starting discretization for %d numerical attributes", len(numeric_attrs))
        discretizer = Discretizer(self.config)
        discretizer.fit(
            self.full_log,
            numeric_attrs,
            cache_path=discretizer_cache_path,
            force=force_rediscretize,
        )
        self.discretizer = discretizer
        logger.info("Discretization complete: %d attributes discretized",
                    len(discretizer.boundaries))
        self.sanitized_value_catalog = {attr: discretizer.all_bin_labels(attr) for attr in discretizer.boundaries.keys()}

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
        self.pn_log = builder.build(self.full_lifecycle_log)
        logger.info("Token replay complete: %d traces accepted", len(self.pn_log.executions))

        # --- Step 4.5: duration extraction ---
        logger.info("Starting duration extraction")
        self.duration_stats: Dict[str, ActionDurationStats] = TemporalExtractor().extract(
            external_durations=external_durations,
            fallback_to_inter_event=True,
            petri_net_log=self.pn_log,
        )
        logger.info(
            "Duration extraction complete: %d activities with duration data",
            len(self.duration_stats),
        )

        # --- Step 5: single-pass log preprocessing ---
        logger.info("Starting log preprocessing")
        self.preprocessed_log: PreprocessedLog = LogPreprocessor(
            config=self.config,
            discretizer=discretizer,
            attribute_categories=self.attribute_categories
        ).preprocess(self.pn_log, pnm.xor_splits)
        _all_labeled = {t.label for t in pnm.petrinet.transitions if t.label is not None}
        _fired = set(self.preprocessed_log.transition_firings.keys())
        _never_fired = _all_labeled - _fired
        _n_tau = len(pnm.petrinet.transitions) - len(_all_labeled)
        logger.info(
            "Log preprocessing complete: %d/%d labeled transitions fired in training data, "
            "%d tau (silent) transitions in net; "
            "%d labeled transitions never fired: %s",
            len(_fired), len(_all_labeled), _n_tau,
            len(_never_fired), sorted(_never_fired) if _never_fired else "none",
        )

        # --- Step 5b: effect co-occurrence classification ---
        self._effect_cooccurrence = EffectCorrelationAnalyzer().analyze(
            self.preprocessed_log, self.config
        )

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

        # --- XOR split filtering → PetriNet pruning ---
        logger.info("Starting XOR split filtering")
        self._filter_xor_splits()
        logger.info("XOR split filtering complete")

        # --- Effect filtering → EffectInfo ---
        logger.info("Starting effect filtering")
        self._filter_effects()
        logger.info("Effect filtering complete")

        # --- Assemble ParseResult ---
        logger.info("Assembling ParseResult")
        self.parse_result: ParseResult = self._build_parse_result()
        logger.info("Pipeline complete")


    def _filter_xor_splits(self) -> None:
        """Assign XOR routing info to all branches without modifying the Petri net.

        The Petri net topology is never changed here: all branches produced by the
        discovery algorithm are preserved.  Routing decisions affect only PDDL action
        costs (via XorBranchInfo.cascade_level and probability), not net structure.

        Populates self._xor_branch_info: Dict[str, XorBranchInfo] for every branch
        of every XOR-split place.
        """
        self._xor_branch_info: Dict[str, List[XorBranchInfo]] = defaultdict(list)

        for place, transitions in list(self.petri_net_model.xor_splits.items()):
            place_name = place.name
            screening = self.xor_screening.get(place_name)
            if screening is None:
                continue


            pruned = set()
            if screening.action == "dt" and place_name in self.xor_guards:
                pruned = self._apply_dt_level(screening, place_name)

            elif screening.action == "deterministic":
                pruned = self._apply_deterministic(screening, place_name)

            elif screening.action == "fallback":
                pruned = self._apply_fallback(screening, place_name)
            # TODO remove pruned
        self._xor_branch_info = dict(self._xor_branch_info)


    def _apply_dt_level(
        self,
        screening: XorSplitScreening,
        place_name: str,
    ) -> Set[str]:
        """Level 1 — DT succeeded: assign guards or probabilistic fallback to each branch.

        All branches are kept in the net. Branches screened as 'pruned' (low
        probability) and non-pruned branches without DT guards (orphaned to fallback)
        receive cascade_level=2 with floor probability. Branches with DT guards
        receive cascade_level=1.
        """
        guards_obj = self.xor_guards[place_name]
        summary = {}
        pruned = set()
        for act_name, xor_branch in screening.branches.items():
            if xor_branch.status == "pruned":
                pruned.add(act_name)
            elif xor_branch.status == "fallback":
                self._xor_branch_info[act_name].append(XorBranchInfo(
                    place_name=place_name,
                    probability=xor_branch.probability if xor_branch else 0.0,
                    total_samples=screening.total_samples,
                    cascade_level=2,
                    guards=None,
                    routing_source="probabilistic",
                ))
            else:
                activity_guards = guards_obj.guards.get(act_name)
                if activity_guards:
                    self._xor_branch_info[act_name].append(XorBranchInfo(
                        place_name=place_name,
                        probability=xor_branch.probability,
                        total_samples=screening.total_samples,
                        cascade_level=1,
                        guards=activity_guards,
                        routing_source="dt",
                    ))
                    summary[act_name] = f"DT(p={xor_branch.probability:.3f}, {len(activity_guards)} guard(s))"
                else: #probabilistic fallback for no guards
                    self._xor_branch_info[act_name].append(XorBranchInfo(
                        place_name=place_name,
                        probability=xor_branch.probability if xor_branch else 0.0,
                        total_samples=screening.total_samples,
                        cascade_level=2,
                        guards=None,
                        routing_source="probabilistic",
                    ))

        logger.debug(
            "[XOR '%s'] DT applied — branch decisions: %s",
            place_name, summary,
        )
        return pruned

    def _apply_deterministic(
        self,
        screening: XorSplitScreening,
        place_name: str,
    ) -> Set[str]:
        """Level 1 — Single certain branch: floor cost for all non-certain branches.

        No branch is removed from the net. The certain branch gets cascade_level=1
        with its observed probability. All other branches get cascade_level=2 with
        floor probability so the planner can still consider them at high cost.
        """
        summary = {}
        pruned = set()
        for act_name, xor_branch in screening.branches.items():
            if xor_branch.status == "certain":
                self._xor_branch_info[act_name].append(XorBranchInfo(
                    place_name=place_name,
                    probability=xor_branch.probability,
                    total_samples=screening.total_samples,
                    cascade_level=1,
                    guards=None,
                    routing_source="deterministic",
                ))
                summary[act_name] = f"certain(p={xor_branch.probability:.3f}, level=1)"
            elif xor_branch.status == "pruned":
                pruned.add(act_name)

        logger.debug(
            "[XOR '%s'] DETERMINISTIC applied — branch decisions: %s",
            place_name, summary,
        )
        return pruned


    def _apply_fallback(
        self,
        screening: XorSplitScreening,
        place_name: str,
    ) -> Set[str]:
        """Level 2/3 — Statistical fallback: assign costs to all branches.

        No branch is removed from the net. Level 3 (equal weights) is used when
        total_samples < probability_min_samples. Level 2 applies xor_statistical_mode.
        """
        pruned = set()
        if screening.total_samples < self.config.probability_min_samples:
            logger.debug(
                "[XOR '%s'] FALLBACK level=3 — %d samples < min %d; "
                "equal weight assigned to all %d branches (no routing preference).",
                place_name, screening.total_samples,
                self.config.probability_min_samples, len(screening.branches.items()),
            )

            for act_name, xor_branch in screening.branches.items():
                if xor_branch.status == "pruned":
                    pruned.add(act_name)
                else:
                    self._xor_branch_info[act_name].append(XorBranchInfo(
                        place_name=place_name,
                        probability=xor_branch.probability,
                        total_samples=screening.total_samples,
                        cascade_level=3,
                        guards=None,
                        routing_source="equal_weight",
                    ))
            return pruned

        mode = self.config.xor_statistical_mode
        summary = {}

        if mode == "majority_only":
            max_act = None
            max_prob = 0.0
            for act_name, xor_branch in screening.branches.items():
                if xor_branch.status == "pruned":
                    pruned.add(act_name)
                else:
                    prob = xor_branch.probability
                    if prob > max_prob:
                        max_act = act_name
                        max_prob = prob
            self._xor_branch_info[max_act].append(XorBranchInfo(
                place_name=place_name,
                probability=max_prob,
                total_samples=screening.total_samples,
                cascade_level=2,
                guards=None,
                routing_source="probabilistic",
            ))

        elif mode == "weighted":
            for act_name, xor_branch in screening.branches.items():
                if xor_branch.status == "pruned":
                    pruned.add(act_name)
                else:
                    self._xor_branch_info[act_name].append(XorBranchInfo(
                        place_name=place_name,
                        probability=xor_branch.probability,
                        total_samples=screening.total_samples,
                        cascade_level=2,
                        guards=None,
                        routing_source="probabilistic",
                    ))

        logger.debug(
            "[XOR '%s'] FALLBACK level=2, mode='%s' — branch decisions: %s",
            place_name, mode, summary,
        )
        return pruned


    def _filter_effects(self) -> None:
        """Build encoder-ready EffectInfo for each (transition, attribute) pair.

        Reads effect_screening and transition_effects; applies cascade levels:
          1 = DT guards available
          2 = statistical fallback (probabilities only)
          3 = effect removed

        Level-3 triggers:
          - total_firings < min_samples: the transition has too few data points to
            trust any effect probability → all attributes excluded for this transition.
          - appearance_action == "never": this specific attribute changes too rarely
            to be a real effect → only that attribute's effect is excluded. A low
            appearance_samples count alone does not exclude an attribute — it only
            downgrades appearance_level to 2 (statistical fallback).

        Value level 3 → value_probabilities is empty, value_guards is None.

        Populates self._transition_effect_info:
            Dict[str, Dict[str, EffectInfo]]  (activity_name → attr → EffectInfo)
        """
        self._transition_effect_info: Dict[str, Dict[str, EffectInfo]] = {}
        min_samples = self.config.probability_min_samples

        for act, t_scr in self.effect_screening.items():
            #fallback 3 on appearance
            # Transition-level gate: too few firings → no reliable effect data at all.
            if t_scr.total_firings < min_samples:
                continue

            attr_infos: Dict[str, EffectInfo] = {}
            t_effects = self.transition_effects.get(act)
            attr_effects = self.attribute_effects.get(act)

            for attr, attr_scr in t_scr.attributes.items():
                if attr_scr.appearance_action == "never":
                    continue

                # --- Appearance level ---
                if attr_scr.appearance_action == "dt":
                    appearance_level = 1
                    appearance_guards = (
                        t_effects.effects[attr].appearance
                        if t_effects and attr in t_effects.effects
                        else None
                    )
                elif attr_scr.appearance_action == "deterministic":
                    appearance_level = 1
                    appearance_guards = None
                else:  # "fallback"
                    appearance_level = 2
                    appearance_guards = None

                # --- Value level ---
                if attr_scr.value_action == "dt":
                    value_level = 1
                    value_guards = (
                        t_effects.effects[attr].value
                        if t_effects and attr in t_effects.effects
                        else None
                    )
                elif attr_scr.value_action == "deterministic":
                    value_level = 1
                    value_guards = None
                else:  # "fallback"
                    value_level = 2
                    value_guards = None

                # --- Active value probabilities (pruned values excluded) ---
                active_value_probs: Dict[Any, float] = {}
                if attr_effects and attr in attr_effects.value_probabilities:
                    all_probs = attr_effects.value_probabilities[attr]
                    for v, v_scr in attr_scr.values.items():
                        if v_scr.status != "pruned" and v in all_probs:
                            active_value_probs[v] = all_probs[v]

                attr_infos[attr] = EffectInfo(
                    attribute=attr,
                    presence_probability=attr_scr.presence_probability,
                    appearance_level=appearance_level,
                    appearance_guards=appearance_guards,
                    value_probabilities=active_value_probs,
                    value_level=value_level,
                    value_guards=value_guards,
                )

            if attr_infos:
                effect_summary = {}
                for attr, info in attr_infos.items():
                    app = "DT" if info.appearance_guards else ("det" if info.appearance_level == 1 else "fallback")
                    val = "DT" if info.value_guards else ("det" if info.value_level == 1 else "fallback")
                    effect_summary[attr] = f"app={app},val={val}(p={info.presence_probability:.2f})"
                logger.debug(
                    "[effects '%s'] %d attribute effect(s) resolved: %s",
                    act, len(attr_infos), effect_summary,
                )
                self._transition_effect_info[act] = attr_infos

    def _build_parse_result(self) -> ParseResult:
        """Assemble the final encoder-ready ParseResult from all pipeline outputs.

        Builds place/transition predecessor maps from the pruned PetriNetModel
        indexes, then constructs a TransitionInfo for every labeled transition,
        wiring in XOR branch info and conditional effects.

        Returns:
            ParseResult with consistent pruned-net indexes and encoder-ready info.
        """
        pnm = self.petri_net_model

        # Place predecessors: place.name → [sanitized labels of transitions
        # whose output arcs lead into that place]
        place_predecessors: Dict[str, List[str]] = {
            p.name: [
                t.label
                for t in trans_list if t.label
            ]
            for p, trans_list in pnm.place_inputs.items()
        }

        # Transition predecessors: sanitized label → [input place names]
        transition_predecessors: Dict[str, List[str]] = {
            t.label: [p.name for p in places]
            for t, places in pnm.trans_inputs.items()
            if t.label
        }

        # TransitionInfo per labeled transition
        transitions: Dict[str, TransitionInfo] = {}
        for t, input_places in pnm.trans_inputs.items():
            if not t.label:
                continue
            act = t.label
            attr_effects = self.attribute_effects.get(act)
            cooccurrence = self._effect_cooccurrence.get(act, (set(), set()))
            transitions[act] = TransitionInfo(
                activity_name=act,
                input_places=[p.name for p in input_places],
                total_firings=attr_effects.total_firings if attr_effects else 0,
                xor_branches=self._xor_branch_info.get(act),
                effects=self._transition_effect_info.get(act, {}),
                duration=self.duration_stats.get(act),
                related_effects=cooccurrence[0],
                incompatible_effects=cooccurrence[1],
            )

        # TransitionInfo for tau transitions that survived XOR screening
        # (i.e., appear in _xor_branch_info).  These become first-class entries
        # so XorBranchProcessor, ActionBuilder, and the serializer can treat
        # them uniformly alongside labeled transitions.
        for trans_obj, tau_label in pnm.silent_transitions.items():
            tau_name = tau_label
            if tau_name not in self._xor_branch_info:
                continue
            input_places_obj = pnm.trans_inputs.get(trans_obj, set())
            tau_firings = sum(
                1
                for fds in self.preprocessed_log.xor_firings.values()
                for fd in fds
                if fd.activity_name == tau_name
            )
            transitions[tau_name] = TransitionInfo(
                activity_name=tau_name,
                input_places=[p.name for p in input_places_obj],
                total_firings=tau_firings,
                xor_branches=self._xor_branch_info[tau_name],
                effects={},
                is_tau=True,
            )
            transition_predecessors[tau_name] = [p.name for p in input_places_obj]

        # Start / end place names from markings (single-place markings assumed)
        start_place = next(iter(pnm.initial_marking)).name
        end_place = next(iter(pnm.final_marking)).name

        return ParseResult(
            petri_net_model=pnm,
            place_predecessors=place_predecessors,
            transition_predecessors=transition_predecessors,
            transitions=transitions,
            start_place=start_place,
            end_place=end_place,
            attribute_catalog=self._build_attribute_catalog(),
            negated_attributes=self._collect_negated_attributes(),
        )


    def _build_attribute_catalog(self) -> Dict[str, AttributeCatalogEntry]:
        """Build the filtered attribute catalog from surviving effects and guards.

        Collects every attribute (and its observed/guard values) that appears in:
          - EffectInfo.value_probabilities  (values a transition can write)
          - appearance_guards / value_guards (DT conditions on effects)
          - XorBranchInfo.guards            (DT conditions on XOR routing)

        Attributes and values pruned or filtered out during the cascade are
        absent.  Boolean guard conditions (Guard.value is None) contribute the
        attribute name but no explicit value.

        Returns:
            Dict mapping sanitized attribute name to AttributeCatalogEntry.
        """
        attr_values: Dict[str, Set[Any]] = {}

        def _touch(attr: str) -> None:
            if attr not in attr_values:
                attr_values[attr] = set()

        def _add_value(attr: str, value: Any) -> None:
            _touch(attr)
            if value is not None:
                attr_values[attr].add(value)

        def _scan_sop(sop: List[List[Guard]]) -> None:
            for path in sop:
                for g in path:
                    _add_value(g.attribute, g.value)

        def _scan_effect_guards(eg: Optional[EffectGuards]) -> None:
            if eg is None:
                return
            for sop in eg.guards.values():
                _scan_sop(sop)

        # Seed from the pre-computed sanitized value catalog (Step 3.5).
        # Guarantees all valid values — including unobserved numerical bins and
        # every categorical value in the full log — are present from the start.
        for attr, values in self.sanitized_value_catalog.items():
            for v in values:
                _add_value(attr, v)

        # Collect from effect info (already filtered by _filter_effects)
        for attr_infos in self._transition_effect_info.values():
            for attr, eff in attr_infos.items():
                _touch(attr)
                for v in eff.value_probabilities:
                    _add_value(attr, v)
                _scan_effect_guards(eff.appearance_guards)
                _scan_effect_guards(eff.value_guards)

        # Collect from XOR branch guards (already filtered by _filter_xor_splits)
        for branch_info_list in self._xor_branch_info.values():
            for branch_info in branch_info_list:
                if branch_info.guards is not None:
                    _scan_sop(branch_info.guards)

        return {
            attr: AttributeCatalogEntry(
                attribute_type=self.attribute_categories.get(attr, 'categorical'),
                possible_values=values,
                bin_boundaries=(
                    self.discretizer.boundaries.get(attr)
                    if self.attribute_categories.get(attr) == 'numerical'
                    else None
                ),
            )
            for attr, values in attr_values.items()
        }

    def _collect_negated_attributes(self) -> Set[str]:
        """Collect attributes that appear with negated=True in any guard.

        Scans all surviving guards — XOR branch guards and effect guards — and
        returns the set of attribute names where at least one Guard has
        negated=True.  The encoder uses this set to decide which attributes
        need negative predicates (_is_not / _false).

        Returns:
            Set of sanitized attribute names that require negative predicates.
        """
        negated: Set[str] = set()

        def _scan_sop(sop: List[List[Guard]]) -> None:
            for path in sop:
                for g in path:
                    if g.negated:
                        negated.add(g.attribute)

        def _scan_effect_guards(eg: Optional[EffectGuards]) -> None:
            if eg is None:
                return
            for sop in eg.guards.values():
                _scan_sop(sop)

        for attr_infos in self._transition_effect_info.values():
            for eff in attr_infos.values():
                _scan_effect_guards(eff.appearance_guards)
                _scan_effect_guards(eff.value_guards)

        for branch_info_list in self._xor_branch_info.values():
            for branch_info in branch_info_list:
                if branch_info.guards is not None:
                    _scan_sop(branch_info.guards)

        return negated


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _write_json(data: Any, path: str) -> None:
    """Write data as indented JSON to path."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
