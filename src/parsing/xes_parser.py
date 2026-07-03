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
from pathlib import Path
from typing import Any, Dict, Optional, Set, List

import numpy as np
import pm4py
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
        discretizer.fit(
            self.full_log,
            numeric_attrs,
            cache_path=discretizer_cache_path,
            force=force_rediscretize,
        )
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
        ).preprocess(self.pn_log, pnm.xor_splits)
        logger.info("Log preprocessing complete")

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

        # --- Snapshot: persist pre-pruning state to disk ---
        logger.info("Saving pre-pruning snapshot")
        self._save_snapshot()
        logger.info("Snapshot saved to %s", self.config.snapshot_dir)

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

    def _filter_xor_splits(self) -> None:
        """Apply XOR split decisions: prune branches and record cascade info.

        Iterates xor_screening to determine which branch transitions to remove
        from the Petri net and what cascade level each remaining branch carries.
        All removals are accumulated and applied in a single prune_transitions()
        call to keep the net consistent.

        Populates self._xor_branch_info: Dict[str, XorBranchInfo] mapping each
        remaining branch's activity name to its encoder-ready routing info.
        """
        to_remove: Set[PetriNet.Transition] = set()
        self._xor_branch_info: Dict[str, XorBranchInfo] = {}

        for place, transitions in list(self.petri_net_model.xor_splits.items()):
            place_name = place.name
            screening = self.xor_screening.get(place_name)
            if screening is None:
                continue

            # Include both labeled and silent transitions so that silent
            # (tau) branches are correctly pruned from the net.
            all_trans: Dict[str, PetriNet.Transition] = {}
            for t in transitions:
                if t.label:
                    all_trans[utils.sanitize_name(t.label)] = t
                else:
                    silent_name = self.petri_net_model.silent_transitions.get(t)
                    if silent_name:
                        all_trans[silent_name] = t

            if screening.action == "dt" and place_name in self.xor_guards:
                self._apply_dt_level(all_trans, screening, place_name, to_remove)

            elif screening.action == "deterministic":
                self._apply_deterministic(all_trans, screening, to_remove)

            elif screening.action == "fallback":
                self._apply_fallback(all_trans, screening, to_remove)

        if to_remove:
            logger.info("Pruning %d branch transitions from Petri net", len(to_remove))
            self.petri_net_model = self.petri_net_model.prune_transitions(to_remove)

    def _apply_dt_level(
        self,
        all_trans: Dict[str, PetriNet.Transition],
        screening: XorSplitScreening,
        place_name: str,
        to_remove: Set[PetriNet.Transition],
    ) -> None:
        """Level 1 — DT succeeded: prune only 'pruned' branches.

        Includes silent (tau) transitions so they are correctly removed when
        their screening status is 'pruned'.  Non-pruned branches (labeled or
        tau) all receive XorBranchInfo so tau XOR branches become first-class
        entries in result.transitions.
        """
        guards_obj = self.xor_guards[place_name]
        for act, t in all_trans.items():
            branch = screening.branches.get(act)
            if branch is None or branch.status == "pruned":
                to_remove.add(t)
            else:
                self._xor_branch_info[act] = XorBranchInfo(
                    probability=branch.probability,
                    total_samples=screening.total_samples,
                    cascade_level=1,
                    guards=guards_obj.guards.get(act),
                )

    def _apply_deterministic(
        self,
        all_trans: Dict[str, PetriNet.Transition],
        screening: XorSplitScreening,
        to_remove: Set[PetriNet.Transition],
    ) -> None:
        """Level 1 — Single certain branch: prune everything else.

        Includes silent (tau) transitions so they are removed when pruned.
        The certain branch is always labeled; silent branches are never certain.
        """
        for act, t in all_trans.items():
            branch = screening.branches.get(act)
            if branch is None or branch.status != "certain":
                to_remove.add(t)


    def _apply_fallback(
        self,
        all_trans: Dict[str, PetriNet.Transition],
        screening: XorSplitScreening,
        to_remove: Set[PetriNet.Transition],
    ) -> None:
        """Level 2/3 — Statistical fallback: apply xor_statistical_mode.

        Level 3 (equal weights, no pruning) is used when total_samples is below
        config.probability_min_samples — probabilities are not reliable enough
        to make any routing decision.  Level 2 applies the configured
        xor_statistical_mode on the active (non-pruned) branches.

        Includes silent (tau) transitions so they are correctly pruned.  Both
        labeled and tau non-pruned branches receive XorBranchInfo so tau XOR
        branches become first-class entries in result.transitions.
        """
        if screening.total_samples < self.config.probability_min_samples:
            # Level 3: too few samples to trust any probability — keep all
            # branches with equal weight and do not remove anything.
            for act, t in all_trans.items():
                branch = screening.branches.get(act)
                self._xor_branch_info[act] = XorBranchInfo(
                    probability=branch.probability if branch else 0.0,
                    total_samples=screening.total_samples,
                    cascade_level=3,
                    guards=None,
                )
            return

        active = {
            act: b for act, b in screening.branches.items()
            if b.status != "pruned"
        }

        if not active:
            # Edge case: samples are sufficient but all branches are below
            # xor_prune_threshold.  Keep everything to avoid an empty net.
            logger.warning(
                "XOR split: all branches pruned despite sufficient samples "
                "(%d) — keeping all branches to avoid empty split.",
                screening.total_samples,
            )
            for act, t in all_trans.items():
                branch = screening.branches.get(act)
                self._xor_branch_info[act] = XorBranchInfo(
                    probability=branch.probability if branch else 0.0,
                    total_samples=screening.total_samples,
                    cascade_level=2,
                    guards=None,
                )
            return

        mode = self.config.xor_statistical_mode

        if mode == "majority_only":
            majority = max(active, key=lambda a: active[a].probability)
            for act, t in all_trans.items():
                if act != majority:
                    to_remove.add(t)
                else:
                    self._xor_branch_info[act] = XorBranchInfo(
                        probability=active[act].probability,
                        total_samples=screening.total_samples,
                        cascade_level=2,
                        guards=None,
                    )

        elif mode == "weighted":
            # Keep all branches; encoder will assign costs via -log(p).
            for act, t in all_trans.items():
                branch = screening.branches.get(act)
                self._xor_branch_info[act] = XorBranchInfo(
                    probability=branch.probability if branch else 0.0,
                    total_samples=screening.total_samples,
                    cascade_level=2,
                    guards=None,
                )

        else:
            # pruned_weighted (default): remove 'pruned', keep active with costs.
            for act, t in all_trans.items():
                branch = screening.branches.get(act)
                if branch is None or branch.status == "pruned":
                    to_remove.add(t)
                else:
                    self._xor_branch_info[act] = XorBranchInfo(
                        probability=branch.probability,
                        total_samples=screening.total_samples,
                        cascade_level=2,
                        guards=None,
                    )


    def _filter_effects(self) -> None:
        """Build encoder-ready EffectInfo for each (transition, attribute) pair.

        Reads effect_screening and transition_effects; applies cascade levels:
          1 = DT guards available
          2 = statistical fallback (probabilities only)
          3 = effect removed

        Two independent level-3 triggers:
          - total_firings < min_samples: the transition has too few data points to
            trust any effect probability → all attributes excluded for this transition.
          - appearance_samples < min_samples: this specific attribute changed too
            rarely → only that attribute's effect is excluded.

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
                    if attr_scr.appearance_samples < min_samples:
                        # fallback 3 for value effect
                        continue  # this attr changed too rarely → remove effect
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
                utils.sanitize_name(t.label)
                for t in trans_list if t.label
            ]
            for p, trans_list in pnm.place_inputs.items()
        }

        # Transition predecessors: sanitized label → [input place names]
        transition_predecessors: Dict[str, List[str]] = {
            utils.sanitize_name(t.label): [p.name for p in places]
            for t, places in pnm.trans_inputs.items()
            if t.label
        }

        # TransitionInfo per labeled transition
        transitions: Dict[str, TransitionInfo] = {}
        for t, input_places in pnm.trans_inputs.items():
            if not t.label:
                continue
            act = utils.sanitize_name(t.label)
            attr_effects = self.attribute_effects.get(act)
            cooccurrence = self._effect_cooccurrence.get(act, (set(), set()))
            transitions[act] = TransitionInfo(
                activity_name=act,
                input_places=[p.name for p in input_places],
                total_firings=attr_effects.total_firings if attr_effects else 0,
                xor_branch=self._xor_branch_info.get(act),
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
            tau_name = utils.sanitize_name(tau_label)
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
                xor_branch=self._xor_branch_info[tau_name],
                effects={},
                is_tau=True,
            )
            transition_predecessors[tau_name] = [p.name for p in input_places_obj]

        # Start / end place names from markings (single-place markings assumed)
        start_place = next(iter(pnm.initial_marking)).name
        end_place = next(iter(pnm.final_marking)).name

        xor_virtual_taus = self._inject_xor_taus(transitions, transition_predecessors)

        return ParseResult(
            petri_net_model=pnm,
            place_predecessors=place_predecessors,
            transition_predecessors=transition_predecessors,
            transitions=transitions,
            start_place=start_place,
            end_place=end_place,
            attribute_catalog=self._build_attribute_catalog(),
            negated_attributes=self._collect_negated_attributes(),
            xor_virtual_taus=xor_virtual_taus,
        )


    def _inject_xor_taus(
        self,
        transitions: Dict[str, TransitionInfo],
        transition_predecessors: Dict[str, List[str]],
    ) -> Set[str]:
        """Inject virtual tau transitions for XOR cascade-level-2 branches that
        also carry appearance-level-2 effects.

        When both XOR-branch cost and effect-appearance cost apply to the same
        transition, they would conflict if stored on a single action.  A virtual
        tau is interposed to carry only the XOR cost, freeing the real transition
        to carry only the effect costs.

        The tau inherits the original transition's XOR branch info (so
        XorBranchProcessor assigns the cost to it) while the original transition
        has its xor_branch cleared and its input redirected through the tau.

        Condition for injection:
            cascade_level == 2  AND  any effect has appearance_level == 2

        Args:
            transitions: Mutable dict of TransitionInfo, modified in place.
            transition_predecessors: Mutable predecessor map, modified in place.

        Returns:
            Set of injected tau names (empty if no injection occurred).
        """
        injected: Set[str] = set()

        for trans_name, info in list(transitions.items()):
            if info.xor_branch is None or info.xor_branch.cascade_level != 2:
                continue
            if not any(e.appearance_level == 2 for e in info.effects.values()):
                continue

            tau_name = f"xor_tau_{trans_name}"
            logger.debug("Injecting XOR tau '%s' before '%s'", tau_name, trans_name)

            transitions[tau_name] = TransitionInfo(
                activity_name=tau_name,
                input_places=list(info.input_places),
                total_firings=info.total_firings,
                xor_branch=info.xor_branch,
                effects={},
            )

            info.xor_branch = None
            info.input_places = [tau_name]

            transition_predecessors[tau_name] = list(transition_predecessors[trans_name])
            transition_predecessors[trans_name] = [tau_name]

            injected.add(tau_name)

        return injected

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

        # Collect from effect info (already filtered by _filter_effects)
        for attr_infos in self._transition_effect_info.values():
            for attr, eff in attr_infos.items():
                _touch(attr)
                for v in eff.value_probabilities:
                    _add_value(attr, v)
                _scan_effect_guards(eff.appearance_guards)
                _scan_effect_guards(eff.value_guards)

        # Collect from XOR branch guards (already filtered by _filter_xor_splits)
        for branch_info in self._xor_branch_info.values():
            if branch_info.guards is not None:
                _scan_sop(branch_info.guards)

        # Enrich categorical possible_values with every value observed in the
        # training log, so values that survive variant filtering but are not
        # selected by the DT/effects pipeline are still recognized at replay time.
        _cat_attrs = {
            a for a, t in self.attribute_categories.items() if t == "categorical"
        }
        for trace in self.log:
            for event in trace:
                for raw_key, raw_val in event.items():
                    attr = utils.sanitize_name(raw_key)
                    if attr in attr_values and attr in _cat_attrs and raw_val is not None:
                        attr_values[attr].add(utils.sanitize_name(str(raw_val)))

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

        for branch_info in self._xor_branch_info.values():
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
