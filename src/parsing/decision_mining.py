"""Decision tree mining for XOR-split guards and conditional effects.

Operates on a PreprocessedLog (produced by LogPreprocessor) instead of
scanning the raw PetriNetLog.  All sanitization, discretization, and
ignored-attribute filtering has already been performed during preprocessing.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from sklearn.tree import DecisionTreeClassifier, _tree
from pm4py import PetriNet
from pm4py.objects.powl.obj import Transition

import core_utils as utils
from models import (
    AnalysisConfig,
    AttributeEffect,
    ConditionalEffect,
    EffectAttrScreening,
    EffectGuards,
    EffectValueScreening,
    Guard,
    PreprocessedLog,
    TransitionEffects,
    TransitionScreening,
    XorBranchScreening,
    XorSplitGuards,
    XorSplitScreening,
    XorSplitStats,
)

logger = utils.get_logger(__name__)


class DecisionMiner:
    """Mines decision logic for XOR splits and conditional effects using DTs.

    Operates on a PreprocessedLog whose TransitionFiringData objects already
    contain sanitized/discretized pre_state and changed_attrs.  This class
    is a pure feature-matrix builder + DT trainer — no log scanning.

    Three-stage pipeline:
    1. screen_xor_splits() / screen_effects() — classify each analysis target
       (dt / deterministic / fallback / never) and tag branches/values
       as active / pruned / certain.
    2. mine_xor_splits() / mine_effects() — train DTs only for targets marked
       "dt" by the screening, filtering pruned branches/values from training.
    3. After training, accuracy below config.dt_min_accuracy triggers a
       probability-based fallback ("fallback" status).
    """

    def __init__(
        self,
        silent_transitions: Dict[Transition, str],
        config: Optional[AnalysisConfig] = None,
    ) -> None:
        self.silent_transitions = silent_transitions
        self.config = config or AnalysisConfig()

    # -------------------------------------------------------------------------
    # Public API — XOR screening
    # -------------------------------------------------------------------------

    def screen_xor_splits(
        self,
        xor_splits: Dict[PetriNet.Place, List[Transition]],
        xor_stats: Dict[str, XorSplitStats],
    ) -> Dict[str, XorSplitScreening]:
        """Screen each XOR-split place and classify its branches.

        For each XOR place, branches with probability below
        config.xor_prune_threshold are marked "pruned". If only one active
        branch remains it is marked "certain" and the action is "deterministic".

        Args:
            xor_splits: XOR-split places mapped to their outgoing transitions.
            xor_stats: Per-place branch probabilities from ProbabilityEstimator.

        Returns:
            Dict mapping place name to XorSplitScreening.
        """
        result: Dict[str, XorSplitScreening] = {}

        for place, transitions in xor_splits.items():
            stats = xor_stats.get(place.name)
            total_samples = stats.total_executions if stats else 0

            raw: Dict[str, Tuple[str, float]] = {}
            for t in transitions:
                act = self._activity_name(t)
                prob = stats.probabilities.get(act, 0.0) if stats else 0.0
                if prob < self.config.xor_prune_threshold:
                    status = "pruned"
                elif prob < self.config.dt_min_prob_to_use or prob * total_samples < self.config.dt_min_samples:
                    status = "fallback"
                else:
                    status = "active"
                raw[act] = (status, prob)

            active_acts = [a for a, (s, _) in raw.items() if s == "active"]

            if len(active_acts) == 1 and all(s != "fallback" for (s, _) in raw.values()):
                sole = active_acts[0]
                raw[sole] = ("certain", raw[sole][1])

            branches = {
                act: XorBranchScreening(activity_name=act, probability=prob, status=status)
                for act, (status, prob) in raw.items()
            }

            if len(active_acts) == 0:
                action = "fallback"
            elif len(active_acts) == 1:
                action = "deterministic"
            elif total_samples < self.config.dt_min_samples:
                action = "fallback"
            else:
                action = "dt"

            if action == "deterministic":
                certain_act = next(a for a, b in branches.items() if b.status == "certain")
                others = [a for a in branches if a != certain_act]
                logger.debug(
                    "[XOR '%s'] DETERMINISTIC — '%s' is the only active branch (p=%.3f); "
                    "%d branch(es) kept in net with floor cost: %s",
                    place.name, certain_act, branches[certain_act].probability,
                    len(others), others,
                )
            elif action == "fallback":
                if total_samples == 0:
                    logger.debug(
                        "[XOR '%s'] FALLBACK — no replay observations; "
                        "all %d branches receive equal weight (cascade_level=3).",
                        place.name, len(branches),
                    )
                elif total_samples < self.config.dt_min_samples:
                    logger.debug(
                        "[XOR '%s'] FALLBACK — insufficient samples (%d < %d); "
                        "probabilistic costs assigned to all %d branches.",
                        place.name, total_samples, self.config.dt_min_samples, len(branches),
                    )
                else:
                    logger.debug(
                        "[XOR '%s'] FALLBACK — all branches below xor_prune_threshold (%.2f); "
                        "probabilistic costs assigned. branches: %s",
                        place.name, self.config.xor_prune_threshold,
                        {a: f"p={b.probability:.3f}" for a, b in branches.items()},
                    )
            else:  # "dt"
                active_acts_log = [a for a, b in branches.items() if b.status == "active"]
                low_prob = [a for a, b in branches.items() if b.status == "pruned"]
                logger.debug(
                    "[XOR '%s'] DT candidate — %d active branch(es) %s, "
                    "%d low-probability branch(es) excluded from training %s "
                    "(xor_prune_threshold=%.2f).",
                    place.name, len(active_acts_log), active_acts_log,
                    len(low_prob), low_prob, self.config.xor_prune_threshold,
                )
            result[place.name] = XorSplitScreening(
                total_samples=total_samples,
                action=action,
                branches=branches,
            )

        return result

    # -------------------------------------------------------------------------
    # Public API — XOR mining
    # -------------------------------------------------------------------------

    def mine_xor_splits(
        self,
        preprocessed_log: PreprocessedLog,
        xor_screening: Dict[str, XorSplitScreening],
    ) -> Dict[str, XorSplitGuards]:
        """Train a decision tree for each XOR split screened as "dt".

        Pruned branches are excluded from the DT training data. Splits whose
        DT accuracy falls below config.dt_min_accuracy are omitted from the
        result (use probability-based fallback).

        Args:
            preprocessed_log: Single-pass preprocessed log.
            xor_screening: Screening results from screen_xor_splits().

        Returns:
            Dict mapping place name to XorSplitGuards (only "dt" successes).
        """
        result: Dict[str, XorSplitGuards] = {}

        for place_name, screening in xor_screening.items():
            if screening.action != "dt":
                continue

            active_branches = {
                name for name, branch in screening.branches.items()
                if branch.status == "active"
            }
            fallback_branches = {name for name, branch in screening.branches.items() if branch.status == "fallback"}

            X, y = self._build_feature_matrix(
                preprocessed_log, place_name, active_branches
            )
            if X.empty or y.nunique() < 2:
                logger.debug(
                    "[XOR '%s'] DT skipped — feature matrix has %d rows and %d distinct class(es) "
                    "after excluding low-probability branches (need ≥2 classes). "
                    "Falling back to probabilistic costs.",
                    place_name, len(X), y.nunique() if not X.empty else 0,
                )
                screening.action = "fallback"
                continue

            guards, accuracy = self._train_and_extract(X, y)

            if accuracy < self.config.dt_min_accuracy:
                logger.debug(
                    "[XOR '%s'] DT accuracy %.3f < threshold %.2f — "
                    "guards discarded, falling back to probabilistic costs.",
                    place_name, accuracy, self.config.dt_min_accuracy,
                )
                screening.action = "fallback"
                continue
            for name in fallback_branches:
                guards.pop(name, [])
            if not guards:
                logger.debug(
                    "[XOR '%s'] DT produced no guards — falling back to probabilistic costs.",
                    place_name,
                )
                screening.action = "fallback"
                continue

            result[place_name] = XorSplitGuards(
                guards=guards,
                total_samples=screening.total_samples,
                dt_accuracy=round(accuracy, 4),
            )
            logger.debug(
                "[XOR '%s'] DT accepted — accuracy=%.3f, guards produced for %d branch(es): %s.",
                place_name, accuracy, len(guards), sorted(guards.keys()),
            )

        return result

    # -------------------------------------------------------------------------
    # Private: XOR feature matrix
    # -------------------------------------------------------------------------

    def _build_feature_matrix(
        self,
        preprocessed_log: PreprocessedLog,
        place_name: str,
        active_branches: Optional[Set[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Build the training feature matrix for one XOR-split place.

        Each TransitionFiringData in xor_firings provides one training row:
        pre_state becomes the feature vector, activity_name becomes the label.

        Args:
            preprocessed_log: Single-pass preprocessed log with xor_firings.
            place_name: Name of the XOR-split place to sample.
            active_branches: If provided, only include firings whose
                activity_name is in this set (pruned branches excluded).

        Returns:
            Tuple (X, y): feature DataFrame and label Series.
        """
        firings = preprocessed_log.xor_firings.get(place_name, [])
        if not firings:
            return pd.DataFrame(), pd.Series(dtype=str)

        if active_branches is not None:
            firings = [fd for fd in firings if fd.activity_name in active_branches]

        if not firings:
            return pd.DataFrame(), pd.Series(dtype=str)

        rows = [fd.pre_state for fd in firings]
        labels = [fd.activity_name for fd in firings]

        return pd.DataFrame(rows), pd.Series(labels, dtype=str)

    # -------------------------------------------------------------------------
    # Private: DT training and guard extraction
    # -------------------------------------------------------------------------

    def _train_and_extract(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[Dict[str, List[List[Guard]]], float]:
        """Encode features, train a DT, compute accuracy, and extract SOP guards.

        Detection rules for column types (applied to each column of X):
        - dtype bool, or binary values in {0, 1} → boolean (cast to int, no dummies)
        - dtype object / string → categorical (one-hot via get_dummies)
        - numeric (float/int, not binary) → left as-is; raw numerics produce no
          PDDL conditions and are excluded from guard extraction

        Args:
            X: Feature DataFrame with sanitized column names.
            y: Label Series (branch activity names).

        Returns:
            Tuple (guards, accuracy):
            - guards: Dict[activity, List[List[Guard]]] in SOP form.
            - accuracy: DT training accuracy (0.0 if training fails).
        """
        X = X.copy()
        X.dropna(axis=1, how="all", inplace=True)
        if X.empty:
            return {}, 0.0

        bool_cols: Set[str] = set()
        categorical_cols: Set[str] = set()

        for col in list(X.columns):
            series = X[col].dropna()
            if series.empty:
                X.drop(columns=[col], inplace=True)
                continue
            if pd.api.types.is_bool_dtype(X[col]):
                X = X.assign(**{col: X[col].fillna(False).astype(int)})
                bool_cols.add(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                unique_vals = set(series.unique())
                if unique_vals <= {0, 1}:
                    X = X.assign(**{col: X[col].fillna(0).astype(int)})
                    bool_cols.add(col)
                else:
                    X.drop(columns=[col], inplace=True)
            else:
                categorical_cols.add(col)

        if X.shape[1] == 0:
            return {}, 0.0

        # Only fill boolean/binary columns with 0; categorical columns must not
        # be filled with 0 — that would create a spurious dummy column "{attr}_0"
        # and produce Guard(attr, "0") in the extracted SOP conditions.
        # Categorical NaN rows produce all-zero one-hot vectors via dummy_na=False,
        # which is the correct encoding for "attribute absent / not yet written".
        X_for_enc = X.copy()
        for col in X_for_enc.columns:
            if col not in categorical_cols:
                X_for_enc[col] = X_for_enc[col].fillna(0)
            else:
                X_for_enc[col] = X_for_enc[col].fillna("none")
        X_enc = pd.get_dummies(X_for_enc, columns=list(categorical_cols), dummy_na=False)
        X_enc = X_enc.rename(columns={c: c for c in X_enc.columns})

        if X_enc.empty or X_enc.shape[1] == 0:
            return {}, 0.0

        feature_names = list(X_enc.columns)
        sanitized_cats = {c for c in categorical_cols}
        sanitized_bools = {c for c in bool_cols}

        clf = DecisionTreeClassifier(
            max_depth=self.config.dt_max_depth,
            min_samples_leaf=max(1, len(y) // 20),
            class_weight="balanced",
            random_state=42,
        )
        try:
            clf.fit(X_enc, y)
        except Exception as exc:
            logger.warning("DT training failed: %s", exc)
            return {}, 0.0

        accuracy = float((clf.predict(X_enc) == y).mean())
        guards = self._build_sop_guards(clf, feature_names, sanitized_cats, sanitized_bools)
        return guards, accuracy

    def _build_sop_guards(
        self,
        clf: DecisionTreeClassifier,
        feature_names: List[str],
        categorical_cols: Set[str],
        bool_cols: Set[str],
    ) -> Dict[str, List[List[Guard]]]:
        """Walk all root-to-leaf paths in the DT, prune low-quality leaves, and
        build SOP guards.

        Each root-to-leaf path yields one inner list (AND of Guard conditions).
        Multiple paths predicting the same class are OR'd together (outer list).

        Args:
            clf: Fitted DecisionTreeClassifier.
            feature_names: Encoded (sanitized) feature names matching clf input.
            categorical_cols: Sanitized base names of one-hot encoded categoricals.
            bool_cols: Sanitized names of boolean features.

        Returns:
            Dict mapping predicted activity name to List[List[Guard]].
        """
        tree_ = clf.tree_
        node_feature = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else ""
            for i in tree_.feature
        ]
        paths: List[Tuple[str, List[Guard], int, float]] = []

        def recurse(node: int, conditions: List[Guard]) -> None:
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                fname = node_feature[node]
                threshold = tree_.threshold[node]
                left = self._format_condition(fname, "<=", threshold, categorical_cols, bool_cols)
                right = self._format_condition(fname, ">", threshold, categorical_cols, bool_cols)
                recurse(tree_.children_left[node],
                        conditions + ([left] if left else []))
                recurse(tree_.children_right[node],
                        conditions + ([right] if right else []))
            else:
                n_leaf = int(tree_.n_node_samples[node])
                values = tree_.value[node][0]
                total_weight = float(np.sum(values))
                purity = float(np.max(values) / total_weight) if total_weight > 0 else 0.0
                activity = str(clf.classes_[int(np.argmax(values))])
                if conditions:
                    paths.append((activity, list(conditions), n_leaf, purity))

        recurse(0, [])

        min_n = self.config.dt_prune_min_leaf_samples
        min_p = self.config.dt_prune_min_purity

        for act, conds, n, p in paths:
            kept = n >= min_n and p >= min_p
            logger.debug(
                "[DT leaf] branch='%s' n=%d purity=%.3f → %s "
                "(thresholds: min_n=%d min_purity=%.2f)",
                act, n, p, "KEPT" if kept else "PRUNED", min_n, min_p,
            )

        activities_before: Set[str] = {act for act, _, _, _ in paths}
        surviving: List[Tuple[str, List[Guard], int, float]] = [
            (act, conds, n, p)
            for act, conds, n, p in paths
            if n >= min_n and p >= min_p
        ]
        activities_after: Set[str] = {act for act, _, _, _ in surviving}

        for orphaned in activities_before - activities_after:
            candidates = [(conds, n, p) for act, conds, n, p in paths if act == orphaned]

            if self.config.dt_prune_orphan_mode == "fallback":
                best_n = max(c[1] for c in candidates)
                best_p = max(c[2] for c in candidates)
                surviving.append((orphaned, [], best_n, best_p))
                logger.debug(
                    "[DT orphan] branch '%s': all %d leaf/leaves below threshold "
                    "(best purity=%.3f, best n=%d; min_purity=%.2f, min_n=%d) — "
                    "no guard produced, branch will use probabilistic fallback.",
                    orphaned, len(candidates), best_p, best_n, min_p, min_n,
                )

            elif self.config.dt_prune_orphan_mode == "keep_best":
                best_conds, best_n, best_p = max(candidates, key=lambda x: (x[2], x[1]))
                surviving.append((orphaned, best_conds, best_n, best_p))
                logger.debug(
                    "[DT orphan] branch '%s': all %d leaf/leaves below threshold — "
                    "kept best (purity=%.3f, n=%d). Guard may be imprecise.",
                    orphaned, len(candidates), best_p, best_n,
                )


        sop: Dict[str, List[List[Guard]]] = defaultdict(list)
        for activity, cond_list, _, _ in surviving:
            sop[activity].append(cond_list)
        return dict(sop)

    def _format_condition(
        self,
        feature_name: str,
        op: str,
        threshold: float,
        categorical_cols: Set[str],
        bool_cols: Set[str],
    ) -> Optional[Guard]:
        """Convert a DT split (feature op threshold) to a Guard object.

        Encoding conventions:
        - One-hot bool (_true/_false suffix): > 0.5 → Guard(attr, None, False),
          <= 0.5 → Guard(attr, None, True). _false columns invert the polarity.
        - Boolean int column: > 0.5 → Guard(attr, None, False),
          <= 0.5 → Guard(attr, None, True).
        - One-hot categorical: > 0.5 → Guard(base, value, False);
          the <= 0.5 side returns None.
        - Raw numeric columns: return None.

        Args:
            feature_name: Sanitized encoded column name.
            op: "<=" or ">".
            threshold: Split threshold from the DT.
            categorical_cols: Sanitized base names of categorical features.
            bool_cols: Sanitized names of boolean features.

        Returns:
            Guard, or None if the split produces no meaningful condition.
        """
        lower = feature_name.lower()

        if lower.endswith("_true"):
            base = feature_name[:-5]
            return Guard(attribute=base, value=None, negated=(op == "<="))
        if lower.endswith("_false"):
            base = feature_name[:-6]
            return Guard(attribute=base, value=None, negated=(op == ">"))

        base = self._get_base_feature(feature_name, categorical_cols)

        if base in categorical_cols:
            if op == ">":
                value = feature_name[len(base) + 1:]
                return Guard(attribute=base, value=value, negated=False)
            return None

        if base in bool_cols:
            return Guard(attribute=base, value=None, negated=(op == "<="))

        return None

    def _get_base_feature(self, col_name: str, categorical_cols: Set[str]) -> str:
        """Resolve the base attribute name from a potentially one-hot encoded column.

        Args:
            col_name: Sanitized (encoded) column name.
            categorical_cols: Set of known categorical base names.

        Returns:
            Base attribute name, or col_name itself if no match is found.
        """
        parts = col_name.split("_")
        for i in range(1, len(parts)):
            candidate = "_".join(parts[:-i])
            if candidate in categorical_cols:
                return candidate
        return col_name

    def _activity_name(self, transition: Transition) -> str:
        """Resolve the sanitized activity name for a transition."""
        if transition.label is not None:
            return transition.label
        return self.silent_transitions.get(transition, f"tau_unknown_{id(transition)}")

    # =========================================================================
    # Public API — effect mining
    # =========================================================================

    def screen_effects(
        self,
        attribute_effects: Dict[str, AttributeEffect],
    ) -> Dict[str, TransitionScreening]:
        """Screen each transition's effect attributes and classify values.

        For each (transition, attribute) pair, determines whether to train
        a DT (appearance and/or value) or skip.  Individual values are tagged
        as active/pruned/certain based on their probability.

        Args:
            attribute_effects: Activity name → AttributeEffect from
                ProbabilityEstimator.

        Returns:
            Dict mapping activity name to TransitionScreening.
            Attributes below effect_never_threshold are excluded.
        """
        result: Dict[str, TransitionScreening] = {}
        never_thr = self.config.effect_never_threshold
        always_thr = self.config.effect_always_threshold
        v_prune = self.config.effect_value_prune_threshold
        v_certain = self.config.effect_value_certain_threshold

        for act, ae in attribute_effects.items():
            attr_screenings: Dict[str, EffectAttrScreening] = {}

            for attr, p in ae.presence_probabilities.items():
                # ignored effect: enough samples to trust the estimate, and it's
                # below the never-threshold. Too few samples falls through to the
                # dt_min_samples/dt_min_prob_to_use fallback check below instead,
                # so a low-data attribute isn't discarded outright.
                if p < never_thr and ae.total_firings > self.config.probability_min_samples:
                    attr_screenings[attr] = EffectAttrScreening(
                        presence_probability=p,
                        appearance_action="never",
                        appearance_samples=0,
                        value_action="never",
                        value_samples=0,
                        values={},
                    )
                    continue
                # sure effect
                if p > always_thr and ae.total_firings > self.config.probability_min_samples:
                    app_action = "deterministic"
                    app_samples = 0
                elif ae.total_firings < self.config.dt_min_samples or p < self.config.dt_min_prob_to_use:
                    app_action = "fallback"
                    app_samples = ae.total_firings
                else:
                    app_action = "dt"
                    app_samples = ae.total_firings

                value_probs = ae.value_probabilities.get(attr, {})
                n_val_samples = round(p * ae.total_firings)

                values: Dict[Any, EffectValueScreening] = {}
                for val, vp in value_probs.items():
                    # we need to trust probability in order to prune or confirm as certain
                    if vp < v_prune and ae.total_firings * vp > self.config.probability_min_samples:
                        vs = "pruned"
                    elif vp > v_certain and ae.total_firings * vp > self.config.probability_min_samples:
                        vs = "certain"
                    elif ae.total_firings * vp > self.config.dt_min_samples and vp > self.config.dt_min_prob_to_use:
                        vs = "active"
                    else:
                        vs = "fallback"
                    values[val] = EffectValueScreening(probability=vp, status=vs)

                n_active = sum(1 for v in values.values() if v.status == "active")
                if n_val_samples < self.config.dt_min_samples:
                    val_action = "fallback"
                elif len(value_probs) <= 1 or n_active < 2 or any(v.status == "certain" for v in values.values()):
                    val_action = "deterministic"
                else:
                    val_action = "dt"

                attr_screenings[attr] = EffectAttrScreening(
                    presence_probability=p,
                    appearance_action=app_action,
                    appearance_samples=app_samples,
                    value_action=val_action,
                    value_samples=n_val_samples,
                    values=values,
                )

            if attr_screenings:
                result[act] = TransitionScreening(
                    transition_name=act,
                    total_firings=ae.total_firings,
                    attributes=attr_screenings,
                )

        return result

    def mine_effects(
        self,
        preprocessed_log: PreprocessedLog,
        effect_screening: Dict[str, TransitionScreening],
    ) -> Dict[str, TransitionEffects]:
        """Train decision trees for conditional effects using screening results.

        Only processes attributes whose screening action is "dt". Pruned values
        are excluded from the 2B DT training data.

        Args:
            preprocessed_log: Single-pass preprocessed log.
            effect_screening: Screening results from screen_effects().

        Returns:
            Dict mapping transition name to TransitionEffects.
        """
        result: Dict[str, TransitionEffects] = {}

        for act, t_scr in effect_screening.items():
            matrix_built = False
            X: pd.DataFrame = pd.DataFrame()
            y_presence: Dict[str, pd.Series] = {}
            y_value: Dict[str, pd.Series] = {}

            effects: Dict[str, ConditionalEffect] = {}

            for attr, attr_scr in t_scr.attributes.items():
                p = attr_scr.presence_probability
                possible_values = list(attr_scr.values.keys())

                # --- 2A — appearance ---
                app_guards: Optional[EffectGuards] = None
                if attr_scr.appearance_action == "dt":
                    if not matrix_built:
                        X, y_presence, y_value = self._build_effect_matrix(
                            preprocessed_log, act
                        )
                        matrix_built = True
                    if attr not in y_presence or X.empty:
                        app_status = "fallback"
                        attr_scr.appearance_action = "fallback"
                    else:
                        app_guards, app_status = self._mine_appearance(
                            X, y_presence[attr]
                        )
                        if app_status != "dt":
                            attr_scr.appearance_action = app_status
                else:
                    app_status = attr_scr.appearance_action

                # --- 2B — value ---
                val_guards: Optional[EffectGuards] = None
                if attr_scr.value_action == "dt":
                    if not matrix_built:
                        X, y_presence, y_value = self._build_effect_matrix(
                            preprocessed_log, act
                        )
                        matrix_built = True
                    if attr not in y_value or X.empty:
                        val_status = "fallback"
                        attr_scr.value_action = "fallback"
                    else:
                        active_vals = {
                            str(v) for v, vs in attr_scr.values.items()
                            if vs.status == "active"
                        }
                        val_guards, val_status = self._mine_value(
                            X, y_value[attr],
                            active_vals if active_vals else None,
                        )
                        if val_status != "dt":
                            attr_scr.value_action = val_status
                else:
                    val_status = attr_scr.value_action

                effects[attr] = ConditionalEffect(
                    attribute=attr,
                    presence_probability=round(p, 4),
                    possible_values=possible_values,
                    appearance=app_guards,
                    appearance_status=app_status,
                    value=val_guards,
                    value_status=val_status,
                )

            if effects:
                result[act] = TransitionEffects(
                    transition_name=act,
                    total_firings=t_scr.total_firings,
                    effects=effects,
                )

        return result

    # -------------------------------------------------------------------------
    # Private: effect feature matrix
    # -------------------------------------------------------------------------

    def _build_effect_matrix(
        self,
        preprocessed_log: PreprocessedLog,
        activity_name: str,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
        """Build training data for all attribute effect analyses of one transition.

        Each TransitionFiringData in transition_firings provides one row:
        pre_state becomes the feature vector, changed_attrs determines
        y_presence and y_value.

        Args:
            preprocessed_log: Single-pass preprocessed log.
            activity_name: Sanitized activity name to look up.

        Returns:
            Tuple (X, y_presence, y_value):
            - X: feature DataFrame (empty if no firings found).
            - y_presence: dict attr → boolean Series.
            - y_value: dict attr → value Series (pd.NA where not changed).
        """
        firings = preprocessed_log.transition_firings.get(activity_name, [])
        if not firings:
            return pd.DataFrame(), {}, {}

        rows = [fd.pre_state for fd in firings]

        all_attrs: Set[str] = set()
        for fd in firings:
            all_attrs.update(fd.changed_attrs.keys())

        y_presence: Dict[str, pd.Series] = {}
        y_value: Dict[str, pd.Series] = {}

        for attr in all_attrs:
            pres_list: List[bool] = []
            val_list: List[Any] = []
            for fd in firings:
                if attr in fd.changed_attrs:
                    pres_list.append(True)
                    val_list.append(fd.changed_attrs[attr])
                else:
                    pres_list.append(False)
                    val_list.append(pd.NA)
            y_presence[attr] = pd.Series(pres_list, dtype=bool)
            y_value[attr] = pd.Series(val_list)

        return pd.DataFrame(rows), y_presence, y_value

    # -------------------------------------------------------------------------
    # Private: effect DT training
    # -------------------------------------------------------------------------

    def _mine_appearance(
        self,
        X: pd.DataFrame,
        y_presence: pd.Series,
    ) -> Tuple[Optional[EffectGuards], str]:
        """Train a 2A (appearance) DT and extract SOP guards.

        Target is binary: "appears" / "not_appears".

        Args:
            X: Feature DataFrame (all firings of the transition).
            y_presence: Boolean Series aligned with X.

        Returns:
            Tuple (guards, status): EffectGuards or None, and status string.
        """
        y = y_presence.map({True: "appears", False: "not_appears"})
        if y.nunique() < 2:
            return None, "fallback"

        raw_guards, accuracy = self._train_and_extract(X, y)
        # no fallback for no guards, remove empty guards
        raw_guards = {appearance: sop for appearance, sop in raw_guards.items() if sop}

        if accuracy < self.config.dt_min_accuracy:
            logger.info(
                "Appearance DT discarded: accuracy %.2f < threshold %.2f.",
                accuracy, self.config.dt_min_accuracy,
            )
            return None, "fallback"

        if not raw_guards or "appears" not in raw_guards:
            logger.info("Appearance DT discarded: 'appears' class pruned away.")
            return None, "fallback"

        return (
            EffectGuards(
                subtype="appearance",
                guards=raw_guards,
                total_samples=len(y),
                dt_accuracy=round(accuracy, 4),
            ),
            "dt",
        )

    def _mine_value(
        self,
        X: pd.DataFrame,
        y_value: pd.Series,
        active_values: Optional[Set[str]] = None,
    ) -> Tuple[Optional[EffectGuards], str]:
        """Train a 2B (value) DT and extract SOP guards.

        Only rows where the attribute actually changed (non-NA y_value) are used.
        If active_values is provided, rows whose value (as string) is not in the
        set are excluded from training (pruned values).

        Args:
            X: Feature DataFrame (all firings of the transition).
            y_value: Series aligned with X; pd.NA where unchanged.
            active_values: If provided, only keep these values in training.

        Returns:
            Tuple (guards, status): EffectGuards or None, and status string.
        """
        mask = y_value.notna()
        X_filtered = X.loc[mask].reset_index(drop=True)
        y_filtered = y_value.loc[mask].reset_index(drop=True).astype(str)

        if active_values is not None:
            value_mask = y_filtered.isin(active_values)
            X_filtered = X_filtered.loc[value_mask].reset_index(drop=True)
            y_filtered = y_filtered.loc[value_mask].reset_index(drop=True)

        if y_filtered.nunique() < 2:
            return None, "insufficient"

        raw_guards, accuracy = self._train_and_extract(X_filtered, y_filtered)

        if accuracy < self.config.dt_min_accuracy:
            logger.info(
                "Value DT discarded: accuracy %.2f < threshold %.2f.",
                accuracy, self.config.dt_min_accuracy,
            )
            return None, "fallback"

        if not raw_guards:
            logger.info("Value DT discarded: all leaves pruned away.")
            return None, "fallback"

        return (
            EffectGuards(
                subtype="value",
                guards=raw_guards,
                total_samples=int(mask.sum()),
                dt_accuracy=round(accuracy, 4),
            ),
            "dt",
        )
