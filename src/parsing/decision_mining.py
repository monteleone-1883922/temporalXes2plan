"""Decision tree mining for XOR-split guards from a PetriNetLog."""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from sklearn.tree import DecisionTreeClassifier, _tree
from pm4py import PetriNet
from pm4py.objects.powl.obj import Transition

import core_utils as utils
from models import AnalysisConfig, Guard, PetriNetLog, XorSplitGuards
from parsing.discretizer import Discretizer

logger = utils.get_logger(__name__)


class DecisionMiner:
    """
    Mines decision logic for XOR splits from a PetriNetLog using decision trees.

    For each XOR-split place, walks every execution to build a feature matrix:
    the accumulated attribute state at the moment each branch fires becomes one
    training row. A DT is then fit to predict the chosen branch from that state.
    Extracted paths are returned in SOP form (outer list = OR, inner = AND).

    Two-stage filtering avoids wasted computation:
    1. count_xor_samples() — O(executions) pre-check; skips splits below
       config.dt_min_samples before any DT work.
    2. After training, splits whose accuracy falls below config.dt_min_accuracy
       are returned with empty guards so the caller can fall back to
       ProbabilityEstimator statistics.
    """

    def __init__(
        self,
        silent_transitions: Dict[Transition, str],
        config: Optional[AnalysisConfig] = None,
        discretizer: Optional[Discretizer] = None,
    ) -> None:
        """
        Initialize DecisionMiner.

        Args:
            silent_transitions: Transition -> tau name mapping, used to label
                branch transitions in the training matrix.
            config: Analysis configuration. Controls dt_min_samples,
                dt_min_accuracy, dt_max_depth, and ignored_attributes.
            discretizer: Optional pre-fitted Discretizer. Numeric attribute
                values are converted to interval labels before building the
                feature matrix, making them usable as categorical DT features.
        """
        self.silent_transitions = silent_transitions
        self.config = config or AnalysisConfig()
        self._discretizer = discretizer

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def count_xor_samples(
        self,
        pn_log: PetriNetLog,
        xor_splits: Dict[PetriNet.Place, List[Transition]],
    ) -> Dict[PetriNet.Place, int]:
        """
        Count traversals through each XOR-split place in the log.

        A traversal is counted when a step fires from one of the place's outgoing
        transitions (i.e. the place appears in step.from_places and the transition
        is one of the XOR branches).

        Args:
            pn_log: Pre-built PetriNetLog.
            xor_splits: XOR-split places mapped to their outgoing transitions.

        Returns:
            Dict mapping place name to traversal count.
        """
        valid_branches: Dict[PetriNet.Place, Set[Transition]] = {
            place: set(transitions) for place, transitions in xor_splits.items()
        }
        counts: Dict[PetriNet.Place, int] = {place.name: 0 for place in xor_splits}

        for execution in pn_log.executions:
            for step in execution.steps:
                for place in step.from_places:
                    if place in valid_branches and step.transition in valid_branches[place]:
                        counts[place.name] += 1

        return counts

    def mine_xor_splits(
        self,
        pn_log: PetriNetLog,
        xor_splits: Dict[PetriNet.Place, List[Transition]],
    ) -> Tuple[
        Dict[str, XorSplitGuards],
        Dict[PetriNet.Place, List[Transition]]
    ] :
        """
        Train a decision tree for each qualifying XOR split and extract SOP guards.

        Splits with fewer than config.dt_min_samples traversals are skipped and
        absent from the result. Splits whose DT accuracy falls below
        config.dt_min_accuracy are included with empty guards so the caller can
        detect them and apply a probability-based fallback.

        Args:
            pn_log: Pre-built PetriNetLog.
            xor_splits: XOR-split places mapped to their outgoing transitions.

        Returns:
            Dict mapping place name to XorSplitGuards.
        """
        sample_counts = self.count_xor_samples(pn_log, xor_splits)
        result: Dict[str, XorSplitGuards] = {}
        skipped_xor_splits: Dict[PetriNet.Place, List[Transition]] = {}

        for place, transitions in xor_splits.items():
            n_samples = sample_counts.get(place.name, 0)
            if n_samples < self.config.dt_min_samples:
                logger.debug(
                    "XOR split '%s': skipped (%d samples < min %d)",
                    place.name, n_samples, self.config.dt_min_samples,
                )
                skipped_xor_splits[place] = transitions
                continue

            X, y = self._build_feature_matrix(pn_log, place, transitions)
            if X.empty or y.nunique() < 2:
                logger.warning(
                    "XOR split '%s': skipped — feature matrix empty or single class.",
                    place.name,
                )
                continue

            guards, accuracy = self._train_and_extract(X, y)

            if accuracy < self.config.dt_min_accuracy:
                logger.info(
                    "XOR split '%s': DT accuracy %.2f < threshold %.2f — "
                    "empty guards returned (use probability fallback).",
                    place.name, accuracy, self.config.dt_min_accuracy,
                )
                guards = {}

            result[place.name] = XorSplitGuards(
                guards=guards,
                total_samples=n_samples,
                dt_accuracy=round(accuracy, 4),
            )

        return result, skipped_xor_splits

    # -------------------------------------------------------------------------
    # Private: feature matrix
    # -------------------------------------------------------------------------

    def _build_feature_matrix(
        self,
        pn_log: PetriNetLog,
        xor_place: PetriNet.Place,
        transitions: List[Transition],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build the training feature matrix for one XOR-split place.

        For each execution, maintains a running attribute state dict that is
        updated after each labeled step. When a step fires from xor_place, the
        state snapshot BEFORE that step's own attributes are applied is emitted
        as one training row, labelled with the chosen branch activity.

        This causal ordering ensures features reflect the decision context
        (what was known before the choice) rather than its effects.

        Missing/None attribute values are omitted from the state. Attributes in
        config.ignored_attributes are skipped. Values are discretized when a
        discretizer is available.

        Args:
            pn_log: Pre-built PetriNetLog.
            xor_place: The XOR-split place to sample.
            transitions: Outgoing branch transitions of xor_place.

        Returns:
            Tuple (X, y): feature DataFrame and label Series, both empty if no
            traversals were found.
        """
        valid_branches: Set[Transition] = set(transitions)
        rows: List[Dict[str, Any]] = []
        labels: List[str] = []

        for execution in pn_log.executions:
            state: Dict[str, Any] = {}
            for step in execution.steps:
                # Emit sample BEFORE updating state with this step's attributes.
                if xor_place in step.from_places and step.transition in valid_branches:
                    rows.append(dict(state))
                    labels.append(self._activity_name(step.transition))

                # Update accumulated state from labeled steps.
                if not step.is_tau:
                    for attr, val in step.attributes.items():
                        if attr in self.config.ignored_attributes or val is None:
                            continue
                        sanitized = utils.sanitize_name(attr)
                        if (self._discretizer is not None
                                and sanitized in self._discretizer.boundaries):
                            val = self._discretizer.transform_value(sanitized, val)
                        if val is not None:
                            state[sanitized] = val

        if not rows:
            return pd.DataFrame(), pd.Series(dtype=str)

        return pd.DataFrame(rows), pd.Series(labels, dtype=str)

    # -------------------------------------------------------------------------
    # Private: DT training and guard extraction
    # -------------------------------------------------------------------------

    def _train_and_extract(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[Dict[str, List[List[Guard]]], float]:
        """
        Encode features, train a DT, compute accuracy, and extract SOP guards.

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
            - guards: Dict[activity, List[List[str]]] in SOP form.
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

        X_enc = pd.get_dummies(X.fillna(0), columns=list(categorical_cols), dummy_na=False)
        X_enc = X_enc.rename(columns={c: utils.sanitize_name(c) for c in X_enc.columns})

        if X_enc.empty or X_enc.shape[1] == 0:
            return {}, 0.0

        feature_names = list(X_enc.columns)
        sanitized_cats = {utils.sanitize_name(c) for c in categorical_cols}
        sanitized_bools = {utils.sanitize_name(c) for c in bool_cols}

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
        """
        Walk all root-to-leaf paths in the DT and build SOP guards.

        Each root-to-leaf path yields one inner list (AND of Guard conditions along
        that path). Multiple paths predicting the same class are OR'd together
        (outer list).

        Splits that produce no Guard (raw numeric, unrepresentable) are omitted.
        Paths that yield no conditions after filtering are excluded.

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
        paths: List[Tuple[str, List[Guard]]] = []

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
                activity = clf.classes_[int(np.argmax(tree_.value[node]))]
                if conditions:
                    paths.append((str(activity), list(conditions)))

        recurse(0, [])

        sop: Dict[str, List[List[Guard]]] = defaultdict(list)
        for activity, cond_list in paths:
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
        """
        Convert a DT split (feature op threshold) to a Guard object.

        The Guard is format-agnostic: it captures the attribute, optional value,
        and negation flag without committing to any target language. Downstream
        encoders (e.g. PDDL) are responsible for rendering it as a string.

        Encoding conventions:
        - One-hot bool (_true/_false suffix): > 0.5 → Guard(attr, None, False),
          <= 0.5 → Guard(attr, None, True). _false columns invert the polarity.
        - Boolean int column: > 0.5 → Guard(attr, None, False),
          <= 0.5 → Guard(attr, None, True).
        - One-hot categorical (base in categorical_cols): > 0.5 →
          Guard(base, value, False); the <= 0.5 side returns None (omitted to
          keep guards positive — "not (risk high)" is rarely useful as a guard).
        - Raw numeric columns: return None (no discrete condition exists).

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

        # One-hot encoded boolean strings (e.g., admitted_true / admitted_false)
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
            return None  # <= side omitted: negative categorical guards not emitted

        if base in bool_cols:
            return Guard(attribute=base, value=None, negated=(op == "<="))

        # Raw numeric — no discrete condition
        return None

    def _get_base_feature(self, col_name: str, categorical_cols: Set[str]) -> str:
        """
        Resolve the base attribute name from a potentially one-hot encoded column.

        Uses progressively shorter prefix candidates until one matches a known
        categorical base, handling multi-word attribute names with underscores.

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
        """
        Resolve the sanitized activity name for a transition.

        Args:
            transition: A PetriNet transition.

        Returns:
            Sanitized label for labeled transitions, or tau name.
        """
        if transition.label is not None:
            return utils.sanitize_name(transition.label)
        return self.silent_transitions.get(transition, f"tau_unknown_{id(transition)}")
