"""Effect co-occurrence analysis: classifies attribute pairs as related or incompatible."""
from collections import defaultdict
from itertools import combinations
from typing import Dict, FrozenSet, Set, Tuple

import core_utils as utils
from models import AnalysisConfig, PreprocessedLog

logger = utils.get_logger(__name__)

# Result type: trans_name → (related_pairs, incompatible_pairs)
_AnalysisResult = Dict[str, Tuple[Set[FrozenSet[str]], Set[FrozenSet[str]]]]


class EffectCorrelationAnalyzer:
    """Classifies attribute pairs within transitions as related or incompatible.

    For each transition T and each pair of attributes (A, B) observed in T's
    firings, computes the joint probability:

        P(A and B both modified | T fired) = count(A∩B) / total_firings(T)

    The probability is then compared against two thresholds from AnalysisConfig:

    - joint >= related_effect_prob  → pair is *related*   (always appear together)
    - joint <= incompatible_effect_prob → pair is *incompatible* (never together)
    - in between                    → no structural constraint

    Both sets are symmetric: if (A, B) is related, so is (B, A), represented
    as a single frozenset {A, B}.
    """

    def analyze(
        self,
        preprocessed_log: PreprocessedLog,
        config: AnalysisConfig,
    ) -> _AnalysisResult:
        """Classify attribute pairs for all transitions.

        Args:
            preprocessed_log: Single-pass preprocessed log with per-transition
                firing data (changed_attrs already populated).
            config: Analysis configuration carrying the two probability thresholds.

        Returns:
            Mapping trans_name → (related_pairs, incompatible_pairs).
            Transitions with fewer than two attributes across all firings are absent.
        """
        joint_probs = self._compute_joint(preprocessed_log)
        result: _AnalysisResult = {}

        for trans_name, pair_probs in joint_probs.items():
            related: Set[FrozenSet[str]] = set()
            incompatible: Set[FrozenSet[str]] = set()

            for pair, prob in pair_probs.items():
                if prob >= config.related_effect_prob:
                    related.add(pair)
                elif prob <= config.incompatible_effect_prob:
                    incompatible.add(pair)

            if related or incompatible:
                result[trans_name] = (related, incompatible)
                logger.debug(
                    "Transition '%s': %d related pairs, %d incompatible pairs",
                    trans_name, len(related), len(incompatible),
                )

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_joint(
        self, preprocessed_log: PreprocessedLog
    ) -> Dict[str, Dict[FrozenSet[str], float]]:
        """Compute raw joint probabilities for all attribute pairs per transition.

        Returns:
            Mapping trans_name → frozenset({a, b}) → P(a ∩ b | T fired).
        """
        result: Dict[str, Dict[FrozenSet[str], float]] = {}

        for trans_name, firings in preprocessed_log.transition_firings.items():
            total = len(firings)
            if total == 0:
                continue

            pair_counts: Dict[FrozenSet[str], int] = defaultdict(int)

            for firing in firings:
                attrs = set(firing.changed_attrs.keys())
                if len(attrs) < 2:
                    continue
                for a, b in combinations(sorted(attrs), 2):
                    pair_counts[frozenset({a, b})] += 1

            if not pair_counts:
                continue

            result[trans_name] = {
                pair: count / total for pair, count in pair_counts.items()
            }

        return result
