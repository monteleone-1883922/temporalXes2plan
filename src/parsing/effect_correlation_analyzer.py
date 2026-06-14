"""Effect co-occurrence analysis: joint probability that two attributes change together."""
from collections import defaultdict
from itertools import combinations
from typing import Dict

import core_utils as utils
from models import PreprocessedLog

logger = utils.get_logger(__name__)


class EffectCorrelationAnalyzer:
    """Computes joint appearance probabilities for attribute pairs within transitions.

    For each transition T and each pair of attributes (A, B) that appear in
    T's firings, computes:

        P(A and B both modified | T fired) = count(A∩B in same firing) / total_firings(T)

    The result matrix is symmetric: joint[A][B] == joint[B][A].
    Pairs with zero joint occurrences are omitted from the output.
    Transitions with fewer than two distinct attributes across all firings produce
    an empty dict.
    """

    def compute(
        self, preprocessed_log: PreprocessedLog
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute joint probabilities for all transitions in the log.

        Args:
            preprocessed_log: Single-pass preprocessed log with per-transition
                firing data (changed_attrs already populated).

        Returns:
            Mapping trans_name → attr_a → attr_b → joint probability.
            Absent keys mean zero joint occurrences.
        """
        result: Dict[str, Dict[str, Dict[str, float]]] = {}

        for trans_name, firings in preprocessed_log.transition_firings.items():
            total = len(firings)
            if total == 0:
                continue

            pair_counts: Dict[tuple, int] = defaultdict(int)

            for firing in firings:
                attrs = set(firing.changed_attrs.keys())
                if len(attrs) < 2:
                    continue
                for a, b in combinations(sorted(attrs), 2):
                    pair_counts[(a, b)] += 1

            if not pair_counts:
                continue

            joint: Dict[str, Dict[str, float]] = defaultdict(dict)
            for (a, b), count in pair_counts.items():
                prob = count / total
                joint[a][b] = prob
                joint[b][a] = prob

            result[trans_name] = dict(joint)
            logger.debug(
                "Transition '%s': %d attribute pairs analysed over %d firings",
                trans_name, len(pair_counts), total,
            )

        return result
