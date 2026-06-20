"""Mine structural attribute preconditions for labeled transitions.

For each transition T, examines the pre_state of every TransitionFiringData.
If an attribute A is present in pre_state and one of its values V appears with
frequency >= attr_precondition_min_frequency across all firings where A is
present, then Guard(A, V) is added as a structural precondition of T.

These preconditions are always-true AND conditions (not XOR routing guards).
They appear as preconditions on every variant action produced by EffectDuplicator.
"""

from collections import Counter
from typing import Any, Dict, List, Set

import core_utils as utils
from models import AnalysisConfig, Guard, TransitionFiringData

logger = utils.get_logger(__name__)


class PreconditionMiner:
    """Mines structural attribute preconditions from transition pre-states."""

    def mine(
        self,
        transition_firings: Dict[str, List[TransitionFiringData]],
        catalog_attributes: Set[str],
        config: AnalysisConfig,
    ) -> Dict[str, List[Guard]]:
        """Return attribute preconditions for each transition.

        Args:
            transition_firings: Preprocessed log index keyed by activity name.
            catalog_attributes: Attributes that survived the effect pipeline.
                Only these are considered for precondition mining.
            config: Analysis configuration carrying thresholds.

        Returns:
            Dict mapping activity name to list of Guards (AND clause, no OR).
        """
        result: Dict[str, List[Guard]] = {}
        ignored = {utils.sanitize_name(a) for a in config.ignored_attributes}

        for activity, firings in transition_firings.items():
            if len(firings) < config.attr_precondition_min_firings:
                continue

            guards = self._mine_activity(
                firings, catalog_attributes, ignored, config
            )
            if guards:
                result[activity] = guards
                logger.info(
                    "Precondition miner: %s → %d structural precondition(s): %s",
                    activity,
                    len(guards),
                    [(g.attribute, g.value) for g in guards],
                )

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mine_activity(
        self,
        firings: List[TransitionFiringData],
        catalog_attributes: Set[str],
        ignored: Set[str],
        config: AnalysisConfig,
    ) -> List[Guard]:
        """Find dominant-value attributes in the pre_state of these firings."""
        # Collect per-attribute value counters from firings where attr is present.
        attr_counters: Dict[str, Counter] = {}
        attr_present_count: Dict[str, int] = {}

        for firing in firings:
            for attr, value in firing.pre_state.items():
                if attr in ignored:
                    continue
                if attr not in catalog_attributes:
                    continue
                attr_counters.setdefault(attr, Counter())
                attr_present_count[attr] = attr_present_count.get(attr, 0) + 1
                attr_counters[attr][str(value)] += 1

        guards: List[Guard] = []
        for attr, counter in sorted(attr_counters.items()):
            n = attr_present_count[attr]
            if n < config.attr_precondition_min_firings:
                continue
            dominant_value, dominant_count = counter.most_common(1)[0]
            frequency = dominant_count / n
            if frequency >= config.attr_precondition_min_frequency:
                guards.append(Guard(
                    attribute=attr,
                    value=dominant_value if dominant_value not in ("True", "False") else None,
                    negated=(dominant_value == "False"),
                ))

        return guards
