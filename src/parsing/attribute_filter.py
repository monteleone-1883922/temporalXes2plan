"""Filter 1 of the attribute filters: attributes whose statistics mark them as noise.

An event attribute is dropped before the model ever sees it when it is
- N1 constant:      a single distinct value, so it carries no information;
- N2 long text:     categorical with more than ``noise_max_distinct_text``
                    distinct values (identifiers, free text). Every value would
                    become a catalog value and multiply the effect groups;
- N3 counter-like:  carried by almost every event, different from the previous
                    event's value almost every time, and with many distinct
                    values (event ids, sequence counters, position markers).

The thresholds are AnalysisConfig fields (noise_*). They describe the data, not
the quality of a model, so the TPE does not tune them.

Only event attributes are examined: trace (``case:*``) attributes never reach
the model's state. Attributes that are already ignored, or listed in
``keep_attributes``, are never reported.

The functions are pure: they take the log and the per-attribute value counts that
LogProcessor.categorize_attributes already computed, so no extra scan is made
for N1 and N2; N3 needs one pass over the traces, and only for the few
attributes that pass its cheap pre-checks.
"""
from typing import Any, Dict, Iterable, Mapping

import core_utils as utils
from models import AnalysisConfig

logger = utils.get_logger(__name__)


def find_noise_attributes(
    full_log: Iterable[Any],
    attribute_categories: Mapping[str, str],
    value_counts: Mapping[str, Mapping[Any, int]],
    config: AnalysisConfig,
) -> Dict[str, str]:
    """Return {attribute: reason} for every event attribute that is noise.

    Args:
        full_log: The event log the model is built from (iterable of traces,
            each an iterable of event dicts). Used only for rule N3.
        attribute_categories: Attribute -> 'boolean' | 'numerical' |
            'categorical' (LogProcessor.categorize_attributes). Entries with a
            ``case:`` prefix are trace attributes and are skipped.
        value_counts: Attribute -> {value: number of events} for event
            attributes (LogProcessor.attribute_value_counts).
        config: Supplies the thresholds, ignored_attributes and keep_attributes.

    Returns:
        A dict in sorted attribute order, so reports and logs are deterministic.
    """
    protected = set(config.ignored_attributes) | set(config.keep_attributes)
    candidates = sorted(
        attr for attr in attribute_categories
        if attr in value_counts and not attr.startswith("case:") and attr not in protected
    )

    n_events = sum(len(trace) for trace in full_log)
    noise: Dict[str, str] = {}
    counter_candidates = []

    for attr in candidates:
        counts = value_counts[attr]
        n_distinct = len(counts)
        n_carrying = sum(counts.values())

        # N1: nothing to learn from an attribute that never changes value.
        if n_distinct == 1:
            only_value = next(iter(counts))
            noise[attr] = f"N1 constant (only value {only_value!r})"
            continue

        # N2: a categorical attribute with this many values would add one
        # catalog value (and one effect-group branch) for each of them.
        if attribute_categories[attr] == "categorical" and n_distinct > config.noise_max_distinct_text:
            noise[attr] = (
                f"N2 long text ({n_distinct} distinct values, "
                f"limit {config.noise_max_distinct_text})"
            )
            continue

        # N3 pre-check (no scan needed): enough values, and carried by enough events.
        carry = n_carrying / n_events if n_events else 0.0
        if n_distinct >= config.noise_counter_min_distinct and carry >= config.noise_counter_min_carry:
            counter_candidates.append(attr)

    if counter_candidates:
        noise.update(_find_counters(full_log, counter_candidates, config))

    return dict(sorted(noise.items()))


def _find_counters(
    full_log: Iterable[Any], candidates: Iterable[str], config: AnalysisConfig,
) -> Dict[str, str]:
    """Rule N3: how often does the value differ from the previous event's?

    A pair is an event that carries the attribute together with the last
    earlier event of the same trace that carried it. One pass over the log
    handles every candidate at once.
    """
    candidates = list(candidates)
    pairs = {a: 0 for a in candidates}
    changes = {a: 0 for a in candidates}

    for trace in full_log:
        previous: Dict[str, Any] = {}
        for event in trace:
            for attr in candidates:
                if attr not in event:
                    continue
                if attr in previous:
                    pairs[attr] += 1
                    if event[attr] != previous[attr]:
                        changes[attr] += 1
                previous[attr] = event[attr]
            # An event without the attribute does not reset `previous`: the
            # comparison is with the last value seen in the trace.

    result: Dict[str, str] = {}
    for attr in candidates:
        if not pairs[attr]:
            continue
        change_share = changes[attr] / pairs[attr]
        if change_share >= config.noise_counter_min_change:
            result[attr] = (
                f"N3 counter-like (value differs from the previous event's in "
                f"{change_share:.0%} of {pairs[attr]} pairs)"
            )
    return result
