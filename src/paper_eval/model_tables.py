"""Row-oriented tables written next to the model, so they can be queried as they are.

Phase "model" knows everything about the attributes and the transitions of the
net at the moment it builds it; some of it (the attributes the filters removed,
their types, their discretization) is gone from current.json. This module turns
that knowledge into plain rows, one per attribute and one per transition, that
build_model writes as JSONL (attributes.jsonl, transition_durations.jsonl) and
paper_eval.query reads with DuckDB, with no conversion in between.

Pure functions: no files, no pipeline, so they are tested without running the
TPE search.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Fate of an attribute of the log.
KEPT = "kept"  # in the final catalog, used by the model
REMOVED_UNUSED = "removed_unused"  # written by a transition but read by no guard (filter 2)
REMOVED_NOISE = "removed_noise"  # constant, long text or counter-like (filter 1)
IGNORED_BY_USER = "ignored_by_user"  # in the log's ignore list (LogSpec.ignore_attributes)
REMOVED_CASCADE = "removed_cascade"  # analysed, but no effect or guard of the model uses it
REMOVED_UNKNOWN = "removed_unknown"  # gone from the final catalog for no known reason (a bug)


def value_labels(values: Optional[Iterable[Any]]) -> List[str]:
    """The distinct values of an attribute as sorted text labels.

    possible_values is a set that can hold both the boolean False and the string
    "False"; they are written the same way, so counting set elements would count
    that value twice. Counting the distinct labels does not."""
    return sorted({str(v) for v in (values or [])})


def _row(
    attribute: str, status: str, entry: Any = None, reason: Optional[str] = None,
    kept_by_user: bool = False, attr_type: Optional[str] = None,
    n_distinct_observed: Optional[int] = None,
) -> Dict[str, Any]:
    """One attribute row. entry is an AttributeCatalogEntry, or None when the
    attribute was removed before the catalog existed."""
    kind = entry.attribute_type if entry is not None else attr_type
    labels = value_labels(entry.possible_values) if entry is not None else None
    # Boundaries are the Discretizer's cut points (KDE + Ckmeans): only numerical
    # attributes have them. n intervals = n_bin_boundaries + 1.
    boundaries = None
    if entry is not None and kind == "numerical":
        boundaries = [float(b) for b in (entry.bin_boundaries or [])]
    return {
        "attribute": attribute,
        "type": kind,
        "status": status,
        "reason": reason,
        "kept_by_user": kept_by_user,
        "n_values": len(labels) if labels is not None else None,
        "possible_values": labels,
        "n_bin_boundaries": len(boundaries) if boundaries is not None else None,
        "bin_boundaries": boundaries,
        "n_distinct_observed": n_distinct_observed,
    }


def attribute_rows(
    parse_catalog: Mapping[str, Any],
    final_catalog: Mapping[str, Any],
    unused: Set[str],
    noise: Mapping[str, str],
    noise_details: Mapping[str, Mapping[str, Any]],
    ignored_by_user: Sequence[str],
    kept_by_user: Sequence[str],
    analysed_types: Optional[Mapping[str, str]] = None,
    static_attributes: Optional[Set[str]] = None,
    static_info: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """One row per attribute of the log and its fate, plus the inconsistencies found.

    Args:
        parse_catalog: the catalog before the unused-attribute filter
            (ParseResult.attribute_catalog); has type, values and boundaries of
            every attribute the model analysed.
        final_catalog: the catalog the model uses (PreparedDomainInput.attribute_catalog).
        unused: attributes the unused-attribute filter drops.
        noise: {attribute: reason} removed by the noise filter.
        noise_details: {attribute: {"type", "n_distinct"}} of those, taken just
            before they were dropped.
        ignored_by_user: the log's own ignore list. The default technical
            ignored set (concept:name, org:resource, ...) is not listed.
        kept_by_user: attributes protected from the two automatic filters.
        analysed_types: {attribute: type} of every attribute the model analysed
            (after the noise filter). The ones that are not in parse_catalog never
            appear in an effect or guard (a constant attribute, for instance): they
            get status removed_cascade, so every attribute of the log is accounted for.
        static_attributes: the start-of-trace attributes the model found (set by the first
            event of a case and never changed, in at least N of the training traces, and written
            by no transition).
        static_info: {attribute: StaticAttributeInfo} with the counts behind them, for every
            attribute the preprocessor saw (ignored attributes are never seen).

    Returns:
        (rows, anomalies). An attribute can have only one fate; when two lists
        disagree, something in the pipeline did not work as expected: each case
        is logged as a warning and returned in `anomalies`, and the row keeps the
        status of the first rule that matched (catalog, unused, noise, ignored).
    """
    protected = set(kept_by_user)
    rows: List[Dict[str, Any]] = []
    anomalies: List[str] = []
    seen: Dict[str, str] = {}  # attribute -> status of the row already written

    def warn(message: str) -> None:
        anomalies.append(message)
        logger.warning("attribute table: %s", message)

    # Attributes the model analysed: the final catalog is a subset of parse_catalog.
    for name in sorted(parse_catalog):
        if name in final_catalog:
            status, entry = KEPT, final_catalog[name]
            if name in unused:
                warn(f"'{name}' is in the final catalog although the unused-attribute filter drops it")
        elif name in unused:
            status, entry = REMOVED_UNUSED, parse_catalog[name]
        else:
            status, entry = REMOVED_UNKNOWN, parse_catalog[name]
            warn(f"'{name}' left the catalog but is neither in the final catalog nor unused")
        rows.append(_row(name, status, entry, kept_by_user=name in protected))
        seen[name] = status
    # A final-catalog attribute missing from parse_catalog would be silently lost.
    for name in sorted(set(final_catalog) - set(parse_catalog)):
        warn(f"'{name}' is in the final catalog but not in the catalog before the unused filter")
        rows.append(_row(name, KEPT, final_catalog[name], kept_by_user=name in protected))
        seen[name] = KEPT

    for name, reason in sorted(noise.items()):
        if name in seen:
            warn(f"'{name}' was removed as noise but is also {seen[name]}")
            continue
        details = noise_details.get(name, {})
        rows.append(_row(
            name, REMOVED_NOISE, reason=str(reason), kept_by_user=name in protected,
            attr_type=details.get("type"), n_distinct_observed=details.get("n_distinct"),
        ))
        seen[name] = REMOVED_NOISE

    for name in sorted(ignored_by_user):
        if name in seen:
            warn(f"'{name}' is in the ignore list but is also {seen[name]}")
            continue
        # Never analysed: no type, no values.
        rows.append(_row(name, IGNORED_BY_USER, kept_by_user=name in protected))
        seen[name] = IGNORED_BY_USER

    # Analysed, but dropped on the way to the catalog: type known, nothing else.
    for name, kind in sorted((analysed_types or {}).items()):
        if name in seen:
            continue
        rows.append(_row(
            name, REMOVED_CASCADE, reason="no effect or guard of the model uses it",
            kept_by_user=name in protected, attr_type=kind,
        ))
        seen[name] = REMOVED_CASCADE

    # Start-of-trace attributes are not a fate of their own (one that a guard reads is
    # still `kept`): two columns say whether the model treated the attribute as one and
    # how often, in the training traces, the first event set it and it never changed.
    static_attributes = static_attributes or set()
    static_info = static_info or {}
    for row in rows:
        info = static_info.get(row["attribute"])
        row["start_of_trace"] = row["attribute"] in static_attributes
        row["start_of_trace_share"] = float(info.share) if info is not None else None
    return rows, anomalies


def transition_duration_rows(prepared_transitions: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per transition of the prepared model (PreparedDomainInput.transitions),
    with the duration statistics the PDDL domain is built from.

    Durations are in seconds, before any rescaling. A transition without a
    duration has has_duration False and the duration fields None.
    n_effect_groups is how many action variants its effects expand into.
    """
    rows: List[Dict[str, Any]] = []
    for name in sorted(prepared_transitions):
        transition = prepared_transitions[name]
        d = transition.duration
        rows.append({
            "transition": name,
            "activity_name": transition.activity_name,
            "n_effect_groups": len(transition.effect_groups),
            "cost": transition.cost,
            "has_duration": d is not None,
            "effective_min": d.effective_min if d else None,
            "effective_max": d.effective_max if d else None,
            "mean": d.mean if d else None,
            "std_dev": d.std_dev if d else None,
            "observed_min": d.observed_min if d else None,
            "observed_max": d.observed_max if d else None,
            "count": d.count if d else None,
            "source": d.source if d else None,
        })
    return rows
