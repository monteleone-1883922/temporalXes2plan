"""Replay a partial XES trace against a serialized Petri net to derive init state."""

import tempfile
from collections import deque
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from pm4py.objects.log.importer.xes import importer as _xes_importer
from parsing.csv_loader import csv_to_event_log
from parsing.log_validator import validate_partial_trace

import core_utils as utils

logger = utils.get_logger(__name__)

# pm4py event attributes that are not case data
_PM4PY_META_KEYS = frozenset({
    "concept:name",
    "time:timestamp",
    "lifecycle:transition",
    "org:resource",
    "org:group",
    "org:role",
    "@@index",
    "@@classifier",
})


class PartialTraceError(Exception):
    """Raised when the partial trace cannot be replayed (blocking error)."""


@dataclass
class TauPath:
    """A sequence of silent transitions that produces the needed token places."""
    tau_sequence: List[str]      # node_ids of tau transitions in firing order
    marking_after: Set[str]      # marking resulting after firing the sequence
    superfluous: int             # tokens produced beyond what was in the needed set


@dataclass
class TauSplitPoint:
    """A choice point where multiple tau paths were available."""
    event_index: int             # index in events[] of the blocked visible activity
    marking_before: Set[str]     # marking before any tau was fired at this split
    replayed_before: List[str]   # replayed list state at the moment of this split
    tau_fired_before: List[str]  # tau_fired list state at the moment of this split
    alternatives: List[TauPath]  # ordered by superfluous asc, then BFS discovery order
    tried_count: int = 0         # how many alternatives have already been attempted


class PartialTraceReplayer:
    """Replay a trace (full or partial) against a serialized Petri net.

    Tau transitions are fired lazily — only when a visible activity is blocked
    by missing input tokens.  When multiple tau paths exist, the one with fewest
    superfluous tokens is chosen first.  Backtracking restores earlier choice
    points if a selected path leads to a dead end later in the trace.

    For evaluation use, pass the full trace + n_prefix so the chosen tau path
    can be validated against the complete execution.  For the web endpoint,
    pass only the partial trace with n_prefix == len(events) — the algorithm
    treats it as a complete trace and the fast path always fires.
    """

    def replay(
        self,
        file_bytes: bytes,
        current_data: Dict[str, Any],
        fmt: str = "xes",
        mapping: Optional[Dict[str, Optional[str]]] = None,
        tau_max_depth: int = 10,
    ) -> Dict[str, Any]:
        """Replay a trace from raw bytes (web API entry point).

        The entire trace is treated as the prefix (n_prefix = all events).

        Args:
            file_bytes: Raw bytes of the XES or CSV file.
            current_data: Parsed current.json dict.
            fmt: "xes" (default) or "csv".
            mapping: Column mapping required when fmt="csv".
            tau_max_depth: Maximum tau chain depth in BFS search.

        Returns:
            Dict with init_places, init_effects, replayed_activities, n_events,
            warnings, tau_fired, tau_split_count, full_trace_validated.

        Raises:
            PartialTraceError: On any blocking validation or replay error.
        """
        warnings: List[str] = []

        if fmt == "csv":
            if mapping is None:
                raise PartialTraceError(
                    "Column mapping is required for CSV partial traces."
                )
            trace = self._parse_csv(file_bytes, mapping)
        else:
            trace = self._parse_xes(file_bytes, warnings)

        events = list(trace)
        return self.replay_with_full_trace(
            events=events,
            n_prefix=len(events),
            current_data=current_data,
            tau_max_depth=tau_max_depth,
            extra_warnings=warnings,
        )

    def replay_with_full_trace(
        self,
        events: List[Any],
        n_prefix: int,
        current_data: Dict[str, Any],
        tau_max_depth: int = 10,
        extra_warnings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Replay a trace with lazy tau firing and backtracking.

        Args:
            events: All pm4py events of the full trace (or just the prefix when
                called from the web endpoint with n_prefix == len(events)).
            n_prefix: Number of leading events that form the observed prefix.
                Attributes are accumulated only for events[:n_prefix].
                init_places is the marking snapshot taken after the last
                prefix event fires.
            current_data: Parsed current.json dict.
            tau_max_depth: Maximum tau chain depth for BFS path search.
            extra_warnings: Optional list prepended to output warnings.

        Returns:
            Dict with init_places, init_effects, replayed_activities, n_events,
            warnings, tau_fired, tau_split_count, full_trace_validated.

        Raises:
            PartialTraceError: If the trace cannot be replayed (unknown activity
                or backtracking exhausted).
        """
        catalog: Dict[str, Any] = current_data.get("attribute_catalog", {})
        transitions: Dict[str, Any] = current_data.get("transitions", {})
        graph: Dict[str, Any] = current_data.get("graph", {})
        metadata: Dict[str, Any] = current_data.get("metadata", {})
        start_place: str = metadata.get("start_place", "")

        warnings: List[str] = list(extra_warnings or [])

        if not events:
            warnings.append("Trace has 0 events: using start place as initial state.")
            logger.warning("PartialTraceReplayer: empty trace, returning start place.")
            return {
                "init_places": [start_place],
                "init_effects": [],
                "replayed_activities": [],
                "n_events": 0,
                "warnings": warnings,
                "tau_fired": [],
                "tau_split_count": 0,
                "full_trace_validated": True,
            }

        activity_to_node, trans_outputs, silent_inputs, silent_outputs = (
            self._build_graph_lookups(graph)
        )

        # Phase 1 — accumulate attributes from prefix events only
        accumulated_attrs: Dict[str, str] = {}
        for event in events[:n_prefix]:
            self._process_event_attributes(event, catalog, accumulated_attrs, warnings)

        # Phase 2 — token replay with lazy tau and backtracking
        marking: Set[str] = {start_place}
        split_stack: List[TauSplitPoint] = []

        # Snapshot: marking + replayed + tau_fired captured at the prefix cut
        snapshot_marking: Optional[Set[str]] = None
        snapshot_replayed: Optional[List[str]] = None
        snapshot_tau_fired: Optional[List[str]] = None

        replayed: List[str] = []
        tau_fired: List[str] = []

        i = 0
        while i < len(events):
            activity = utils.sanitize_name(str(events[i].get("concept:name", "")))
            if not activity:
                i += 1
                continue

            if activity not in activity_to_node:
                logger.error(
                    "PartialTraceReplayer: activity '%s' not found in Petri net. "
                    "Replayed so far: %s | marking: %s",
                    activity, replayed, sorted(marking),
                )
                raise PartialTraceError(
                    f"Activity '{activity}' not found in the Petri net. "
                    f"Replayed so far ({len(replayed)}): {replayed}"
                )

            node_id = activity_to_node[activity]
            trans_info = transitions.get(activity, {})
            input_places: List[str] = trans_info.get("input_places", [])
            missing = [p for p in input_places if p not in marking]

            if missing:
                paths = self._bfs_tau_paths(
                    marking, set(missing), silent_inputs, silent_outputs, tau_max_depth
                )

                if not paths:
                    logger.error(
                        "PartialTraceReplayer: stuck at '%s', no tau path found. "
                        "Required: %s | Marking: %s | Missing: %s | "
                        "Replayed so far (%d): %s",
                        activity, input_places, sorted(marking), missing,
                        len(replayed), replayed,
                    )
                    marking, i, replayed, tau_fired = self._backtrack(
                        split_stack, activity, replayed
                    )
                    # Invalidate snapshot if we jumped back before the cut
                    if i < n_prefix:
                        snapshot_marking = None
                        snapshot_replayed = None
                        snapshot_tau_fired = None
                    continue

                paths.sort(key=lambda p: p.superfluous)

                # Record a split point only when multiple paths exist and none
                # is uniquely best (i.e. the best still produces superfluous tokens).
                if len(paths) > 1:
                    split_stack.append(TauSplitPoint(
                        event_index=i,
                        marking_before=copy(marking),
                        replayed_before=list(replayed),
                        tau_fired_before=list(tau_fired),
                        alternatives=paths,
                        tried_count=0,
                    ))

                best = paths[0]
                tau_fired.extend(best.tau_sequence)
                marking = copy(best.marking_after)
                # Do not advance i — retry the same activity with the updated marking
                continue

            # All required tokens present — fire the visible activity
            for p in input_places:
                marking.remove(p)
            for p in trans_outputs.get(node_id, []):
                marking.add(p)
            replayed.append(activity)
            i += 1

            # Capture snapshot immediately after the last prefix event fires
            if i == n_prefix:
                snapshot_marking = copy(marking)
                snapshot_replayed = list(replayed)
                snapshot_tau_fired = list(tau_fired)

                # Fast path: no tau splits up to the cut → validation not needed
                if not split_stack:
                    break

        full_trace_validated = (i >= len(events))

        # If the loop ran to completion without hitting the fast path,
        # snapshot may already be set; if n_prefix >= len(events), set it now.
        if snapshot_marking is None:
            snapshot_marking = copy(marking)
            snapshot_replayed = list(replayed)
            snapshot_tau_fired = list(tau_fired)

        init_effects = [
            {"attribute": attr, "value": value}
            for attr, value in accumulated_attrs.items()
        ]

        return {
            "init_places": list(snapshot_marking),
            "init_effects": init_effects,
            "replayed_activities": snapshot_replayed,
            "n_events": n_prefix,
            "warnings": warnings,
            "tau_fired": snapshot_tau_fired,
            "tau_split_count": len(split_stack),
            "full_trace_validated": full_trace_validated,
        }

    # ------------------------------------------------------------------
    # Tau search
    # ------------------------------------------------------------------

    def _bfs_tau_paths(
        self,
        marking: Set[str],
        needed: Set[str],
        silent_inputs: Dict[str, List[str]],
        silent_outputs: Dict[str, List[str]],
        max_depth: int,
    ) -> List[TauPath]:
        """BFS over tau-reachable markings to find chains that produce all needed places.

        Args:
            marking: Current marking before any tau is fired.
            needed: Set of place IDs required by the blocked activity.
            silent_inputs: tau node_id → list of input place IDs.
            silent_outputs: tau node_id → list of output place IDs.
            max_depth: Maximum number of tau transitions in a chain.

        Returns:
            List of TauPath whose marking_after contains all needed places.
            Empty if none found within max_depth.
        """
        m_orig = frozenset(marking)
        results: List[TauPath] = []
        # queue entries: (current_marking_as_set, tau_sequence_so_far)
        queue: deque = deque([(copy(marking), [])])
        visited: Set[frozenset] = {m_orig}

        while queue:
            m, seq = queue.popleft()

            for tau_id, inputs in silent_inputs.items():
                if not inputs:
                    continue
                if tau_id in seq:
                    continue
                if not all(p in m for p in inputs):
                    continue
                if len(seq) >= max_depth:
                    continue

                m2 = copy(m)
                for p in inputs:
                    m2.remove(p)
                for p in silent_outputs.get(tau_id, []):
                    m2.add(p)

                new_seq = seq + [tau_id]

                if needed.issubset(m2):
                    superfluous = len((m2 - set(m_orig)) - needed)
                    results.append(TauPath(
                        tau_sequence=new_seq,
                        marking_after=m2,
                        superfluous=superfluous,
                    ))
                    # Don't expand further from a goal state
                    continue

                fs = frozenset(m2)
                if fs not in visited:
                    visited.add(fs)
                    queue.append((m2, new_seq))

        return results

    # ------------------------------------------------------------------
    # Backtracking
    # ------------------------------------------------------------------

    def _backtrack(
        self,
        split_stack: List[TauSplitPoint],
        blocked_activity: str,
        replayed: List[str],
    ) -> Tuple[Set[str], int, List[str], List[str]]:
        """Restore the most recent tau split point and advance to its next alternative.

        Exhausted split points are popped from the stack before trying the next.

        Returns:
            Tuple (restored_marking, event_index, restored_replayed, restored_tau_fired).

        Raises:
            PartialTraceError: When the entire split stack is exhausted.
        """
        while split_stack:
            split = split_stack[-1]
            split.tried_count += 1

            if split.tried_count < len(split.alternatives):
                alt = split.alternatives[split.tried_count]
                logger.debug(
                    "PartialTraceReplayer: backtracking to split at event %d, "
                    "trying alternative %d/%d (superfluous=%d)",
                    split.event_index,
                    split.tried_count,
                    len(split.alternatives) - 1,
                    alt.superfluous,
                )
                restored_tau = list(split.tau_fired_before) + list(alt.tau_sequence)
                return (
                    copy(alt.marking_after),
                    split.event_index,
                    list(split.replayed_before),
                    restored_tau,
                )

            # All alternatives for this split exhausted — pop and try earlier split
            split_stack.pop()

        raise PartialTraceError(
            f"Replay stuck at '{blocked_activity}': backtracking exhausted. "
            f"Replayed so far: {replayed}"
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_csv(self, csv_bytes: bytes, mapping: Dict[str, Optional[str]]):
        """Parse CSV bytes and return the single trace."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as tmp:
                tmp.write(csv_bytes)
                tmp.flush()
                log = csv_to_event_log(tmp.name, mapping)
        except PartialTraceError:
            raise
        except Exception as exc:
            raise PartialTraceError(f"Invalid CSV file: {exc}") from exc

        if len(log) == 0:
            raise PartialTraceError("CSV file contains no traces.")
        if len(log) > 1:
            raise PartialTraceError(
                f"Expected a single-trace CSV file, found {len(log)} traces."
            )
        trace = log[0]
        errors = validate_partial_trace(trace)
        if errors:
            raise PartialTraceError("; ".join(errors))
        return trace

    def _parse_xes(self, xes_bytes: bytes, warnings: List[str]):
        """Parse XES bytes and return the single trace."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".xes", delete=True) as tmp:
                tmp.write(xes_bytes)
                tmp.flush()
                log = _xes_importer.apply(tmp.name)
        except PartialTraceError:
            raise
        except Exception as exc:
            raise PartialTraceError(f"Invalid XES file: {exc}") from exc

        if len(log) == 0:
            raise PartialTraceError("XES file contains no traces.")
        if len(log) > 1:
            raise PartialTraceError(
                f"Expected a single-trace XES file, found {len(log)} traces."
            )
        trace = log[0]
        errors = validate_partial_trace(trace)
        if errors:
            raise PartialTraceError("; ".join(errors))
        return trace

    # ------------------------------------------------------------------
    # Graph lookups
    # ------------------------------------------------------------------

    def _build_graph_lookups(
        self, graph: Dict[str, Any]
    ) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """Build activity→node_id, node_id→output_places, and silent transition maps."""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        visible_types = {"transition", "and_split"}
        node_by_id = {n["id"]: n for n in nodes}

        activity_to_node: Dict[str, str] = {}
        for node in nodes:
            if node.get("type") in visible_types:
                label = node.get("label", "")
                if label:
                    activity_to_node[utils.sanitize_name(label)] = node["id"]

        trans_outputs: Dict[str, List[str]] = {}
        silent_inputs: Dict[str, List[str]] = {}
        silent_outputs: Dict[str, List[str]] = {}

        for node in nodes:
            if node.get("type") == "silent":
                silent_inputs[node["id"]] = []
                silent_outputs[node["id"]] = []

        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            src_node = node_by_id.get(src, {})
            tgt_node = node_by_id.get(tgt, {})

            if src_node.get("type") in visible_types:
                trans_outputs.setdefault(src, []).append(tgt)
            elif src_node.get("type") == "silent":
                silent_outputs.setdefault(src, []).append(tgt)

            if tgt_node.get("type") == "silent":
                silent_inputs.setdefault(tgt, []).append(src)

        return activity_to_node, trans_outputs, silent_inputs, silent_outputs

    # ------------------------------------------------------------------
    # Full-trace attribute scan
    # ------------------------------------------------------------------

    def scan_final_attributes(
        self,
        events: List[Any],
        current_data: Dict[str, Any],
    ) -> Dict[str, str]:
        """Scan all events in a trace and accumulate final attribute values.

        No Petri net verification is performed. Later events overwrite earlier
        ones for the same attribute, yielding the last observed value for each.

        Args:
            events: All pm4py events of the trace.
            current_data: Parsed current.json dict.

        Returns:
            Dict mapping sanitized attribute name → normalized value.
        """
        catalog = current_data.get("attribute_catalog", {})
        accumulated: Dict[str, str] = {}
        warnings: List[str] = []
        for event in events:
            self._process_event_attributes(event, catalog, accumulated, warnings)
        return accumulated

    # ------------------------------------------------------------------
    # Attribute pre-processing
    # ------------------------------------------------------------------

    def _process_event_attributes(
        self,
        event: Any,
        catalog: Dict[str, Any],
        accumulated: Dict[str, str],
        warnings: List[str],
    ) -> None:
        """Extract and normalize attributes from one event into accumulated."""
        for raw_key, raw_value in event.items():
            if raw_key in _PM4PY_META_KEYS or raw_value is None:
                continue

            attr = utils.sanitize_name(raw_key)
            if attr not in catalog:
                msg = f"Attribute '{raw_key}' ignored: not in attribute catalog."
                warnings.append(msg)
                logger.warning("PartialTraceReplayer: %s", msg)
                continue

            entry = catalog[attr]
            attr_type = entry.get("type", "categorical")
            possible_values: List[str] = entry.get("possible_values", [])
            bin_boundaries: List[float] = entry.get("bin_boundaries", [])

            value = self._normalize_value(
                attr, raw_value, attr_type, possible_values, bin_boundaries, warnings
            )
            if value is not None:
                accumulated[attr] = value

    def _normalize_value(
        self,
        attr: str,
        raw_value: Any,
        attr_type: str,
        possible_values: List[str],
        bin_boundaries: List[float],
        warnings: List[str],
    ) -> Optional[str]:
        """Convert a raw event attribute value to the catalog representation."""
        if attr_type == "boolean":
            return "true" if str(raw_value).lower() in ("true", "1", "yes") else "false"

        if attr_type == "numerical":
            if not bin_boundaries:
                msg = f"Attribute '{attr}' ignored: no bin boundaries available."
                warnings.append(msg)
                logger.warning("PartialTraceReplayer: %s", msg)
                return None
            try:
                label = utils.discretize_value(
                    attr, float(raw_value), {attr: bin_boundaries}
                )
            except (TypeError, ValueError):
                msg = f"Value '{raw_value}' for '{attr}' ignored: cannot convert to float."
                warnings.append(msg)
                logger.warning("PartialTraceReplayer: %s", msg)
                return None
            return label

        # categorical
        value = utils.sanitize_value(attr, str(raw_value))
        if possible_values and value not in possible_values:
            msg = f"Value '{value}' for '{attr}' ignored: not a known category."
            warnings.append(msg)
            logger.warning("PartialTraceReplayer: %s", msg)
            return None
        return value
