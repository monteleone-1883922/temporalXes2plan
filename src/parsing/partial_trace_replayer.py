"""Replay a partial XES trace against a serialized Petri net to derive init state."""

import tempfile
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


class PartialTraceReplayer:
    """Replay a single-trace XES file against a serialized Petri net.

    Two phases:
    1. Pre-process each event: filter to catalog attributes, discretize
       numerical values using stored bin_boundaries, normalize booleans.
    2. Token replay: simulate marking from start_place through each activity.

    Returns the current marking (init_places — a list because AND-splits can
    place tokens in multiple places simultaneously) and the last-seen value of
    every catalog attribute encountered (init_effects), plus non-blocking warnings.
    """

    def replay(
        self,
        file_bytes: bytes,
        current_data: Dict[str, Any],
        fmt: str = "xes",
        mapping: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict[str, Any]:
        """Replay a partial trace and return the derived init state.

        Args:
            file_bytes: Raw bytes of the XES or CSV file.
            current_data: Parsed current.json dict (graph, transitions,
                attribute_catalog, metadata).
            fmt: File format — "xes" (default) or "csv".
            mapping: Column mapping required when fmt="csv". Dict with keys
                case_id, activity, timestamp, lifecycle (optional).

        Returns:
            Dict with keys: init_places, init_effects, replayed_activities,
            n_events, warnings.

        Raises:
            PartialTraceError: On any blocking validation or replay error.
        """
        catalog: Dict[str, Any] = current_data.get("attribute_catalog", {})
        transitions: Dict[str, Any] = current_data.get("transitions", {})
        graph: Dict[str, Any] = current_data.get("graph", {})
        metadata: Dict[str, Any] = current_data.get("metadata", {})
        start_place: str = metadata.get("start_place", "")

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

        if not events:
            warnings.append("Trace has 0 events: using start place as initial state.")
            logger.warning("PartialTraceReplayer: empty trace, returning start place.")
            return {
                "init_places": [start_place],
                "init_effects": [],
                "replayed_activities": [],
                "n_events": 0,
                "warnings": warnings,
            }

        # Build graph lookups
        activity_to_node, trans_outputs, silent_inputs, silent_outputs = (
            self._build_graph_lookups(graph)
        )

        # Phase 1: pre-process events
        accumulated_attrs: Dict[str, str] = {}
        for event in events:
            self._process_event_attributes(event, catalog, accumulated_attrs, warnings)

        # Phase 2: token replay — marking is a multiset (list) to support AND-splits
        marking: Set[str] = {start_place}
        self._fire_silent_transitions(marking, silent_inputs, silent_outputs)
        replayed: List[str] = []

        for event in events:
            activity = utils.sanitize_name(str(event.get("concept:name", "")))
            if not activity:
                continue

            if activity not in activity_to_node:
                raise PartialTraceError(
                    f"Activity '{activity}' not found in the Petri net."
                )

            node_id = activity_to_node[activity]
            trans_info = transitions.get(activity, {})
            input_places: List[str] = trans_info.get("input_places", [])

            # Verify every required input token is present in the current marking
            missing = [p for p in input_places if p not in marking]
            if missing:
                raise PartialTraceError(
                    f"Replay stuck at '{activity}': marking {marking} "
                    f"does not contain required places {input_places}."
                )

            # Consume one token per input place, then produce one per output place
            for p in input_places:
                marking.remove(p)
            for p in trans_outputs.get(node_id, []):
                marking.add(p)
            replayed.append(activity)

            # Fire any enabled silent transitions after each visible firing
            self._fire_silent_transitions(marking, silent_inputs, silent_outputs)

        init_effects = [
            {"attribute": attr, "value": value}
            for attr, value in accumulated_attrs.items()
        ]

        return {
            "init_places": list(marking),
            "init_effects": init_effects,
            "replayed_activities": replayed,
            "n_events": len(events),
            "warnings": warnings,
        }

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

    def _fire_silent_transitions(
        self,
        marking: Set[str],
        silent_inputs: Dict[str, List[str]],
        silent_outputs: Dict[str, List[str]],
    ) -> None:
        """Fire all enabled silent transitions to fixpoint (BFS)."""
        changed = True
        while changed:
            changed = False
            for node_id, inputs in silent_inputs.items():
                if inputs and all(p in marking for p in inputs):
                    for p in inputs:
                        marking.remove(p)
                    for p in silent_outputs.get(node_id, []):
                        marking.add(p)
                    changed = True

    def _build_graph_lookups(
        self, graph: Dict[str, Any]
    ) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
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

        # Output places for every non-silent transition (visible + and_split)
        trans_outputs: Dict[str, List[str]] = {}
        # Input→output map for silent transitions (for automatic firing)
        silent_inputs: Dict[str, List[str]] = {}   # node_id → [input place ids]
        silent_outputs: Dict[str, List[str]] = {}  # node_id → [output place ids]

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
        value = utils.sanitize_name(str(raw_value))
        if possible_values and value not in possible_values:
            msg = f"Value '{value}' for '{attr}' ignored: not a known category."
            warnings.append(msg)
            logger.warning("PartialTraceReplayer: %s", msg)
            return None
        return value
