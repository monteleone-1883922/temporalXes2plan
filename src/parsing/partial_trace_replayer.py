"""Replay a partial XES trace against a serialized Petri net to derive init state."""

import tempfile
from collections import deque
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from pm4py.objects.log.importer.xes import importer as _xes_importer

from core_utils import sanitize_value
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
    pddl_state_before: Optional[Dict[str, str]] = None  # PDDL attr state snapshot for backtrack restore; None means sim was already stopped


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
        """Replay a trace with lazy tau firing, backtracking, and PDDL state simulation.

        Runs two parallel simulations:
        - Token replay: marking-based, always completes, produces init_places.
        - PDDL attribute simulation: checks preconditions and applies action effects
          exactly as the PDDL planner would, producing init_effects and is_replayable.

        Args:
            events: All pm4py events of the full trace (or just the prefix when
                called from the web endpoint with n_prefix == len(events)).
            n_prefix: Number of leading events that form the observed prefix.
                init_places is the marking snapshot taken after the last prefix event.
                init_effects is derived from the PDDL state at the prefix cut.
            current_data: Parsed current.json dict.
            tau_max_depth: Maximum tau chain depth for BFS path search.
            extra_warnings: Optional list prepended to output warnings.

        Returns:
            Dict with init_places, init_effects, replayed_activities, n_events,
            warnings, tau_fired, tau_split_count, full_trace_validated, is_replayable.

        Raises:
            PartialTraceError: If the trace cannot be token-replayed (unknown activity
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
                "is_replayable": True,
            }

        activity_to_node, trans_outputs, silent_inputs, silent_outputs, tau_id_to_label = (
            self._build_graph_lookups(graph)
        )

        # Pre-pass: build per-event attribute snapshot from ALL events.
        # attrs_at[i] = cumulative log-attribute state BEFORE event[i] fires.
        # Used for best-match variant selection (observed vs PDDL effects).
        acc: Dict[str, str] = {}
        attrs_at: List[Dict[str, str]] = [{}]
        for event in events:
            self._process_event_attributes(event, catalog, acc, warnings)
            attrs_at.append(dict(acc))
        # attrs_at[n_prefix] is the log-attribute state after prefix — fallback for init_effects

        # Build XOR conditions lookup for PDDL precondition checking
        xor_lookup = self._build_xor_conditions_lookup(
            current_data.get("xor_splits", {})
        )

        # Token replay state
        marking: Set[str] = {start_place}
        split_stack: List[TauSplitPoint] = []
        replayed: List[str] = []
        tau_fired: List[str] = []

        # Snapshots captured at the prefix cut
        snapshot_marking: Optional[Set[str]] = None
        snapshot_replayed: Optional[List[str]] = None
        snapshot_tau_fired: Optional[List[str]] = None

        # PDDL attribute simulation state — initialized to the log-observed state at the
        # prefix cut, matching the init_effects given to the planner for the suffix.
        pddl_state: Dict[str, str] = dict(attrs_at[n_prefix])
        is_replayable: bool = True

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
                    marking, i, replayed, tau_fired, is_replayable = self._backtrack(
                        split_stack, activity, replayed, pddl_state
                    )
                    # Invalidate snapshot if we jumped back before the cut
                    if i < n_prefix:
                        snapshot_marking = None
                        snapshot_replayed = None
                        snapshot_tau_fired = None
                    continue

                paths.sort(key=lambda p: p.superfluous)

                # PDDL simulation: check tau preconditions before firing sequence.
                # Only active for suffix events (i >= n_prefix); prefix taus are not checked.
                if i >= n_prefix and is_replayable:
                    for tau_node_id in paths[0].tau_sequence:
                        tau_label = tau_id_to_label.get(tau_node_id, "")
                        if not tau_label:
                            continue
                        tau_input_places = silent_inputs.get(tau_node_id, [])
                        for place in tau_input_places:
                            place_map = xor_lookup.get(place, {})
                            tau_xor_clauses = place_map.get(tau_label, [])
                            if tau_xor_clauses and not self._check_conditions(tau_xor_clauses, pddl_state):
                                logger.debug(
                                    "PartialTraceReplayer: PDDL precondition failed for tau '%s' "
                                    "at place '%s'. pddl_state=%s",
                                    tau_label, place, pddl_state,
                                )
                                is_replayable = False
                                break
                        if not is_replayable:
                            break

                # Record a split point only when multiple paths exist
                if len(paths) > 1:
                    split_stack.append(TauSplitPoint(
                        event_index=i,
                        marking_before=copy(marking),
                        replayed_before=list(replayed),
                        tau_fired_before=list(tau_fired),
                        alternatives=paths,
                        tried_count=0,
                        pddl_state_before=dict(pddl_state) if is_replayable else None,
                    ))

                best = paths[0]
                tau_fired.extend(best.tau_sequence)
                marking = copy(best.marking_after)
                # Do not advance i — retry the same activity with the updated marking
                continue

            # All required tokens present — PDDL simulation for visible activity.
            # Only active for suffix events (i >= n_prefix); prefix activities are not checked.
            if i >= n_prefix and is_replayable:
                activity_preconditions = trans_info.get("preconditions", [])

                xor_or_clauses: List[List[Dict[str, Any]]] = []
                for place in input_places:
                    place_map = xor_lookup.get(place, {})
                    clauses = place_map.get(activity, [])
                    if clauses:
                        xor_or_clauses = clauses
                        break

                effect_groups = trans_info.get("effect_groups", [])
                observed = attrs_at[i + 1] if (i + 1) < len(attrs_at) else attrs_at[-1]

                chosen_variant = self._select_best_variant(
                    effect_groups, activity_preconditions, xor_or_clauses,
                    pddl_state, observed,
                )

                if chosen_variant is None:
                    logger.debug(
                        "PartialTraceReplayer: no executable PDDL variant for '%s'. "
                        "pddl_state=%s xor_clauses=%s",
                        activity, pddl_state, xor_or_clauses,
                    )
                    is_replayable = False
                else:
                    for assignment in chosen_variant.get("assignments", []):
                        pddl_state[assignment["attribute"]] = sanitize_value(assignment["attribute"], assignment["value"])

            # Fire the visible activity (token update)
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

        full_trace_validated = (i >= len(events))

        if snapshot_marking is None:
            snapshot_marking = copy(marking)
            snapshot_replayed = list(replayed)
            snapshot_tau_fired = list(tau_fired)

        # init_effects always reflects the log-observed attribute state at the prefix cut.
        # The PDDL simulation (pddl_state) is used only to compute is_replayable.
        init_effects = [
            {"attribute": attr, "value": value}
            for attr, value in attrs_at[n_prefix].items()
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
            "is_replayable": is_replayable,
        }

    # ------------------------------------------------------------------
    # PDDL simulation helpers
    # ------------------------------------------------------------------

    def _build_xor_conditions_lookup(
        self,
        xor_splits_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, List]]:
        """Return {place_id: {sanitized_activity_name: or_clauses}} from xor_splits.

        or_clauses is the raw SOP list from current.json (outer OR, inner AND).
        Empty list means no constraint — callers skip the check.
        """
        result: Dict[str, Dict[str, List]] = {}
        for place_id, split_data in xor_splits_data.items():
            branches: Dict[str, Any] = split_data.get("branches", {})
            place_map: Dict[str, List] = {}
            for activity_name, branch_data in branches.items():
                sanitized = utils.sanitize_name(activity_name)
                place_map[sanitized] = branch_data.get("conditions", [])
            if place_map:
                result[place_id] = place_map
        return result

    def _check_conditions(
        self,
        or_clauses: List[List[Dict[str, Any]]],
        state: Dict[str, str],
    ) -> bool:
        """Return True if state satisfies at least one AND-clause in or_clauses.

        Empty or_clauses means no constraint → always True.
        "=" predicate: state.get(attr) == value (missing attr → fails).
        "<>" predicate: attr is set AND state.get(attr) != value.
        """
        if not or_clauses:
            return True
        for and_clause in or_clauses:
            satisfied = True
            for cond in and_clause:
                attr = cond.get("attribute", "")
                pred = cond.get("predicate", "=")
                value = cond.get("value", "")
                current = state.get(attr)
                if pred == "=":
                    if current != value:
                        satisfied = False
                        break
                elif pred == "<>":
                    if current is None or current == value:
                        satisfied = False
                        break
            if satisfied:
                return True
        return False

    def _select_best_variant(
        self,
        effect_groups: List[Dict[str, Any]],
        activity_preconditions: List[List[Dict[str, Any]]],
        xor_or_clauses: List[List[Dict[str, Any]]],
        pddl_state: Dict[str, str],
        observed_attrs: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Filter effect_groups by preconditions; return best match to observed_attrs.

        Returns None when activity-level or XOR preconditions fail (no variant can fire).
        Returns a synthetic empty variant dict when effect_groups is empty but
        preconditions pass (action fires with no attribute effects).

        Scoring: number of assignments in the variant that match observed_attrs.
        Tie-break: variant probability descending.
        """
        if not self._check_conditions(activity_preconditions, pddl_state):
            return None
        if not self._check_conditions(xor_or_clauses, pddl_state):
            return None

        if not effect_groups:
            return {"assignments": [], "guard": [], "probability": 1.0}

        best: Optional[Dict[str, Any]] = None
        best_score = -1
        best_prob = -1.0

        for variant in effect_groups:
            if not self._check_conditions(variant.get("guard", []), pddl_state):
                continue
            score = sum(
                1 for a in variant.get("assignments", [])
                if observed_attrs.get(a["attribute"]) == a["value"]
            )
            prob = variant.get("probability", 0.0)
            if score > best_score or (score == best_score and prob > best_prob):
                best = variant
                best_score = score
                best_prob = prob

        return best

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
        pddl_state: Dict[str, str],
    ) -> Tuple[Set[str], int, List[str], List[str], bool]:
        """Restore the most recent tau split point and advance to its next alternative.

        Exhausted split points are popped from the stack before trying the next.
        Restores pddl_state in place from the split point snapshot.

        Returns:
            Tuple (restored_marking, event_index, restored_replayed, restored_tau_fired,
            is_replayable_restored).

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
                if split.pddl_state_before is not None:
                    pddl_state.clear()
                    pddl_state.update(split.pddl_state_before)
                    is_replayable_restored = True
                else:
                    is_replayable_restored = False
                return (
                    copy(alt.marking_after),
                    split.event_index,
                    list(split.replayed_before),
                    restored_tau,
                    is_replayable_restored,
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
    ) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
        """Build activity→node_id, node_id→output_places, silent transition maps, and tau_id→label."""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        visible_types = {"transition", "and_split"}
        node_by_id = {n["id"]: n for n in nodes}

        activity_to_node: Dict[str, str] = {}
        tau_id_to_label: Dict[str, str] = {}
        for node in nodes:
            if node.get("type") in visible_types:
                label = node.get("label", "")
                if label:
                    activity_to_node[utils.sanitize_name(label)] = node["id"]
            elif node.get("type") == "silent":
                label = node.get("label", "")
                if label:
                    tau_id_to_label[node["id"]] = utils.sanitize_name(label)

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

        return activity_to_node, trans_outputs, silent_inputs, silent_outputs, tau_id_to_label

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
                accumulated[attr] = sanitize_value(attr, value)

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
