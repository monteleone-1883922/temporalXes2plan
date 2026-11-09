"""A cheap necessary condition for "this tail can be replayed", used to prune the replayer's search.

The replayer (trace_replayer.TraceReplayer._walk) walks a FIXED sequence of transitions and, at every
labeled step, chooses one of the transition's effect groups (a joint assignment of attribute values
whose own guard must hold), subject to the XOR guard of the activity. A trace the model cannot
explain makes it try every combination (BPIC17's O_Create Offer has thousands of groups, so
thousands to the power of the number of offers). Here the same question is asked about each
attribute ALONE, which takes polynomial time:

  viable[i][a] = the values attribute `a` may have before step i such that SOME choice of groups in
                 the steps i.. keeps every guard on `a` satisfiable and ends with the observed final
                 value of `a` (any value, when `a` is not part of the target).

Per attribute, a step maps a value v to the set of values the attribute can have afterwards:
  * every effect group whose guard, PROJECTED on `a`, holds for v is a possible choice; it leaves
    `a` = the value it assigns, or v when it does not assign `a`;
  * the XOR guard of the activity, projected on `a`, must hold for v;
  * a silent step or one without groups leaves v unchanged (after the XOR guard).
The projection of a guard (a sum of products) keeps, in each clause, only the literals on `a`; a
clause with none is always satisfied. The real guard implies its projection, and the real search is
a restriction of this one (it also couples the attributes through the groups and through the
literals on other attributes), so:

  * if the relaxation says a state is not viable, no real continuation from it succeeds: it can be
    pruned and the verdict of the replayer cannot change;
  * if it says viable, nothing is proved and the real search decides.

Guards are evaluated with Guard.holds itself ({} stands for "attribute not written yet"), so the
semantics (a negated literal fails on a missing attribute, value None reads as "true") are the
replayer's own.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple

from models import FiringStep, Guard

_SKIP = object()  # "this group does not assign the attribute"

# A projected guard: None = no constraint on the attribute, else a set of clauses, each a set of
# literals on the attribute (the guard holds when every literal of SOME clause holds).
Projected = Optional[FrozenSet[FrozenSet[Guard]]]


def project(sop: Optional[List[List[Guard]]], attribute: str) -> Projected:
    """The guard `sop` restricted to the literals on `attribute` (see the module docstring)."""
    if not sop:
        return None
    clauses = []
    for clause in sop:
        literals = frozenset(g for g in clause if g.attribute == attribute)
        if not literals:
            return None  # this clause says nothing about the attribute: always satisfiable
        clauses.append(literals)
    return frozenset(clauses)


def projected_holds(projected: Projected, attribute: str, value: Optional[str]) -> bool:
    """Whether a projected guard holds when `attribute` has `value` (None = not written yet)."""
    if projected is None:
        return True
    state: Dict[str, str] = {} if value is None else {attribute: value}
    return any(all(g.holds(state) for g in clause) for clause in projected)


class Viability:
    """The viable sets of one trace and what the replayer asks of them."""

    def __init__(
        self,
        relaxation: "AttributeRelaxation",
        steps: List[FiringStep],
        viable: List[Dict[str, FrozenSet[Optional[str]]]],
        tracked: List[str],
        domains: Dict[str, FrozenSet[Optional[str]]],
    ) -> None:
        self._relaxation = relaxation
        self._steps = steps
        self.viable = viable  # viable[i][a], i = 0..len(steps); viable[len(steps)] is the target
        self.tracked = tracked
        # The values the table was computed over. A value outside it (never mentioned by the model
        # nor seen in the trace) cannot be refuted, so it counts as viable: pruning must stay sound.
        self._domains = domains
        self._unfiltered: Dict[int, bool] = {}

    def _nothing_to_filter(self, index: int) -> bool:
        """True when EVERY value any group of step `index` can assign is viable afterwards: then no
        group can be filtered out as long as the current state itself is viable, and scanning
        the (possibly thousands of) candidates would only cost time."""
        cached = self._unfiltered.get(index)
        if cached is None:
            step, after = self._steps[index], self.viable[index + 1]
            cached = all(
                self._viable(after, a, v)
                for a in self.tracked
                for v in self._relaxation.assigned_values(step, a)
            )
            self._unfiltered[index] = cached
        return cached

    def _viable(self, row: Mapping[str, FrozenSet[Optional[str]]], attribute: str, value: Any) -> bool:
        return value in row[attribute] or value not in self._domains[attribute]

    def state_ok(self, index: int, state: Mapping[str, Any]) -> bool:
        """Is the attribute state before step `index` viable (index = len(steps): the final one)."""
        row = self.viable[index]
        return all(self._viable(row, a, state.get(a)) for a in self.tracked)

    def filter_candidates(self, index: int, state: Mapping[str, Any], candidates: List[Any]) -> List[Any]:
        """The effect groups that, applied at step `index`, leave a viable state.

        A group is kept when every value it assigns is viable after the step and it assigns every
        attribute whose current value is no longer viable (the others keep their value)."""
        after = self.viable[index + 1]
        broken = {a for a in self.tracked if not self._viable(after, a, state.get(a))}
        if not broken and self._nothing_to_filter(index):
            return candidates
        kept = []
        for group in candidates:
            fixed: Set[str] = set()
            ok = True
            for attribute, value in group.assignments:
                if attribute not in after:
                    continue
                if not self._viable(after, attribute, value):
                    ok = False
                    break
                if attribute in broken:
                    fixed.add(attribute)
            if ok and len(fixed) == len(broken):
                kept.append(group)
        return kept

    def first_blocked_step(self, state: Mapping[str, Any], start: int = 0) -> Optional[Tuple[int, str]]:
        """(index of the first step no choice of groups can get past, "xor" or "effect"), or None.

        Forward over the same per-attribute relaxation, from `state` before step `start`; only used
        to say WHERE a tail that cannot be replayed stops, with the same two reasons the real walk
        gives. The index is in the whole firing sequence (not relative to `start`)."""
        reach: Dict[str, Set[Optional[str]]] = {a: {state.get(a)} for a in self.tracked}
        for i in range(start, len(self._steps)):
            step = self._steps[i]
            for attribute in self.tracked:
                follow: Set[Optional[str]] = set()
                for value in reach[attribute]:
                    follow |= self._relaxation.next_values(step, attribute, value)
                if not follow:
                    kind = "xor" if not any(
                        self._relaxation.xor_holds(step, attribute, v) for v in reach[attribute]
                    ) else "effect"
                    return i, kind
                reach[attribute] = follow
        return None


class AttributeRelaxation:
    """Builds Viability objects for the traces of one model (tables are cached across traces).

    groups_of(activity)  effect groups of an activity, with values in the replay's value space;
    xor_guard_for(activity)  the combined XOR guard (a sum of products, [] = none).
    """

    def __init__(
        self,
        transition_names: Iterable[str],
        groups_of: Callable[[str], List[Any]],
        xor_guard_for: Callable[[str], List[List[Guard]]],
        declared_values: Mapping[str, Iterable[Any]],
    ) -> None:
        self._groups_of = groups_of
        self._xor_guard_for = xor_guard_for
        self._transition_names = list(transition_names)
        self._declared_seed = {a: set(v) for a, v in declared_values.items()}
        self._declared: Optional[Dict[str, Set[Optional[str]]]] = None
        self._mentioned: Dict[Tuple[str, bool], Set[str]] = {}
        self._models: Dict[Tuple[str, bool, str], Tuple[Projected, Optional[FrozenSet[Tuple[Any, Projected]]]]] = {}
        self._next: Dict[Tuple[str, bool, str, Optional[str]], FrozenSet[Optional[str]]] = {}

    # ------------------------------------------------------------------ values

    def declared(self, attribute: str) -> Set[Optional[str]]:
        """Every value the MODEL talks about for `attribute` (catalog, groups, guards), and None."""
        if self._declared is None:
            table: Dict[str, Set[Optional[str]]] = {a: set(v) for a, v in self._declared_seed.items()}

            def add_guard(g: Guard) -> None:
                table.setdefault(g.attribute, set()).add(g.value if g.value is not None else "true")

            for name in self._transition_names:
                for group in self._groups_of(name):
                    for attr, value in group.assignments:
                        table.setdefault(attr, set()).add(value)
                    for clause in group.guard:
                        for g in clause:
                            add_guard(g)
                for clause in self._xor_guard_for(name):
                    for g in clause:
                        add_guard(g)
            self._declared = table
        return set(self._declared.get(attribute, set())) | {None}

    # ------------------------------------------------------------- step tables

    def _model(self, step: FiringStep, attribute: str):
        """(XOR guard projected, options) of a step for one attribute; options is None when the
        step does not choose a group (silent, or no groups), else the distinct
        (value assigned or _SKIP, group guard projected) pairs over its groups."""
        key = (step.activity_name, step.is_tau, attribute)
        cached = self._models.get(key)
        if cached is not None:
            return cached
        xor = project(self._xor_guard_for(step.activity_name), attribute)
        groups = [] if step.is_tau else self._groups_of(step.activity_name)
        options: Optional[FrozenSet[Tuple[Any, Projected]]] = None
        if groups:
            seen: Set[Tuple[Any, Projected]] = set()
            for group in groups:
                assigned = _SKIP
                for attr, value in group.assignments:
                    if attr == attribute:
                        assigned = value
                seen.add((assigned, project(group.guard, attribute)))
            options = frozenset(seen)
        self._models[key] = (xor, options)
        return self._models[key]

    def assigned_values(self, step: FiringStep, attribute: str) -> Set[Any]:
        """Every value some group of the step assigns to `attribute` (empty for a step without groups)."""
        options = self._model(step, attribute)[1]
        return {assigned for assigned, _ in (options or ()) if assigned is not _SKIP}

    def xor_holds(self, step: FiringStep, attribute: str, value: Optional[str]) -> bool:
        return projected_holds(self._model(step, attribute)[0], attribute, value)

    def next_values(self, step: FiringStep, attribute: str, value: Optional[str]) -> FrozenSet[Optional[str]]:
        """Values `attribute` can have after `step` when it has `value` before (empty: blocked)."""
        key = (step.activity_name, step.is_tau, attribute, value)
        cached = self._next.get(key)
        if cached is not None:
            return cached
        xor, options = self._model(step, attribute)
        if not projected_holds(xor, attribute, value):
            result: FrozenSet[Optional[str]] = frozenset()
        elif options is None:
            result = frozenset({value})
        else:
            result = frozenset(
                value if assigned is _SKIP else assigned
                for assigned, guard in options
                if projected_holds(guard, attribute, value)
            )
        self._next[key] = result
        return result

    def mentioned(self, step: FiringStep) -> Set[str]:
        """Attributes a step reads or writes (its XOR guard, its groups' assignments and guards)."""
        key = (step.activity_name, step.is_tau)
        cached = self._mentioned.get(key)
        if cached is not None:
            return cached
        attrs = {g.attribute for clause in self._xor_guard_for(step.activity_name) for g in clause}
        if not step.is_tau:
            for group in self._groups_of(step.activity_name):
                attrs.update(a for a, _ in group.assignments)
                attrs.update(g.attribute for clause in group.guard for g in clause)
        self._mentioned[key] = attrs
        return attrs

    # --------------------------------------------------------------- the table

    def build(
        self,
        steps: List[FiringStep],
        target: Optional[Mapping[str, Any]],
        observed_values: Optional[Mapping[str, Iterable[Any]]] = None,
    ) -> Viability:
        """Viable sets for a whole firing sequence.

        target: the final attribute values the replay must reach (None: any final state, only the
            guards matter). observed_values: for each attribute the values seen in the trace (the
            states the replay can start from), so that a value the model never mentions is still
            tracked."""
        target = target or {}
        tracked_set: Set[str] = set(target)
        for step in steps:
            tracked_set |= self.mentioned(step)
        tracked = sorted(tracked_set)
        n = len(steps)
        viable: List[Dict[str, FrozenSet[Optional[str]]]] = [dict() for _ in range(n + 1)]
        domains: Dict[str, FrozenSet[Optional[str]]] = {}
        for attribute in tracked:
            domain = self.declared(attribute)
            domain.update((observed_values or {}).get(attribute, ()))
            if attribute in target:
                domain.add(target[attribute])
                current: FrozenSet[Optional[str]] = frozenset({target[attribute]})
            else:
                current = frozenset(domain)
            domains[attribute] = frozenset(domain)
            viable[n][attribute] = current
            for i in range(n - 1, -1, -1):
                step = steps[i]
                current = frozenset(
                    v for v in domain if not self.next_values(step, attribute, v).isdisjoint(current)
                )
                viable[i][attribute] = current
        return Viability(self, steps, viable, tracked, domains)
