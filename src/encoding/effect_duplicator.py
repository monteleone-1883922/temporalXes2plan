"""Action duplication for conditional effects (appearance and value guards)."""
import dataclasses
import math
import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import core_utils as utils
from models import (
    AnalysisConfig, AttributeCatalogEntry, EffectGuards, EffectInfo,
    Guard, ParseResult, TransitionInfo,
)
from encoding.action_registry import ActionRegistry
from encoding.effect_encoder import value_to_pddl_effects
from encoding.guard_encoder import and_clause_to_pddl
from encoding.pddl_model import PDDLAction, PDDLBaseAction, PDDLDurativeAction

logger = utils.get_logger(__name__)


class EffectDuplicator:
    """Duplicates actions to resolve conditional effects statically.

    Processes each non-deterministic EffectInfo on every transition,
    expanding ActionRegistry variants through appearance and value phases.
    After all effects of a transition are processed, deduplicates
    preconditions and merges compatible variants.  Finalises probability
    costs across the entire registry at the end.
    """

    def __init__(
        self, parse_result: ParseResult, config: AnalysisConfig
    ) -> None:
        self._pr = parse_result
        self._cfg = config

    def apply(self, registry: ActionRegistry) -> None:
        for trans_name, info in sorted(self._pr.transitions.items()):
            non_det = self._collect_non_deterministic(info)
            if not non_det:
                continue

            base_name = f"execute_{trans_name}"
            if base_name not in registry:
                logger.warning(
                    "Action %s not found in registry, skipping", base_name
                )
                continue

            for attr_name, effect in sorted(non_det.items()):
                catalog_entry = self._pr.attribute_catalog.get(attr_name)
                if catalog_entry is None:
                    logger.warning(
                        "Attribute %s not in catalog, skipping", attr_name
                    )
                    continue
                self._process_effect(
                    registry, base_name, info, attr_name, effect, catalog_entry
                )

            self._post_process(registry, base_name, info)

        self._finalise_costs(registry)

    # ------------------------------------------------------------------
    # Effect classification
    # ------------------------------------------------------------------

    def _collect_non_deterministic(
        self, info: TransitionInfo
    ) -> Dict[str, EffectInfo]:
        """Return effects not already handled by ActionBuilder."""
        result: Dict[str, EffectInfo] = {}
        for attr_name, effect in info.effects.items():
            if self._is_fully_deterministic(effect):
                continue
            result[attr_name] = effect
        return result

    @staticmethod
    def _is_fully_deterministic(effect: EffectInfo) -> bool:
        if effect.appearance_level != 1 or effect.value_level != 1:
            return False
        if effect.appearance_guards is not None or effect.value_guards is not None:
            return False
        return len(effect.value_probabilities) == 1

    # ------------------------------------------------------------------
    # Per-effect processing
    # ------------------------------------------------------------------

    def _process_effect(
        self,
        registry: ActionRegistry,
        base_name: str,
        info: TransitionInfo,
        attr_name: str,
        effect: EffectInfo,
        catalog_entry: AttributeCatalogEntry,
    ) -> None:
        current_variants = registry.get(base_name)
        with_effect, without_effect = self._appearance_phase(
            current_variants, info, attr_name, effect
        )
        valued = self._value_phase(
            with_effect, attr_name, effect, catalog_entry
        )
        registry.replace(base_name, without_effect + valued)

    # ------------------------------------------------------------------
    # Appearance phase
    # ------------------------------------------------------------------

    def _appearance_phase(
        self,
        variants: List[PDDLBaseAction],
        info: TransitionInfo,
        attr_name: str,
        effect: EffectInfo,
    ) -> Tuple[List[PDDLBaseAction], List[PDDLBaseAction]]:
        """Split variants into those that receive the effect and those that don't.

        Returns:
            (with_effect, without_effect) variant lists.
        """
        if effect.appearance_level == 1 and effect.appearance_guards is None:
            return self._appearance_deterministic(
                variants, info, attr_name
            )
        if effect.appearance_level == 1 and effect.appearance_guards is not None:
            return self._appearance_dt(
                variants, info, attr_name, effect.appearance_guards
            )
        if effect.appearance_level == 2:
            return self._appearance_probabilistic(
                variants, info, attr_name, effect.presence_probability
            )
        return [], list(variants)

    def _appearance_deterministic(
        self,
        variants: List[PDDLBaseAction],
        info: TransitionInfo,
        attr_name: str,
    ) -> Tuple[List[PDDLBaseAction], List[PDDLBaseAction]]:
        with_eff: List[PDDLBaseAction] = []
        without: List[PDDLBaseAction] = []
        for v in variants:
            if self._has_incompatible(v, attr_name, info):
                without.append(v)
            else:
                with_eff.append(v)
        return with_eff, without

    def _appearance_dt(
        self,
        variants: List[PDDLBaseAction],
        info: TransitionInfo,
        attr_name: str,
        guards: EffectGuards,
    ) -> Tuple[List[PDDLBaseAction], List[PDDLBaseAction]]:
        appears_clauses = guards.guards.get("appears", [])
        if not appears_clauses:
            return [], list(variants)

        with_eff: List[PDDLBaseAction] = []
        without: List[PDDLBaseAction] = []

        for v in variants:
            if self._has_incompatible(v, attr_name, info):
                without.append(v)
                continue

            if self._has_related(v, attr_name, info):
                expanded = self._expand_with_guards(v, appears_clauses)
                with_eff.extend(expanded)
            else:
                #fixme: shouldn't this be a copy of v?
                without.append(v)
                expanded = self._expand_with_guards(v, appears_clauses)
                with_eff.extend(expanded)

        return with_eff, without

    def _appearance_probabilistic(
        self,
        variants: List[PDDLBaseAction],
        info: TransitionInfo,
        attr_name: str,
        presence_probability: float,
    ) -> Tuple[List[PDDLBaseAction], List[PDDLBaseAction]]:
        track_cost = self._cfg.effect_appearance_mode != "duplicate_no_cost"
        with_eff: List[PDDLBaseAction] = []
        without: List[PDDLBaseAction] = []

        for v in variants:
            if self._has_incompatible(v, attr_name, info):
                without.append(v)
                continue

            if self._has_related(v, attr_name, info):
                if track_cost:
                    v = dataclasses.replace(
                        v,
                        effect_probability=v.effect_probability * presence_probability,
                    )
                with_eff.append(v)
            else:
                copy = dataclasses.replace(v)
                if track_cost:
                    copy = dataclasses.replace(
                        copy,
                        effect_probability=copy.effect_probability * presence_probability,
                    )
                    v = dataclasses.replace(
                        v,
                        effect_probability=v.effect_probability * (1 - presence_probability),
                    )
                with_eff.append(copy)
                without.append(v)

        return with_eff, without

    # ------------------------------------------------------------------
    # Value phase
    # ------------------------------------------------------------------

    def _value_phase(
        self,
        variants: List[PDDLBaseAction],
        attr_name: str,
        effect: EffectInfo,
        catalog_entry: AttributeCatalogEntry,
    ) -> List[PDDLBaseAction]:
        if not variants:
            return []

        if effect.value_level == 1 and effect.value_guards is None:
            return self._value_deterministic(
                variants, attr_name, effect, catalog_entry
            )
        if effect.value_level == 1 and effect.value_guards is not None:
            return self._value_dt(
                variants, attr_name, effect.value_guards, catalog_entry
            )
        if effect.value_level == 2:
            return self._value_probabilistic(
                variants, attr_name, effect, catalog_entry
            )
        return variants

    def _value_deterministic(
        self,
        variants: List[PDDLBaseAction],
        attr_name: str,
        effect: EffectInfo,
        catalog_entry: AttributeCatalogEntry,
    ) -> List[PDDLBaseAction]:
        values = list(effect.value_probabilities.keys())
        if len(values) != 1:
            logger.warning(
                "Deterministic value for %s has %d values, expected 1",
                attr_name, len(values),
            )
            return variants
        value = values[0]
        pddl_effects = value_to_pddl_effects(
            attr_name, value, catalog_entry, self._pr.negated_attributes
        )
        return [self._add_effects(v, pddl_effects, attr_name) for v in variants]

    def _value_dt(
        self,
        variants: List[PDDLBaseAction],
        attr_name: str,
        guards: EffectGuards,
        catalog_entry: AttributeCatalogEntry,
    ) -> List[PDDLBaseAction]:
        result: List[PDDLBaseAction] = []
        for v in variants:
            existing = _get_preconditions(v)
            for value_str, or_clauses in sorted(guards.guards.items()):
                pddl_effects = value_to_pddl_effects(
                    attr_name, value_str, catalog_entry,
                    self._pr.negated_attributes,
                )
                for clause in or_clauses:
                    conditions = and_clause_to_pddl(clause)
                    if _conditions_conflict(existing, conditions):
                        continue
                    new_v = self._add_conditions(v, conditions)
                    new_v = self._add_effects(new_v, pddl_effects, attr_name)
                    result.append(new_v)
        return result

    def _value_probabilistic(
        self,
        variants: List[PDDLBaseAction],
        attr_name: str,
        effect: EffectInfo,
        catalog_entry: AttributeCatalogEntry,
    ) -> List[PDDLBaseAction]:
        track_cost = self._cfg.effect_value_mode != "duplicate_no_cost"
        result: List[PDDLBaseAction] = []
        for v in variants:
            for value, prob in sorted(
                effect.value_probabilities.items(), key=lambda kv: str(kv[0])
            ):
                pddl_effects = value_to_pddl_effects(
                    attr_name, value, catalog_entry,
                    self._pr.negated_attributes,
                )
                new_v = self._add_effects(v, pddl_effects, attr_name)
                if track_cost:
                    new_v = dataclasses.replace(
                        new_v,
                        effect_probability=new_v.effect_probability * prob,
                    )
                result.append(new_v)
        return result

    # ------------------------------------------------------------------
    # Post-processing: deduplicate + merge
    # ------------------------------------------------------------------

    def _post_process(
        self,
        registry: ActionRegistry,
        base_name: str,
        info: TransitionInfo,
    ) -> None:
        variants = registry.get(base_name)
        variants = self._merge_compatible(variants, info)
        variants = self._rename_variants(variants, base_name)
        registry.replace(base_name, variants)

    def _merge_compatible(
        self,
        variants: List[PDDLBaseAction],
        info: TransitionInfo,
    ) -> List[PDDLBaseAction]:
        groups: Dict[FrozenSet[str], List[PDDLBaseAction]] = {}
        for v in variants:
            key = frozenset(_get_preconditions(v))
            groups.setdefault(key, []).append(v)

        result: List[PDDLBaseAction] = []
        for _key, group in groups.items():
            if len(group) == 1:
                result.append(group[0])
                continue
            merged = self._try_merge_group(group, info)
            result.extend(merged)
        return result

    def _try_merge_group(
        self,
        group: List[PDDLBaseAction],
        info: TransitionInfo,
    ) -> List[PDDLBaseAction]:
        merged: List[PDDLBaseAction] = []
        remaining = list(group)

        while remaining:
            current = remaining.pop(0)
            to_retry: List[PDDLBaseAction] = []

            for other in remaining:
                if self._can_merge(current, other, info):
                    current = self._merge_two(current, other)
                else:
                    to_retry.append(other)

            merged.append(current)
            remaining = to_retry

        return merged

    def _can_merge(
        self,
        a: PDDLBaseAction,
        b: PDDLBaseAction,
        info: TransitionInfo,
    ) -> bool:
        if a.effect_attributes != b.effect_attributes:
            return False

        effects_a = set(_get_effects(a))
        effects_b = set(_get_effects(b))
        for attr in a.effect_attributes:
            if {e for e in effects_a if attr in e} != {e for e in effects_b if attr in e}:
                return False

        all_attrs = a.effect_attributes | b.effect_attributes
        for pair in info.incompatible_effects:
            if set(pair) <= all_attrs:
                return False

        return True

    @staticmethod
    def _merge_two(a: PDDLBaseAction, b: PDDLBaseAction) -> PDDLBaseAction:
        effects_a = _get_effects(a)
        effects_b = _get_effects(b)
        merged_effects = _dedup(effects_a + effects_b)
        merged_attrs = a.effect_attributes | b.effect_attributes
        prob = min(a.effect_probability + b.effect_probability, 1.0)

        if isinstance(a, PDDLDurativeAction) and isinstance(b, PDDLDurativeAction):
            merged_eff_start = _dedup(a.effects_at_start + b.effects_at_start)
            merged_eff_end = _dedup(a.effects_at_end + b.effects_at_end)
            return dataclasses.replace(
                a,
                effects_at_start=merged_eff_start,
                effects_at_end=merged_eff_end,
                effect_attributes=merged_attrs,
                effect_probability=prob,
            )
        if isinstance(a, PDDLAction) and isinstance(b, PDDLAction):
            return dataclasses.replace(
                a,
                effects=merged_effects,
                effect_attributes=merged_attrs,
                effect_probability=prob,
            )
        return a

    @staticmethod
    def _rename_variants(
        variants: List[PDDLBaseAction], base_name: str
    ) -> List[PDDLBaseAction]:
        if len(variants) <= 1:
            return variants
        return [
            dataclasses.replace(v, name=f"{base_name}_v{i}")
            for i, v in enumerate(variants)
        ]

    # ------------------------------------------------------------------
    # Cost finalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _finalise_costs(registry: ActionRegistry) -> None:
        for action in registry.all_actions():
            if action.effect_probability < 1.0:
                log_cost = -math.log(action.effect_probability)
                current = action.additional_cost or 0.0
                action.additional_cost = current + log_cost

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_incompatible(
        action: PDDLBaseAction,
        attr_name: str,
        info: TransitionInfo,
    ) -> bool:
        for pair in info.incompatible_effects:
            pair_set = set(pair)
            if attr_name in pair_set:
                if (pair_set - {attr_name}) & action.effect_attributes:
                    return True
        return False

    @staticmethod
    def _has_related(
        action: PDDLBaseAction,
        attr_name: str,
        info: TransitionInfo,
    ) -> bool:
        for pair in info.related_effects:
            pair_set = set(pair)
            if attr_name in pair_set:
                if (pair_set - {attr_name}) & action.effect_attributes:
                    return True
        return False

    # ------------------------------------------------------------------
    # Action mutation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_with_guards(
        action: PDDLBaseAction, or_clauses: List[List[Guard]]
    ) -> List[PDDLBaseAction]:
        result: List[PDDLBaseAction] = []
        existing = _get_preconditions(action)
        for clause in or_clauses:
            conditions = and_clause_to_pddl(clause)
            if _conditions_conflict(existing, conditions):
                continue
            new_action = EffectDuplicator._add_conditions(action, conditions)
            result.append(new_action)
        return result

    @staticmethod
    def _add_conditions(
        action: PDDLBaseAction, conditions: List[str]
    ) -> PDDLBaseAction:
        if isinstance(action, PDDLDurativeAction):
            return dataclasses.replace(
                action,
                conditions_at_start=action.conditions_at_start | set(conditions),
            )
        if isinstance(action, PDDLAction):
            return dataclasses.replace(
                action, preconditions=action.preconditions | set(conditions),
            )
        raise TypeError(f"Unsupported action type: {type(action)}")

    @staticmethod
    def _add_effects(
        action: PDDLBaseAction, effects: List[str], attr_name: Optional[str] = None
    ) -> PDDLBaseAction:
        new_attrs = (
            action.effect_attributes | {attr_name}
            if attr_name is not None
            else action.effect_attributes
        )
        if isinstance(action, PDDLDurativeAction):
            return dataclasses.replace(
                action,
                effects_at_end=action.effects_at_end + effects,
                effect_attributes=new_attrs,
            )
        if isinstance(action, PDDLAction):
            return dataclasses.replace(
                action,
                effects=action.effects + effects,
                effect_attributes=new_attrs,
            )
        raise TypeError(f"Unsupported action type: {type(action)}")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _dedup(items: List[str]) -> List[str]:
    """Remove duplicates preserving order."""
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _get_preconditions(action: PDDLBaseAction) -> Set[str]:
    # TODO: for durative actions the merge key is the union of all three condition
    # slots, so two actions with identical conditions distributed differently across
    # slots (e.g. X in at_start vs X in over_all) would be considered equivalent
    # and incorrectly merged.  Fix by using a per-slot tuple key instead.
    if isinstance(action, PDDLDurativeAction):
        return (
            action.conditions_at_start
            | action.conditions_over_all
            | action.conditions_at_end
        )
    if isinstance(action, PDDLAction):
        return action.preconditions
    return set()


def _get_effects(action: PDDLBaseAction) -> List[str]:
    if isinstance(action, PDDLDurativeAction):
        return action.effects_at_start + action.effects_at_end
    if isinstance(action, PDDLAction):
        return action.effects
    return []


def _negation_of(condition: str) -> Optional[str]:
    """Return the syntactic negation of a PDDL guard predicate, or None."""
    m = re.match(r'^\((\w+)_is (\S+)\)$', condition)
    if m:
        return f"({m.group(1)}_is_not {m.group(2)})"
    m = re.match(r'^\((\w+)_is_not (\S+)\)$', condition)
    if m:
        return f"({m.group(1)}_is {m.group(2)})"
    m = re.match(r'^\((\w+)_true\)$', condition)
    if m:
        return f"({m.group(1)}_false)"
    m = re.match(r'^\((\w+)_false\)$', condition)
    if m:
        return f"({m.group(1)}_true)"
    return None


def _conditions_conflict(existing: Set[str], new_conditions: List[str]) -> bool:
    """Return True if any new condition contradicts an existing precondition.

    Detects two kinds of contradiction:
    - Direct negation: (X_is Y) vs (X_is_not Y), or (X_true) vs (X_false).
    - Same-attribute different value: (X_is Y) vs (X_is Z) with Z != Y.

    Called once per AND-clause so OR clauses are handled naturally: a
    conflicting clause is simply skipped while compatible ones proceed.
    """
    for cond in new_conditions:
        neg = _negation_of(cond)
        if neg and neg in existing:
            return True
        m = re.match(r'^\((\w+)_is (\S+)\)$', cond)
        if m:
            attr, val = m.group(1), m.group(2)
            attr_pattern = re.compile(rf'^\({re.escape(attr)}_is (\S+)\)$')
            for e in existing:
                em = attr_pattern.match(e)
                if em and em.group(1) != val:
                    return True
    return False


def _extract_effect_attributes(effects: Set[str]) -> Set[str]:
    """Extract attribute names from PDDL effect strings.

    Recognises patterns like (attr_is val), (attr_true), (not (attr_is val)),
    (attr_is_not val), (attr_false).  The 'marked' predicate is excluded.
    """
    attrs: Set[str] = set()
    for eff in effects:
        cleaned = eff.replace("(not ", "").replace("(", "").replace(")", "").strip()
        parts = cleaned.split()
        if not parts:
            continue
        pred = parts[0]
        if pred == "marked":
            continue
        for suffix in ("_is_not", "_is", "_true", "_false"):
            if pred.endswith(suffix):
                attr = pred[: -len(suffix)]
                if attr:
                    attrs.add(attr)
                break
    return attrs
