from typing import Dict, List, Optional, Any

import core_utils as utils
from models import EffectInfo, ParseResult, TransitionInfo, AttributeCatalogEntry
from encoding.pddl_model import PDDLAction

logger = utils.get_logger(__name__)


class ActionBuilder:
    """Builds PDDL actions from a ParseResult.

    Produces two kinds of actions:
    - Place-marking actions: propagate a token from a completed transition to its output places.
    - Transition base actions: fire a transition when all input places are marked,
      producing the marked predicate for the transition plus any deterministic effects.
    """

    def __init__(self, parse_result: ParseResult) -> None:
        self._pr = parse_result

    def build_all(self) -> List[PDDLAction]:
        """Build all PDDL actions for the domain."""
        actions: List[PDDLAction] = []
        actions.extend(self._build_place_marking_actions())
        actions.extend(self._build_transition_actions())
        actions.extend(self._build_tau_actions())
        return actions

    # ------------------------------------------------------------------
    # Place-marking actions
    # ------------------------------------------------------------------

    def _build_place_marking_actions(self) -> List[PDDLAction]:
        """One action per (place, predecessor_transition) pair."""
        actions: List[PDDLAction] = []

        for place_name, predecessors in sorted(self._pr.place_predecessors.items()):
            for trans_name in sorted(predecessors):
                actions.append(PDDLAction(
                    name=f"mark_{place_name}_from_{trans_name}",
                    preconditions=[f"(marked {trans_name})"],
                    effects=[f"(marked {place_name})"],
                ))

        return actions

    # ------------------------------------------------------------------
    # Transition base actions (labeled)
    # ------------------------------------------------------------------

    def _build_transition_actions(self) -> List[PDDLAction]:
        """One base action per labeled transition."""
        actions: List[PDDLAction] = []

        for trans_name, info in sorted(self._pr.transitions.items()):
            preconditions = self._transition_preconditions(trans_name)
            effects = self._transition_effects(trans_name, info)
            actions.append(PDDLAction(
                name=f"execute_{trans_name}",
                preconditions=preconditions,
                effects=effects,
            ))

        return actions

    # ------------------------------------------------------------------
    # Tau (silent) transition actions
    # ------------------------------------------------------------------

    def _build_tau_actions(self) -> List[PDDLAction]:
        """One action per silent transition."""
        actions: List[PDDLAction] = []

        for trans_obj, tau_label in sorted(
            self._pr.petri_net_model.silent_transitions.items(),
            key=lambda kv: kv[1],
        ):
            tau_name = utils.sanitize_name(tau_label)
            input_places = self._pr.petri_net_model.trans_inputs.get(trans_obj, set())
            preconditions = [
                f"(marked {utils.sanitize_name(p.name)})"
                for p in sorted(input_places, key=lambda p: p.name)
            ]
            effects = [f"(marked {tau_name})"]

            actions.append(PDDLAction(
                name=f"execute_{tau_name}",
                preconditions=preconditions,
                effects=effects,
            ))

        return actions

    # ------------------------------------------------------------------
    # Preconditions & effects helpers
    # ------------------------------------------------------------------

    def _transition_preconditions(self, trans_name: str) -> List[str]:
        """Build precondition list: all predecessor places must be marked."""
        predecessor_places = self._pr.transition_predecessors.get(trans_name, [])
        return [
            f"(marked {p})"
            for p in sorted(predecessor_places)
        ]

    def _transition_effects(
        self, trans_name: str, info: TransitionInfo
    ) -> List[str]:
        """Build effect list: mark self + deterministic attribute effects."""
        effects = [f"(marked {trans_name})"]

        for attr_name, effect_info in sorted(info.effects.items()):
            det_effects = self._deterministic_effect(attr_name, effect_info)
            if det_effects is not None:
                effects.extend(det_effects)

        return effects

    def _deterministic_effect(
        self, attr_name: str, effect: EffectInfo
    ) -> Optional[List[str]]:
        """Return PDDL effect strings if the effect is deterministic, else None.

        An effect is deterministic when both appearance_level and value_level
        are 1 (screened as certain) AND both appearance_guards and value_guards
        are None (no DT conditions needed).
        """
        if effect.appearance_level != 1 or effect.value_level != 1:
            return None
        if effect.appearance_guards is not None or effect.value_guards is not None:
            return None

        values = list(effect.value_probabilities.keys())
        if len(values) != 1:
            return None

        value = values[0]
        catalog_entry = self._pr.attribute_catalog.get(attr_name)
        if catalog_entry is None:
            logger.warning("Attribute %s not in catalog, skipping effect", attr_name)
            return None

        if catalog_entry.attribute_type == "boolean":
            return self._boolean_effect(attr_name, value)
        else:
            return self._categorical_effect(attr_name, value, catalog_entry)

    def _boolean_effect(self, attr_name: str, value: Any) -> List[str]:
        """PDDL effects for a deterministic boolean attribute change."""
        if value is True or str(value).lower() == "true":
            return [f"({attr_name}_true)", f"(not ({attr_name}_false))"]
        else:
            return [f"({attr_name}_false)", f"(not ({attr_name}_true))"]

    def _categorical_effect(
        self, attr_name: str, value: Any, catalog_entry: AttributeCatalogEntry
    ) -> List[str]:
        """PDDL effects for a deterministic categorical attribute change."""
        val_str = str(value)
        effects = [f"({attr_name}_is {val_str})"]

        for other_val in sorted(str(v) for v in catalog_entry.possible_values):
            if other_val == val_str:
                effects.append(f"(not ({attr_name}_is_not {other_val}))")
            else:
                effects.append(f"({attr_name}_is_not {other_val})")
                effects.append(f"(not ({attr_name}_is {other_val}))")

        return effects
