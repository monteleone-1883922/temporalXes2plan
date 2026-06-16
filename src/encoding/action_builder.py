from typing import Any, Dict, List, Optional, Set, Tuple

import core_utils as utils
from models import EffectInfo, ParseResult, TransitionInfo, AttributeCatalogEntry
from encoding.pddl_model import (
    PDDLAction, PDDLBaseAction, PDDLCondition, PDDLDurativeAction, PDDLEffect,
)
from encoding.effect_encoder import value_to_pddl_effects

logger = utils.get_logger(__name__)


class ActionBuilder:
    """Builds PDDL actions from a ParseResult.

    Produces three kinds of actions:
    - Place-marking actions: propagate a token from a completed transition to
      its output places (always instantaneous).
    - Transition base actions: fire a transition when all input places are
      marked.  When use_durative=True and the transition has duration data,
      a PDDLDurativeAction is produced instead of a PDDLAction.
    - Tau (silent) transition actions: same structure as labeled transitions
      but never durative (tau transitions carry no duration in the model).
    """

    def __init__(self, parse_result: ParseResult, use_durative: bool = False) -> None:
        self._pr = parse_result
        self._use_durative = use_durative

    def build_all(self) -> List[PDDLBaseAction]:
        """Build all PDDL actions for the domain."""
        actions: List[PDDLBaseAction] = []
        actions.extend(self._build_place_marking_actions())
        actions.extend(self._build_transition_actions())
        actions.extend(self._build_tau_actions())
        return actions

    # ------------------------------------------------------------------
    # Place-marking actions (always instantaneous)
    # ------------------------------------------------------------------

    def _build_place_marking_actions(self) -> List[PDDLAction]:
        """One action per (place, predecessor_transition) pair."""
        actions: List[PDDLAction] = []

        for place_name, predecessors in sorted(self._pr.place_predecessors.items()):
            for trans_name in sorted(predecessors):
                actions.append(PDDLAction(
                    name=f"mark_{place_name}_from_{trans_name}",
                    preconditions={PDDLCondition.marked(trans_name)},
                    effects=[PDDLEffect.marking(place_name)],
                ))

        return actions

    # ------------------------------------------------------------------
    # Transition base actions (labeled)
    # ------------------------------------------------------------------

    def _build_transition_actions(self) -> List[PDDLBaseAction]:
        """One base action per labeled transition."""
        actions: List[PDDLBaseAction] = []

        for trans_name, info in sorted(self._pr.transitions.items()):
            preconditions = self._transition_preconditions(trans_name)
            effects, effect_attrs = self._transition_effects(trans_name, info)

            if self._use_durative and info.duration is not None:
                actions.append(PDDLDurativeAction(
                    name=f"execute_{trans_name}",
                    duration_min=info.duration.effective_min,
                    duration_max=info.duration.effective_max,
                    conditions_at_start=preconditions,
                    effects_at_end=effects,
                    effect_attributes=effect_attrs,
                ))
            else:
                actions.append(PDDLAction(
                    name=f"execute_{trans_name}",
                    preconditions=preconditions,
                    effects=effects,
                    effect_attributes=effect_attrs,
                ))

        return actions

    # ------------------------------------------------------------------
    # Tau (silent) transition actions (always instantaneous)
    # ------------------------------------------------------------------

    def _build_tau_actions(self) -> List[PDDLAction]:
        """One action per silent transition.  Tau transitions are never durative."""
        actions: List[PDDLAction] = []

        for trans_obj, tau_label in sorted(
            self._pr.petri_net_model.silent_transitions.items(),
            key=lambda kv: kv[1],
        ):
            tau_name = utils.sanitize_name(tau_label)
            input_places = self._pr.petri_net_model.trans_inputs.get(trans_obj, set())
            preconditions = {
                PDDLCondition.marked(utils.sanitize_name(p.name))
                for p in input_places
            }
            effects = [PDDLEffect.marking(tau_name)]

            actions.append(PDDLAction(
                name=f"execute_{tau_name}",
                preconditions=preconditions,
                effects=effects,
            ))

        return actions

    # ------------------------------------------------------------------
    # Preconditions & effects helpers
    # ------------------------------------------------------------------

    def _transition_preconditions(self, trans_name: str) -> Set[PDDLCondition]:
        """Build precondition set: all predecessor places must be marked."""
        predecessor_places = self._pr.transition_predecessors.get(trans_name, [])
        return {PDDLCondition.marked(p) for p in predecessor_places}

    def _transition_effects(
        self, trans_name: str, info: TransitionInfo
    ) -> Tuple[List[PDDLEffect], Set[str]]:
        """Build effect list and attribute set: mark self + deterministic attribute effects."""
        effects: List[PDDLEffect] = [PDDLEffect.marking(trans_name)]
        effect_attrs: Set[str] = set()

        for attr_name, effect_info in sorted(info.effects.items()):
            det_effects = self._deterministic_effect(attr_name, effect_info)
            if det_effects is not None:
                effects.extend(det_effects)
                effect_attrs.add(attr_name)

        return effects, effect_attrs

    def _deterministic_effect(
        self, attr_name: str, effect: EffectInfo
    ) -> Optional[List[PDDLEffect]]:
        """Return PDDLEffect list if the effect is deterministic, else None.

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

        return value_to_pddl_effects(
            attr_name, value, catalog_entry, self._pr.negated_attributes
        )
