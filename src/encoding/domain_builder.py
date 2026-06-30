from typing import Dict, List, Optional, Set

import core_utils as utils
from models import AnalysisConfig, AttributeCatalogEntry, ParseResult, PetriNetModel
from encoding.pddl_model import PDDLDurativeAction, PDDLDomain, PDDLObject, PDDLPredicate, PDDLType
from encoding.action_builder import ActionBuilder
from encoding.action_registry import ActionRegistry
from encoding.effect_duplicator import EffectDuplicator
from encoding.xor_branch_processor import XorBranchProcessor

logger = utils.get_logger(__name__)


class DomainBuilder:
    """Builds a PDDLDomain from a ParseResult."""

    def build_with_registry(
        self,
        parse_result: ParseResult,
        domain_name: str = "process",
        use_durative: bool = False,
        use_costs: bool = False,
        config: Optional[AnalysisConfig] = None,
    ) -> tuple[PDDLDomain, "ActionRegistry"]:
        """Like build(), but also returns the ActionRegistry after EffectDuplicator.

        Args:
            parse_result: Encoder-ready output from the parsing pipeline.
            domain_name: Name for the PDDL domain.
            use_durative: Encode durative actions when True.
            use_costs: Include action cost effects and functions section when True.
            config: Analysis configuration; defaults to AnalysisConfig().

        Returns:
            Tuple of (PDDLDomain, ActionRegistry).
        """
        domain, registry = self._build_internal(
            parse_result, domain_name, use_durative, use_costs, config
        )
        return domain, registry

    def build(
        self,
        parse_result: ParseResult,
        domain_name: str = "process",
        use_durative: bool = False,
        use_costs: bool = False,
        config: Optional[AnalysisConfig] = None,
    ) -> PDDLDomain:
        """Transform a ParseResult into a complete PDDLDomain.

        Args:
            parse_result: Encoder-ready output from the parsing pipeline.
            domain_name: Name for the PDDL domain.
            use_durative: When True, transitions that carry duration data are
                encoded as durative-action blocks instead of instantaneous
                actions.  Adds :durative-actions to requirements automatically.
            use_costs: When True, adds (increase (total-cost) X) to action
                effects and the (:functions (total-cost)) section.
            config: Analysis configuration controlling effect duplication
                modes and cost tracking.  Defaults to AnalysisConfig().

        Returns:
            A fully populated PDDLDomain.
        """
        domain, _ = self._build_internal(
            parse_result, domain_name, use_durative, use_costs, config
        )
        return domain

    def _build_internal(
        self,
        parse_result: ParseResult,
        domain_name: str,
        use_durative: bool,
        use_costs: bool,
        config: Optional[AnalysisConfig],
    ) -> tuple[PDDLDomain, "ActionRegistry"]:
        if config is None:
            config = AnalysisConfig()

        types = self._build_types(parse_result.attribute_catalog)
        constants = self._build_constants(parse_result)
        predicates = self._build_predicates(
            parse_result.attribute_catalog, parse_result.negated_attributes
        )

        action_builder = ActionBuilder(parse_result, use_durative=use_durative)
        base_actions = action_builder.build_all()

        registry = ActionRegistry.from_base_actions(base_actions)
        XorBranchProcessor(parse_result).apply(registry)
        EffectDuplicator(parse_result, config).apply(registry)
        actions = registry.all_actions()

        requirements = [":strips", ":typing"]
        if any(isinstance(a, PDDLDurativeAction) for a in actions):
            requirements.append(":durative-actions")

        has_costs = use_costs
        if has_costs:
            requirements += [":action-costs", ":numeric-fluents"]

        effective_deadline = use_durative and any(
            isinstance(a, PDDLDurativeAction) for a in actions
        )
        domain = PDDLDomain(
            name=domain_name,
            requirements=requirements,
            types=types,
            constants=constants,
            predicates=predicates,
            actions=actions,
            has_costs=has_costs,
            has_deadline=effective_deadline,
        )
        return domain, registry

    def _build_types(
        self, attribute_catalog: Dict[str, AttributeCatalogEntry]
    ) -> List[PDDLType]:
        """Build PDDL type hierarchy.

        Creates petri_element as supertype for place and transition,
        plus one value type per categorical/numerical attribute.
        """
        types = [
            PDDLType("petri_element"),
            PDDLType("place", parent="petri_element"),
            PDDLType("transition", parent="petri_element"),
        ]

        for attr_name, entry in sorted(attribute_catalog.items()):
            if entry.attribute_type in ("categorical", "numerical"):
                types.append(PDDLType(f"{attr_name}_val"))

        return types

    def _build_constants(self, parse_result: ParseResult) -> List[PDDLObject]:
        """Build PDDL constants from Petri net elements and attribute values."""
        constants: List[PDDLObject] = []

        place_names = self._collect_place_names(parse_result.petri_net_model)
        for name in sorted(place_names):
            constants.append(PDDLObject(name, "place"))

        transition_names = self._collect_transition_names(parse_result)
        for name in sorted(transition_names):
            constants.append(PDDLObject(name, "transition"))

        for attr_name, entry in sorted(parse_result.attribute_catalog.items()):
            if entry.attribute_type in ("categorical", "numerical"):
                type_name = f"{attr_name}_val"
                for val in sorted(str(v) for v in entry.possible_values):
                    constants.append(PDDLObject(val, type_name))

        return constants

    def _build_predicates(
        self,
        attribute_catalog: Dict[str, AttributeCatalogEntry],
        negated_attributes: set,
    ) -> List[PDDLPredicate]:
        """Build PDDL predicates: marked + attribute predicates.

        Negative predicates (_is_not / _false) are generated only for
        attributes in negated_attributes, i.e. those that appear negated
        in at least one guard condition.
        """
        predicates: List[PDDLPredicate] = [
            PDDLPredicate("marked", [("?x", "petri_element")]),
        ]

        for attr_name, entry in sorted(attribute_catalog.items()):
            if entry.attribute_type in ("categorical", "numerical"):
                type_name = f"{attr_name}_val"
                predicates.append(
                    PDDLPredicate(f"{attr_name}_is", [("?v", type_name)])
                )
                if attr_name in negated_attributes:
                    predicates.append(
                        PDDLPredicate(f"{attr_name}_is_not", [("?v", type_name)])
                    )
            elif entry.attribute_type == "boolean":
                predicates.append(PDDLPredicate(f"{attr_name}_true"))
                if attr_name in negated_attributes:
                    predicates.append(PDDLPredicate(f"{attr_name}_false"))

        return predicates

    def _collect_place_names(self, model: PetriNetModel) -> Set[str]:
        """Collect sanitized names for all places in the Petri net."""
        names: Set[str] = set()
        for place in model.petrinet.places:
            names.add(utils.sanitize_name(place.name))
        return names

    def _collect_transition_names(self, parse_result: ParseResult) -> Set[str]:
        """Collect sanitized names for all transitions (labeled + tau)."""
        names: Set[str] = set()
        for t_name in parse_result.transitions:
            names.add(t_name)
        for _trans_obj, tau_name in parse_result.petri_net_model.silent_transitions.items():
            names.add(utils.sanitize_name(tau_name))
        return names
