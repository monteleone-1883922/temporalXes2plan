from typing import Dict, List, Set

import core_utils as utils
from models import AttributeCatalogEntry, ParseResult, PetriNetModel
from encoding.pddl_model import PDDLAction, PDDLDomain, PDDLObject, PDDLPredicate, PDDLType
from encoding.action_builder import ActionBuilder

logger = utils.get_logger(__name__)


class DomainBuilder:
    """Builds a PDDLDomain from a ParseResult."""

    def build(self, parse_result: ParseResult, domain_name: str = "process") -> PDDLDomain:
        """Transform a ParseResult into a complete PDDLDomain.

        Args:
            parse_result: Encoder-ready output from the parsing pipeline.
            domain_name: Name for the PDDL domain.

        Returns:
            A fully populated PDDLDomain.
        """
        types = self._build_types(parse_result.attribute_catalog)
        constants = self._build_constants(parse_result)
        predicates = self._build_predicates(parse_result.attribute_catalog)

        action_builder = ActionBuilder(parse_result)
        actions = action_builder.build_all()

        return PDDLDomain(
            name=domain_name,
            requirements=[":strips", ":typing"],
            types=types,
            constants=constants,
            predicates=predicates,
            actions=actions,
        )

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
        self, attribute_catalog: Dict[str, AttributeCatalogEntry]
    ) -> List[PDDLPredicate]:
        """Build PDDL predicates: marked + attribute predicates."""
        predicates: List[PDDLPredicate] = [
            PDDLPredicate("marked", [("?x", "petri_element")]),
        ]

        for attr_name, entry in sorted(attribute_catalog.items()):
            if entry.attribute_type in ("categorical", "numerical"):
                type_name = f"{attr_name}_val"
                predicates.append(
                    PDDLPredicate(f"{attr_name}_is", [("?v", type_name)])
                )
                predicates.append(
                    PDDLPredicate(f"{attr_name}_is_not", [("?v", type_name)])
                )
            elif entry.attribute_type == "boolean":
                predicates.append(PDDLPredicate(f"{attr_name}_true"))
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
