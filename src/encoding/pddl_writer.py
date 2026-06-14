from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from encoding.pddl_model import PDDLAction, PDDLDomain, PDDLObject, PDDLPredicate, PDDLType

import core_utils as utils

logger = utils.get_logger(__name__)


class PDDLWriter:
    """Serializes a PDDLDomain to PDDL text format."""

    def write_domain(self, domain: PDDLDomain, path: Path) -> str:
        """Write the domain to a .pddl file and return the text.

        Args:
            domain: The domain to serialize.
            path: Output file path.

        Returns:
            The PDDL text that was written.
        """
        text = self._render_domain(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.info("Domain written to %s", path)
        return text

    def _render_domain(self, domain: PDDLDomain) -> str:
        sections = [
            f"(define (domain {domain.name})",
            self._render_requirements(domain.requirements),
            self._render_types(domain.types),
            self._render_constants(domain.constants),
            self._render_predicates(domain.predicates),
        ]

        for action in domain.actions:
            sections.append(self._render_action(action))

        sections.append(")")
        return "\n\n".join(sections) + "\n"

    def _render_requirements(self, requirements: List[str]) -> str:
        return f"  (:requirements {' '.join(requirements)})"

    def _render_types(self, types: List[PDDLType]) -> str:
        lines = ["  (:types"]

        by_parent: Dict[str, List[str]] = defaultdict(list)
        for t in types:
            parent = t.parent if t.parent else "object"
            by_parent[parent].append(t.name)

        for parent in sorted(by_parent):
            children = " ".join(sorted(by_parent[parent]))
            lines.append(f"    {children} - {parent}")

        lines.append("  )")
        return "\n".join(lines)

    def _render_constants(self, constants: List[PDDLObject]) -> str:
        lines = ["  (:constants"]

        by_type: Dict[str, List[str]] = defaultdict(list)
        for obj in constants:
            by_type[obj.type_name].append(obj.name)

        for type_name in sorted(by_type):
            names = " ".join(sorted(by_type[type_name]))
            lines.append(f"    {names} - {type_name}")

        lines.append("  )")
        return "\n".join(lines)

    def _render_predicates(self, predicates: List[PDDLPredicate]) -> str:
        lines = ["  (:predicates"]

        for pred in predicates:
            if pred.parameters:
                params = " ".join(
                    f"{name} - {ptype}" for name, ptype in pred.parameters
                )
                lines.append(f"    ({pred.name} {params})")
            else:
                lines.append(f"    ({pred.name})")

        lines.append("  )")
        return "\n".join(lines)

    def _render_action(self, action: PDDLAction) -> str:
        lines = [f"  (:action {action.name}"]

        if action.parameters:
            params = " ".join(
                f"{name} - {ptype}" for name, ptype in action.parameters
            )
            lines.append(f"    :parameters ({params})")
        else:
            lines.append("    :parameters ()")

        if len(action.preconditions) == 0:
            lines.append("    :precondition ()")
        elif len(action.preconditions) == 1:
            lines.append(f"    :precondition {action.preconditions[0]}")
        else:
            lines.append("    :precondition (and")
            for p in action.preconditions:
                lines.append(f"      {p}")
            lines.append("    )")

        if len(action.effects) == 0:
            lines.append("    :effect ()")
        elif len(action.effects) == 1:
            lines.append(f"    :effect {action.effects[0]}")
        else:
            lines.append("    :effect (and")
            for e in action.effects:
                lines.append(f"      {e}")
            lines.append("    )")

        lines.append("  )")
        return "\n".join(lines)
