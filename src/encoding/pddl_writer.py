from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from encoding.pddl_model import (
    PDDLAction, PDDLBaseAction, PDDLDurativeAction,
    PDDLDomain, PDDLObject, PDDLPredicate, PDDLType,
)

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
        extra_preds = [domain.deadline_predicate] if domain.has_deadline else []
        sections = [
            f"(define (domain {domain.name})",
            self._render_requirements(domain.requirements),
            self._render_types(domain.types),
            self._render_constants(domain.constants),
            self._render_predicates(domain.predicates, extra_preds),
        ]

        if domain.has_costs:
            sections.append("  (:functions\n    (total-cost)\n  )")

        for action in domain.actions:
            if isinstance(action, PDDLDurativeAction):
                sections.append(self._render_durative_action(action, domain.has_costs, domain.has_deadline, domain.deadline_predicate))
            else:
                sections.append(self._render_action(action, domain.has_costs))

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

    def _render_predicates(self, predicates: List[PDDLPredicate], extra_names: Optional[List[str]] = None) -> str:
        lines = ["  (:predicates"]

        for pred in predicates:
            if pred.parameters:
                params = " ".join(
                    f"{name} - {ptype}" for name, ptype in pred.parameters
                )
                lines.append(f"    ({pred.name} {params})")
            else:
                lines.append(f"    ({pred.name})")

        for name in (extra_names or []):
            lines.append(f"    ({name})")

        lines.append("  )")
        return "\n".join(lines)

    def _render_action(self, action: PDDLAction, has_costs: bool = False) -> str:
        lines = [f"  (:action {action.name}"]

        if action.parameters:
            params = " ".join(
                f"{name} - {ptype}" for name, ptype in action.parameters
            )
            lines.append(f"    :parameters ({params})")
        else:
            lines.append("    :parameters ()")

        lines.append(self._render_condition_block(
            ":precondition", action.preconditions, indent="    "
        ))
        cost = self._total_action_cost(action) if has_costs else None
        lines.append(self._render_effect_block(
            ":effect", action.effects, indent="    ", cost=cost
        ))

        lines.append("  )")
        return "\n".join(lines)

    def _render_durative_action(self, action: PDDLDurativeAction, has_costs: bool = False, has_deadline: bool = False, deadline_predicate: str = "deadline_exceeded") -> str:
        lines = [f"  (:durative-action {action.name}"]

        if action.parameters:
            params = " ".join(
                f"{name} - {ptype}" for name, ptype in action.parameters
            )
            lines.append(f"    :parameters ({params})")
        else:
            lines.append("    :parameters ()")

        lines.append(
            f"    :duration (and (>= ?duration {action.duration_min})"
            f" (<= ?duration {action.duration_max}))"
        )

        condition_parts = []
        if action.conditions_at_start:
            condition_parts.append(
                self._render_timed_block("at start", action.conditions_at_start)
            )
        if action.conditions_over_all:
            condition_parts.append(
                self._render_timed_block("over all", action.conditions_over_all)
            )
        if has_deadline:
            condition_parts.append(f"(over all (not ({deadline_predicate})))")
        if action.conditions_at_end:
            condition_parts.append(
                self._render_timed_block("at end", action.conditions_at_end)
            )

        if not condition_parts:
            lines.append("    :condition ()")
        elif len(condition_parts) == 1:
            lines.append(f"    :condition {condition_parts[0]}")
        else:
            lines.append("    :condition (and")
            for part in condition_parts:
                lines.append(f"      {part}")
            lines.append("    )")

        cost = self._total_action_cost(action) if has_costs else None
        effect_parts = []
        if action.effects_at_start:
            effect_parts.append(
                self._render_timed_effect_block("at start", action.effects_at_start)
            )
        if action.effects_at_end or cost is not None:
            effect_parts.append(
                self._render_timed_effect_block("at end", action.effects_at_end, cost=cost)
            )

        if not effect_parts:
            lines.append("    :effect ()")
        elif len(effect_parts) == 1:
            lines.append(f"    :effect {effect_parts[0]}")
        else:
            lines.append("    :effect (and")
            for part in effect_parts:
                lines.append(f"      {part}")
            lines.append("    )")

        lines.append("  )")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _total_action_cost(self, action: PDDLBaseAction) -> Optional[float]:
        """Return total cost if > 0, else None."""
        total = (action.base_cost or 0.0) + (action.additional_cost or 0.0)
        return total if total > 0.0 else None

    def _render_condition_block(
        self, keyword: str, items, indent: str = "    "
    ) -> str:
        """Render a :precondition block from a collection of PDDLCondition."""
        ordered = sorted(c.to_pddl() for c in items)
        if not ordered:
            return f"{indent}{keyword} ()"
        if len(ordered) == 1:
            return f"{indent}{keyword} {ordered[0]}"
        inner = f"\n{indent}  ".join(ordered)
        return f"{indent}{keyword} (and\n{indent}  {inner}\n{indent})"

    def _render_effect_block(
        self, keyword: str, items, indent: str = "    ", cost: Optional[float] = None
    ) -> str:
        """Render a :effect block from a collection of PDDLEffect, with optional cost increase."""
        ordered = sorted(e.to_pddl() for e in items)
        if cost is not None:
            ordered.append(f"(increase (total-cost) {cost:.4f})")
        if not ordered:
            return f"{indent}{keyword} ()"
        if len(ordered) == 1:
            return f"{indent}{keyword} {ordered[0]}"
        inner = f"\n{indent}  ".join(ordered)
        return f"{indent}{keyword} (and\n{indent}  {inner}\n{indent})"

    def _render_timed_block(self, timing: str, items) -> str:
        """Render a timed condition block (at start/over all/at end) from PDDLCondition."""
        ordered = sorted(c.to_pddl() for c in items)
        if len(ordered) == 1:
            return f"({timing} {ordered[0]})"
        inner = "\n        ".join(ordered)
        return f"({timing} (and\n        {inner}\n      ))"

    def _render_timed_effect_block(
        self, timing: str, items, cost: Optional[float] = None
    ) -> str:
        """Render a timed effect block (at start/at end) from PDDLEffect, with optional cost."""
        ordered = sorted(e.to_pddl() for e in items)
        if cost is not None:
            ordered.append(f"(increase (total-cost) {cost:.4f})")
        if len(ordered) == 1:
            return f"({timing} {ordered[0]})"
        inner = "\n        ".join(ordered)
        return f"({timing} (and\n        {inner}\n      ))"
