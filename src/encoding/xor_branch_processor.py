"""XOR branch processing: guard duplication (level 1) and cost assignment (level 2)."""
import dataclasses
import math
from typing import List, Optional

import core_utils as utils
from models import Guard, ParseResult
from encoding.action_registry import ActionRegistry
from encoding.guard_encoder import and_clause_to_pddl
from encoding.pddl_model import PDDLAction, PDDLBaseAction, PDDLDurativeAction

logger = utils.get_logger(__name__)


class XorBranchProcessor:
    """Processes XOR split branches according to their cascade level.

    Dispatches per-transition based on cascade_level:

    - Level 1 (DT guards available): multiplies action variants by OR-clauses
      (Cartesian product). With N > 1 clauses, variants are renamed with _v{i}.
    - Level 2 (statistical fallback): assigns additional_cost = -log(p) to all
      existing variants of the action, without duplicating them.
    - Level 3 (no data): no-op.
    """

    def __init__(self, parse_result: ParseResult) -> None:
        self._pr = parse_result

    def apply(self, registry: ActionRegistry) -> None:
        """Process all XOR branches in the registry.

        Args:
            registry: The ActionRegistry to update in place.
        """
        for trans_name, info in sorted(self._pr.transitions.items()):
            if info.xor_branch is None:
                continue

            base_name = f"execute_{trans_name}"
            if base_name not in registry:
                logger.warning("Action %s not found in registry, skipping", base_name)
                continue

            level = info.xor_branch.cascade_level
            if level == 1:
                self._apply_guards(registry, base_name, info.xor_branch.guards)
            elif level == 2:
                self._apply_cost(registry, base_name, info.xor_branch.probability)
            # level 3: no-op

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_guards(
        self,
        registry: ActionRegistry,
        base_name: str,
        or_clauses: Optional[List[List[Guard]]],
    ) -> None:
        """Expand variants by XOR guard OR-clauses (cascade_level == 1)."""
        if not or_clauses:
            return
        current_variants = registry.get(base_name)
        new_variants = self._expand(current_variants, or_clauses, base_name)
        registry.replace(base_name, new_variants)

    def _apply_cost(
        self,
        registry: ActionRegistry,
        base_name: str,
        probability: float,
    ) -> None:
        """Assign additional_cost = -log(p) to all variants (cascade_level == 2)."""
        cost = -math.log(probability)
        updated = [
            dataclasses.replace(v, additional_cost=cost)
            for v in registry.get(base_name)
        ]
        registry.replace(base_name, updated)

    def _expand(
        self,
        current_variants: List[PDDLBaseAction],
        or_clauses: List[List[Guard]],
        base_name: str,
    ) -> List[PDDLBaseAction]:
        """Compute the Cartesian product of current variants × OR-clauses.

        Args:
            current_variants: Existing action variants for this action.
            or_clauses: SOP guards (outer list = OR, inner list = AND).
            base_name: Original action name, used for renaming.

        Returns:
            Expanded list of variants with guard conditions injected.
        """
        single_clause = len(or_clauses) == 1

        expanded: List[PDDLBaseAction] = []
        for variant in current_variants:
            for clause in or_clauses:
                conditions = and_clause_to_pddl(clause)
                expanded.append(self._add_conditions(variant, conditions))

        if single_clause:
            return expanded

        return [
            dataclasses.replace(v, name=f"{base_name}_v{i}")
            for i, v in enumerate(expanded)
        ]

    def _add_conditions(
        self, action: PDDLBaseAction, conditions: List[str]
    ) -> PDDLBaseAction:
        """Return a copy of action with extra conditions added to preconditions.

        For PDDLAction: appended to preconditions.
        For PDDLDurativeAction: appended to conditions_at_start.

        Args:
            action: The action to copy and extend.
            conditions: PDDL predicate strings to add.

        Returns:
            A new action instance (original is not mutated).
        """
        if isinstance(action, PDDLDurativeAction):
            return dataclasses.replace(
                action,
                conditions_at_start=action.conditions_at_start + conditions,
            )
        if isinstance(action, PDDLAction):
            return dataclasses.replace(
                action,
                preconditions=action.preconditions + conditions,
            )
        raise TypeError(f"Unsupported action type: {type(action)}")
