"""Duplication phase: adds XOR split guard conditions to transition actions."""
import dataclasses
from typing import List

import core_utils as utils
from models import ParseResult, Guard
from encoding.action_registry import ActionRegistry
from encoding.guard_encoder import and_clause_to_pddl
from encoding.pddl_model import PDDLAction, PDDLBaseAction, PDDLDurativeAction

logger = utils.get_logger(__name__)


class XorGuardDuplicator:
    """Expands transition actions based on XOR split DT guards.

    For each transition that is a branch of an XOR split with level-1 guards
    (decision tree available), the current variants of its action are multiplied
    by the number of OR-clauses in the guard:

        current variants:  [V1, V2]
        OR-clauses (SOP):  [C_a, C_b]
        result:            [V1+C_a, V1+C_b, V2+C_a, V2+C_b]

    When a single OR-clause is present no duplication occurs — the conditions
    are added to the existing variants directly (names unchanged).  When N > 1
    clauses are present, variants are renamed with a _v{i} suffix.
    """

    def __init__(self, parse_result: ParseResult) -> None:
        self._pr = parse_result

    def apply(self, registry: ActionRegistry) -> None:
        """Apply XOR guard expansion to all eligible transition actions.

        Args:
            registry: The ActionRegistry to update in place.
        """
        for trans_name, info in sorted(self._pr.transitions.items()):
            if info.xor_branch is None or info.xor_branch.guards is None:
                continue

            base_name = f"execute_{trans_name}"
            if base_name not in registry:
                logger.warning("Action %s not found in registry, skipping", base_name)
                continue

            or_clauses = info.xor_branch.guards
            if not or_clauses:
                continue

            current_variants = registry.get(base_name)
            new_variants = self._expand(current_variants, or_clauses, base_name)
            registry.replace(base_name, new_variants)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand(
        self,
        current_variants: List[PDDLBaseAction],
        or_clauses: List[List[Guard]],
        base_name: str,
    ) -> List[PDDLBaseAction]:
        """Compute the Cartesian product of current variants × OR-clauses.

        Args:
            current_variants: Existing action variants for this action.
            or_clauses: SOP guards from XorBranchInfo (outer list = OR).
            base_name: Original action name, used for renaming.

        Returns:
            Expanded list of variants with guard conditions injected.
        """
        single_clause = len(or_clauses) == 1

        expanded: List[PDDLBaseAction] = []
        for variant in current_variants:
            for clause in or_clauses:
                conditions = and_clause_to_pddl(clause)
                new_variant = self._add_conditions(variant, conditions)
                expanded.append(new_variant)

        if single_clause:
            return expanded

        # Rename to base_name_v0, base_name_v1, ...
        renamed: List[PDDLBaseAction] = []
        for i, variant in enumerate(expanded):
            renamed.append(dataclasses.replace(variant, name=f"{base_name}_v{i}"))
        return renamed

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
