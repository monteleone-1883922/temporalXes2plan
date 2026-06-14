"""Registry that tracks action variants produced by successive duplication phases."""
from typing import Dict, List

from encoding.pddl_model import PDDLBaseAction


class ActionRegistry:
    """Maps each base action name to its current list of variants.

    Variants are accumulated incrementally: each duplication phase reads the
    current list for an action, expands it, and writes the result back via
    replace().  The final flat list is produced by all_actions().

    Place-marking and tau actions enter the registry with a single variant and
    are never touched by the duplication phases (they carry no guards).
    """

    def __init__(self, variants: Dict[str, List[PDDLBaseAction]]) -> None:
        self._variants = variants
        self._order: List[str] = list(variants)

    @classmethod
    def from_base_actions(cls, actions: List[PDDLBaseAction]) -> "ActionRegistry":
        """Initialize the registry with one variant per action.

        Args:
            actions: Flat list of base actions produced by ActionBuilder.

        Returns:
            A new ActionRegistry where each action maps to [itself].
        """
        variants: Dict[str, List[PDDLBaseAction]] = {}
        for action in actions:
            variants[action.name] = [action]
        return cls(variants)

    def replace(self, base_name: str, new_variants: List[PDDLBaseAction]) -> None:
        """Replace the current variants for a base action.

        Args:
            base_name: The original action name (key in the registry).
            new_variants: The expanded list of variants to store.
        """
        if base_name not in self._variants:
            raise KeyError(f"Action '{base_name}' not found in registry")
        self._variants[base_name] = new_variants

    def get(self, base_name: str) -> List[PDDLBaseAction]:
        """Return the current variants for a base action.

        Args:
            base_name: The original action name.

        Returns:
            Current list of variants (may be the original single-element list).
        """
        return self._variants[base_name]

    def all_actions(self) -> List[PDDLBaseAction]:
        """Flatten all variants into a single ordered list.

        Order follows the original insertion order of base action names.

        Returns:
            All current variants across all base actions.
        """
        result: List[PDDLBaseAction] = []
        for name in self._order:
            result.extend(self._variants[name])
        return result

    def __contains__(self, base_name: str) -> bool:
        return base_name in self._variants
