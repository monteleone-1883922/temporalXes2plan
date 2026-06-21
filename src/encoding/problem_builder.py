"""PDDL problem file builder using the same predicate objects as the domain encoder."""

from typing import Any, Dict, List, Optional

from encoding.pddl_model import PDDLCondition, PDDLEffect


class ProblemBuilder:
    """Builds a PDDL problem file from UI-supplied init effects and a SOP goal.

    Conditions and effects are converted to the same PDDLCondition / PDDLEffect
    objects used by the domain encoder, so predicate names are guaranteed to
    match the domain by construction.

    Predicate shapes produced (same as the domain):
        Boolean attr, value "true"  + predicate "=" → (attr_true)
        Boolean attr, value "true"  + predicate "<>" → (attr_false)
        Boolean attr, value "false" + predicate "=" → (attr_false)
        Boolean attr, value "false" + predicate "<>" → (attr_true)
        Other attr,   predicate "=" → (attr_is value)
        Other attr,   predicate "<>" → (attr_is_not value)
    """

    def build(
        self,
        problem_name: str,
        domain_name: str,
        init_effects: List[Dict[str, Any]],
        goal_sop: List[List[Dict[str, Any]]],
        attribute_catalog: Dict[str, Any],
        init_place: Optional[str] = None,
    ) -> str:
        """Build and return a PDDL problem definition as a string.

        Args:
            problem_name: Name embedded in the problem header.
            domain_name: Name of the PDDL domain this problem targets.
            init_effects: Attribute value assignments for :init.
                Each dict: {"attribute": str, "value": str}.
            goal_sop: Goal in Sum-of-Products form.
                Outer list = OR clauses, inner list = AND conditions.
                Each condition: {"attribute": str, "predicate": "="|"<>", "value": str}.
            attribute_catalog: Serialized catalog from current.json.
                Maps attr name → {"type": "boolean"|"numerical"|"categorical", ...}.
            init_place: Optional place ID to mark as the starting token position.
                Emitted as (marked <place_id>) as the first :init atom.

        Returns:
            Complete PDDL problem text.
        """
        init_atoms = self._build_init_atoms(init_place, init_effects, attribute_catalog)
        goal_str = self._build_goal(goal_sop, attribute_catalog)

        lines = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "",
            "  (:init",
        ]
        if init_atoms:
            lines += [f"    {atom}" for atom in init_atoms]
        lines += ["  )", "", "  (:goal", f"    {goal_str}", "  )", "", ")"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _build_init_atoms(
        self,
        init_place: Optional[str],
        effects: List[Dict[str, Any]],
        catalog: Dict[str, Any],
    ) -> List[str]:
        atoms = []
        if init_place:
            atoms.append(PDDLEffect.marking(init_place).to_pddl())
        for eff in effects:
            attr = eff["attribute"]
            value = str(eff.get("value", ""))
            is_bool = catalog.get(attr, {}).get("type") == "boolean"

            if is_bool:
                effect = (
                    PDDLEffect.set_attr_true(attr)
                    if value == "true"
                    else PDDLEffect.set_attr_false(attr)
                )
            else:
                if not value:
                    continue
                effect = PDDLEffect.set_attr_is(attr, value)

            atom = effect.to_pddl()
            if not effect.clear:
                atoms.append(atom)

        return atoms

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------

    def _build_goal(
        self,
        sop: List[List[Dict[str, Any]]],
        catalog: Dict[str, Any],
    ) -> str:
        non_empty = [clause for clause in sop if clause]
        if not non_empty:
            return "(and)"

        clauses = [self._render_clause(clause, catalog) for clause in non_empty]

        if len(clauses) == 1:
            return clauses[0]
        return "(or " + " ".join(clauses) + ")"

    def _render_clause(
        self,
        conditions: List[Dict[str, Any]],
        catalog: Dict[str, Any],
    ) -> str:
        atoms = [
            self._condition_to_pddl(cond, catalog)
            for cond in conditions
        ]
        if len(atoms) == 1:
            return atoms[0]
        return "(and " + " ".join(atoms) + ")"

    def _condition_to_pddl(
        self,
        cond: Dict[str, Any],
        catalog: Dict[str, Any],
    ) -> str:
        attr = cond["attribute"]
        value = str(cond.get("value", ""))
        predicate = cond.get("predicate", "=")
        is_bool = catalog.get(attr, {}).get("type") == "boolean"

        if is_bool:
            want_true = (predicate == "=" and value == "true") or (
                predicate == "<>" and value == "false"
            )
            return (
                PDDLCondition.attr_true(attr).to_pddl()
                if want_true
                else PDDLCondition.attr_false(attr).to_pddl()
            )

        if predicate == "=":
            return PDDLCondition.attr_is(attr, value).to_pddl()
        return PDDLCondition.attr_is_not(attr, value).to_pddl()
