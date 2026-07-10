"""PDDL problem file builder using the same predicate objects as the domain encoder."""

from typing import Any, Dict, List, Optional

import core_utils as utils
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
        init_places: Optional[List[str]] = None,
        metric: Optional[str] = None,
        cost_weight: float = 0.001,
        require_completion: bool = False,
        end_place: Optional[str] = None,
        deadline: Optional[float] = None,
        temporal: bool = False
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
            init_places: Optional list of place IDs to mark as starting tokens.
                Each is emitted as (marked <place_id>) in :init; supports AND-splits.
            metric: Optimization metric — "minimize_cost", "minimize_time",
                "minimize_weighted", or None.
            cost_weight: Scaling factor α for "minimize_weighted" metric.
                Emits ``(:metric minimize (+ (total-time) (* α (total-cost))))``.
                Ignored for other metric values.
            require_completion: If True, appends (marked end_place) to every goal clause.
            end_place: Sanitized name of the Petri net sink place; required when
                require_completion is True.

        Returns:
            Complete PDDL problem text.
        """
        init_atoms = self._build_init_atoms(init_places, init_effects, attribute_catalog, metric in ("minimize_cost", "minimize_weighted"), deadline, temporal)
        goal_str = self._build_goal(goal_sop, attribute_catalog, require_completion, end_place)
        metric_str = self._build_metric(metric, cost_weight)

        lines = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "",
            "  (:init",
        ]
        if init_atoms:
            lines += [f"    {atom}" for atom in init_atoms]
        lines += ["  )", "", "  (:goal", f"    {goal_str}", "  )"]
        if metric_str:
            lines += ["", f"  {metric_str}"]
        lines += ["", ")"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _build_metric(self, metric: Optional[str], cost_weight: float = 0.001) -> Optional[str]:
        """Return the (:metric ...) string, or None if not applicable."""
        if metric == "minimize_cost":
            return "(:metric minimize (total-cost))"
        if metric == "minimize_time":
            return "(:metric minimize (total-time))"
        if metric == "minimize_weighted":
            return f"(:metric minimize (+ (total-time) (* {cost_weight} (total-cost))))"
        return None

    def _build_init_atoms(
        self,
        init_places: Optional[List[str]],
        effects: List[Dict[str, Any]],
        catalog: Dict[str, Any],
        has_costs: bool = False,
        deadline: Optional[float] = None,
        temporal: bool = False
    ) -> List[str]:
        atoms = []
        for place in (init_places or []):
            atoms.append(PDDLEffect.marking(place).to_pddl())
        if temporal or (deadline is not None and deadline > 0):
            atoms.append("(deadline_ok)")
        if has_costs:
            atoms.append("(= (total-cost) 0)")
        if deadline is not None and deadline > 0:
            atoms.append(f"(at {deadline:.1f} (not (deadline_ok)))")
        already_set: set = set()
        for eff in effects:
            attr = eff["attribute"]
            value = utils.sanitize_value(attr ,str(eff.get("value", "")))
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
                already_set.add(attr)

        # Emit (attr_is {attr}_val_none) for every categorical/numerical
        # attribute not already initialised by init_effects.  This makes the
        # initial PDDL state well-formed: every attribute starts as "unwritten"
        # rather than simply absent, which matches the log_preprocessor semantics.
        for attr, entry in catalog.items():
            attr_type = entry.get("type") if isinstance(entry, dict) else getattr(entry, "attribute_type", None)
            if attr_type not in ("categorical", "numerical"):
                continue
            if attr in already_set:
                continue
            #none_val = utils.sanitize_value(attr, "none")
            #atoms.append(PDDLEffect.set_attr_is(attr, none_val).to_pddl())

        return atoms

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------

    def _build_goal(
        self,
        sop: List[List[Dict[str, Any]]],
        catalog: Dict[str, Any],
        require_completion: bool = False,
        end_place: Optional[str] = None,
    ) -> str:
        completion_atom = f"(marked {end_place})" if require_completion and end_place else None

        non_empty = [clause for clause in sop if clause]

        if not non_empty:
            return completion_atom if completion_atom else "(and)"

        clauses = [
            self._render_clause(clause, catalog, completion_atom)
            for clause in non_empty
        ]

        if len(clauses) == 1:
            return clauses[0]
        return "(or " + " ".join(clauses) + ")"

    def _render_clause(
        self,
        conditions: List[Dict[str, Any]],
        catalog: Dict[str, Any],
        extra_atom: Optional[str] = None,
    ) -> str:
        atoms = [self._condition_to_pddl(cond, catalog) for cond in conditions]
        if extra_atom:
            atoms.append(extra_atom)
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
