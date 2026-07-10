
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import core_utils

logger = core_utils.get_logger(__name__)

# ---------------------------------------------------------------------------
# Structured predicate / effect types
# ---------------------------------------------------------------------------

ConditionKind = Literal["marked", "attr_is", "attr_is_not", "attr_true", "attr_false"]


@dataclass(frozen=True)
class PDDLCondition:
    """A structured PDDL predicate instance used as an action precondition or guard.

    Encodes one of the five predicate shapes produced by this encoding:
        marked       → (marked <attribute>)
        attr_is      → (<attribute>_is <value>)
        attr_is_not  → (<attribute>_is_not <value>)
        attr_true    → (<attribute>_true)
        attr_false   → (<attribute>_false)

    Attributes:
        kind: Predicate shape identifier.
        attribute: Place name for "marked"; sanitized attribute name otherwise.
        value: Category/bucket value for attr_is / attr_is_not; None elsewhere.
    """

    kind: ConditionKind
    attribute: str
    value: Optional[str] = None

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def marked(cls, place: str) -> "PDDLCondition":
        return cls(kind="marked", attribute=place)

    @classmethod
    def attr_is(cls, attribute: str, value: str) -> "PDDLCondition":
        return cls(kind="attr_is", attribute=attribute, value=str(value))

    @classmethod
    def attr_is_not(cls, attribute: str, value: str) -> "PDDLCondition":
        return cls(kind="attr_is_not", attribute=attribute, value=str(value))

    @classmethod
    def attr_true(cls, attribute: str) -> "PDDLCondition":
        return cls(kind="attr_true", attribute=attribute)

    @classmethod
    def attr_false(cls, attribute: str) -> "PDDLCondition":
        return cls(kind="attr_false", attribute=attribute)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_pddl(self) -> str:
        """Render to a PDDL predicate string."""
        attr = core_utils.sanitize_name(self.attribute)
        value = core_utils.sanitize_value(self.attribute, self.value)
        if self.kind == "marked":
            return f"(marked {attr})"
        if self.kind == "attr_is":
            return f"({attr}_is {value})"
        if self.kind == "attr_is_not":
            return f"({attr}_is_not {value})"
        if self.kind == "attr_true":
            return f"({attr}_true)"
        if self.kind == "attr_false":
            return f"({attr}_false)"
        raise ValueError(f"Unknown kind: {self.kind!r}")

    def __str__(self) -> str:
        return self.to_pddl()

    # ------------------------------------------------------------------
    # Logic helpers
    # ------------------------------------------------------------------

    def logical_negation(self) -> "PDDLCondition":
        """Return the predicate that is logically opposite to this one.

        Raises:
            ValueError: If called on a "marked" predicate (no negation in this encoding).
        """
        if self.kind == "attr_is":
            return PDDLCondition(kind="attr_is_not", attribute=self.attribute, value=self.value)
        if self.kind == "attr_is_not":
            return PDDLCondition(kind="attr_is", attribute=self.attribute, value=self.value)
        if self.kind == "attr_true":
            return PDDLCondition(kind="attr_false", attribute=self.attribute)
        if self.kind == "attr_false":
            return PDDLCondition(kind="attr_true", attribute=self.attribute)
        raise ValueError(f"Cannot negate a '{self.kind}' predicate")

    def same_attribute_as(self, other: "PDDLCondition") -> bool:
        """True when both conditions reference the same attribute (neither is 'marked')."""
        return (
            self.kind != "marked"
            and other.kind != "marked"
            and self.attribute == other.attribute
        )


@dataclass(frozen=True)
class PDDLEffect:
    """A structured PDDL effect literal.

    Represents either a positive assertion or a (not (...)) retraction.

    Attributes:
        kind: Predicate shape (same vocabulary as PDDLCondition).
        attribute: Place name for "marked"; sanitized attribute name otherwise.
        value: Category/bucket value for attr_is / attr_is_not; None elsewhere.
        clear: When True the effect is wrapped in (not (...)).
    """

    kind: ConditionKind
    attribute: str
    value: Optional[str] = None
    clear: bool = False

    # ------------------------------------------------------------------
    # Factory classmethods — positive effects
    # ------------------------------------------------------------------

    @classmethod
    def marking(cls, name: str) -> "PDDLEffect":
        return cls(kind="marked", attribute=name)

    @classmethod
    def unmarking(cls, name: str) -> "PDDLEffect":
        return cls(kind="marked", attribute=name, clear=True)

    @classmethod
    def set_attr_is(cls, attribute: str, value: str) -> "PDDLEffect":
        return cls(kind="attr_is", attribute=attribute, value=str(value))

    @classmethod
    def set_attr_is_not(cls, attribute: str, value: str) -> "PDDLEffect":
        return cls(kind="attr_is_not", attribute=attribute, value=str(value))

    @classmethod
    def set_attr_true(cls, attribute: str) -> "PDDLEffect":
        return cls(kind="attr_true", attribute=attribute)

    @classmethod
    def set_attr_false(cls, attribute: str) -> "PDDLEffect":
        return cls(kind="attr_false", attribute=attribute)

    # ------------------------------------------------------------------
    # Factory classmethods — clear (retraction) effects
    # ------------------------------------------------------------------

    @classmethod
    def clear_attr_is(cls, attribute: str, value: str) -> "PDDLEffect":
        return cls(kind="attr_is", attribute=attribute, value=str(value), clear=True)

    @classmethod
    def clear_attr_is_not(cls, attribute: str, value: str) -> "PDDLEffect":
        return cls(kind="attr_is_not", attribute=attribute, value=str(value), clear=True)

    @classmethod
    def clear_attr_true(cls, attribute: str) -> "PDDLEffect":
        return cls(kind="attr_true", attribute=attribute, clear=True)

    @classmethod
    def clear_attr_false(cls, attribute: str) -> "PDDLEffect":
        return cls(kind="attr_false", attribute=attribute, clear=True)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_pddl(self) -> str:
        """Render to a PDDL effect string."""
        attr = core_utils.sanitize_name(self.attribute)
        value = core_utils.sanitize_value(self.attribute, self.value)
        if self.kind == "marked":
            inner = f"(marked {attr})"
        elif self.kind == "attr_is":
            inner = f"({attr}_is {value})"
        elif self.kind == "attr_is_not":
            inner = f"({attr}_is_not {value})"
        elif self.kind == "attr_true":
            inner = f"({attr}_true)"
        elif self.kind == "attr_false":
            inner = f"({attr}_false)"
        else:
            raise ValueError(f"Unknown kind: {self.kind!r}")
        return f"(not {inner})" if self.clear else inner

    def __str__(self) -> str:
        return self.to_pddl()


# ---------------------------------------------------------------------------
# PDDL domain declaration types
# ---------------------------------------------------------------------------

@dataclass
class PDDLType:
    """A PDDL type declaration with optional parent type."""
    name: str
    parent: Optional[str] = None

    def __str__(self) -> str:
        return core_utils.sanitize_name(self.name)


@dataclass
class PDDLObject:
    """A PDDL constant or object instance."""
    name: str
    type_name: str

    def __str__(self) -> str:
        return core_utils.sanitize_name(self.name)


@dataclass
class PDDLPredicate:
    """A PDDL predicate *declaration* with typed parameters (used in :predicates block)."""
    name: str
    parameters: List[Tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        name = core_utils.sanitize_name(self.name)
        if not self.parameters:
            return f"({name})"
        ptypes_dict = defaultdict(list)
        for pname, ptype in self.parameters:
            ptypes_dict[ptype].append(pname)
        # pname is a PDDL variable (e.g. "?v") — must not be sanitized
        params = " ".join(
            f"{' '.join(pnames)} - {core_utils.sanitize_name(ptype)}"
            for ptype, pnames in ptypes_dict.items()
        )
        return f"({name} {params})"


# ---------------------------------------------------------------------------
# Container types for domain sections
# ---------------------------------------------------------------------------

class _GroupedBlockMixin:
    """Shared rendering for containers that group their items by a key into a
    `(:keyword\n    member1 member2 - key\n  )` block (used by PDDLConstants
    and PDDLTypes, which differ only in what they group by).
    """

    items: List[Any]

    def _render_grouped_block(self, keyword: str, key_fn) -> str:
        groups: Dict[str, List[str]] = defaultdict(list)
        for item in self.items:
            groups[key_fn(item)].append(str(item))
        lines = [f"  ({keyword}"]
        for key in sorted(groups):
            members = " ".join(sorted(groups[key]))
            lines.append(f"    {members} - {key}")
        lines.append("  )")
        return "\n".join(lines)


@dataclass
class PDDLConstants(_GroupedBlockMixin):
    """Container for PDDL constants; serializes to a (:constants ...) block.

    Groups constants by type when rendering. Both name and type_name are
    sanitized at render time via PDDLObject.__str__ and sanitize_name.
    """
    items: List[PDDLObject] = field(default_factory=list)

    def add(self, obj: PDDLObject) -> None:
        self.items.append(obj)

    def __iter__(self):
        return iter(self.items)

    def __str__(self) -> str:
        return self._render_grouped_block(
            ":constants", lambda obj: core_utils.sanitize_name(obj.type_name)
        )


@dataclass
class PDDLTypes(_GroupedBlockMixin):
    """Container for PDDL type declarations; serializes to a (:types ...) block.

    Groups types by parent when rendering (types with no parent are grouped
    under "object").
    """
    items: List[PDDLType] = field(default_factory=list)

    def add(self, t: PDDLType) -> None:
        self.items.append(t)

    def __iter__(self):
        return iter(self.items)

    def __str__(self) -> str:
        return self._render_grouped_block(
            ":types",
            lambda t: core_utils.sanitize_name(t.parent) if t.parent else "object",
        )


@dataclass
class PDDLPredicates:
    """Container for PDDL predicate declarations; serializes to a (:predicates ...) block."""
    items: List[PDDLPredicate] = field(default_factory=list)

    def add(self, pred: PDDLPredicate) -> None:
        self.items.append(pred)

    def __iter__(self):
        return iter(self.items)

    def __str__(self) -> str:
        lines = ["  (:predicates"]
        for pred in self.items:
            lines.append(f"    {pred}")
        lines.append("  )")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDDL action types
# ---------------------------------------------------------------------------

@dataclass
class PDDLBaseAction:
    """Common base for all PDDL action types.

    Holds only the fields shared by instantaneous and durative actions.

    Attributes:
        base_cost: Fixed action cost (reserved for future use).
        additional_cost: XOR branch cost derived from statistical fallback: -log(p).
        effect_probability: Accumulated probability from conditional effect
            duplication.  Starts at 1.0; multiplied by each probabilistic
            effect's presence/value probability during EffectDuplicator
            processing.  Converted to -log(p) and added to additional_cost
            at finalisation time.
        effect_attributes: Sanitized names of attributes modified by this
            action's effects.  Populated by ActionBuilder for deterministic
            effects and updated by EffectDuplicator as conditional effects
            are resolved.  Used for fast compatibility checks during
            duplication.
    """
    name: str
    parameters: List[Tuple[str, str]] = field(default_factory=list)
    base_cost: Optional[float] = None
    additional_cost: Optional[float] = None
    effect_probability: float = 1.0
    effect_attributes: Set[str] = field(default_factory=set)
    has_costs: bool = False
    has_deadline: bool = False

    # ------------------------------------------------------------------
    # Rendering helpers shared by PDDLAction / PDDLDurativeAction
    # ------------------------------------------------------------------

    def _total_cost(self) -> float:
        return (self.base_cost or 1.0) + (self.additional_cost or 0.0)

    def _render_parameters_block(self) -> str:
        if self.parameters:
            by_type: Dict[str, List[str]] = defaultdict(list)
            for pname, ptype in self.parameters:
                type_name = core_utils.sanitize_name(ptype)
                by_type[type_name].append(pname)
            params = " ".join(
                f"{' '.join(pnames)} - {ptype}"
                for ptype, pnames in by_type.items()
            )
            return f"    :parameters ({params})"
        else:
            return "    :parameters ()"

    @staticmethod
    def _and_block(ordered: List[str], base_indent: str) -> str:
        """Join already-sorted rendered items into a single PDDL expression.

        A single item is returned bare; multiple items are wrapped in
        `(and ...)`, indented two spaces deeper than base_indent and closed
        back at base_indent. Shared by all four block-rendering wrappers
        below, which differ only in keyword/timing syntax and indentation.
        """
        if not ordered:
            return "()"
        if len(ordered) == 1:
            return ordered[0]
        inner = f"\n{base_indent}  ".join(ordered)
        return f"(and\n{base_indent}  {inner}\n{base_indent})"

    @classmethod
    def _render_condition_block(
        cls,
        keyword: str,
        items: Set[PDDLCondition],
        indent: str = "    ",
        extra_atoms: Optional[List[str]] = None,
    ) -> str:
        ordered = sorted([str(c) for c in items] + (extra_atoms or []))
        return f"{indent}{keyword} {cls._and_block(ordered, indent)}"

    @classmethod
    def _render_effect_block(
        cls,
        keyword: str,
        items: List[PDDLEffect],
        indent: str = "    ",
        cost: Optional[float] = None,
    ) -> str:
        ordered = sorted(str(e) for e in items)
        if cost is not None:
            ordered.append(f"(increase (total-cost) {cost:.4f})")
        return f"{indent}{keyword} {cls._and_block(ordered, indent)}"

    @classmethod
    def _render_timed_block(
        cls,
        timing: str,
        items: Set[PDDLCondition],
        extra_atoms: Optional[List[str]] = None,
    ) -> str:
        ordered = sorted([str(c) for c in items] + (extra_atoms or []))
        return f"({timing} {cls._and_block(ordered, '      ')})"

    @classmethod
    def _render_timed_effect_block(
        cls, timing: str, items: List[PDDLEffect], cost: Optional[float] = None
    ) -> str:
        ordered = sorted(str(e) for e in items)
        if cost is not None:
            ordered.append(f"(increase (total-cost) {cost:.4f})")
        return f"({timing} {cls._and_block(ordered, '      ')})"


@dataclass
class PDDLAction(PDDLBaseAction):
    """A PDDL instantaneous action."""
    preconditions: Set[PDDLCondition] = field(default_factory=set)
    effects: List[PDDLEffect] = field(default_factory=list)

    def to_pddl(
        self,
        deadline_predicate: str = "deadline_ok",
    ) -> str:
        """Render to a complete (:action ...) block."""
        deadline_atoms = [f"({deadline_predicate})"] if self.has_deadline else []
        cost = self._total_cost() if self.has_costs else None
        lines = [
            f"  (:action {core_utils.sanitize_name(self.name)}",
            self._render_parameters_block(),
            self._render_condition_block(
                ":precondition", self.preconditions,
                indent="    ", extra_atoms=deadline_atoms
            ),
            self._render_effect_block(":effect", self.effects, indent="    ", cost=cost),
            "  )"
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_pddl()


@dataclass
class PDDLDurativeAction(PDDLBaseAction):
    """A PDDL durative action with a duration constraint.

    Conditions are split into at-start, over-all, and at-end slots.
    Effects are split into at-start and at-end slots.
    All collection fields default to empty; populate only the slots you need.

    Attributes:
        duration_min: Lower bound for the duration constraint (seconds).
        duration_max: Upper bound for the duration constraint (seconds).
        conditions_at_start: Conditions that must hold when the action begins.
        conditions_over_all: Conditions that must hold throughout execution.
        conditions_at_end: Conditions that must hold when the action ends.
        effects_at_start: Effects applied at the start of the action.
        effects_at_end: Effects applied at the end of the action.
    """
    duration_min: float = 0.0
    duration_max: float = 0.0
    conditions_at_start: Set[PDDLCondition] = field(default_factory=set)
    conditions_over_all: Set[PDDLCondition] = field(default_factory=set)
    conditions_at_end: Set[PDDLCondition] = field(default_factory=set)
    effects_at_start: List[PDDLEffect] = field(default_factory=list)
    effects_at_end: List[PDDLEffect] = field(default_factory=list)

    @property
    def effects(self) -> List[PDDLEffect]:
        return self.effects_at_end

    @property
    def preconditions(self) -> Set[PDDLCondition]:
        return self.conditions_at_start

    def to_pddl(
        self,
        deadline_predicate: str = "deadline_ok",
    ) -> str:
        """Render to a complete (:durative-action ...) block."""
        lines = [f"  (:durative-action {core_utils.sanitize_name(self.name)}", self._render_parameters_block(),
                 f"    :duration (and (>= ?duration {self.duration_min})"
                 f" (<= ?duration {self.duration_max}))"]

        condition_parts = []
        if self.conditions_at_start:
            condition_parts.append(self._render_timed_block("at start", self.conditions_at_start))
        if self.conditions_over_all or self.has_deadline:
            condition_parts.append(self._render_timed_block("over all", self.conditions_over_all, [f"({deadline_predicate})"] if self.has_deadline else []))
        if self.conditions_at_end:
            condition_parts.append(self._render_timed_block("at end", self.conditions_at_end))

        if not condition_parts:
            lines.append("    :condition ()")
        elif len(condition_parts) == 1:
            lines.append(f"    :condition {condition_parts[0]}")
        else:
            lines.append("    :condition (and")
            for part in condition_parts:
                lines.append(f"      {part}")
            lines.append("    )")

        cost = self._total_cost() if self.has_costs else None
        effect_parts = []
        if self.effects_at_start:
            effect_parts.append(self._render_timed_effect_block("at start", self.effects_at_start))
        if self.effects_at_end or cost is not None:
            effect_parts.append(self._render_timed_effect_block("at end", self.effects_at_end, cost=cost))

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

    def __str__(self) -> str:
        return self.to_pddl()


# ---------------------------------------------------------------------------
# PDDL domain
# ---------------------------------------------------------------------------

@dataclass
class PDDLDomain:
    """Complete PDDL domain representation."""
    name: str
    requirements: List[str] = field(default_factory=list)
    types: PDDLTypes = field(default_factory=PDDLTypes)
    constants: PDDLConstants = field(default_factory=PDDLConstants)
    predicates: PDDLPredicates = field(default_factory=PDDLPredicates)
    actions: List[PDDLBaseAction] = field(default_factory=list)
    has_costs: bool = False
    has_deadline: bool = False
    deadline_predicate: str = "deadline_ok"

    def __post_init__(self) -> None:
        if self.has_deadline:
            self.predicates.add(PDDLPredicate(self.deadline_predicate))
            for act in self.actions:
                act.has_deadline = True
        if self.has_costs:
            for act in self.actions:
                act.has_costs = True

    def write(self, path: Path) -> str:
        """Serialize this domain to PDDL text, write it to path, and return the text."""
        text = str(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.info("Domain written to %s", path)
        return text

    def __str__(self) -> str:
        sections = [
            f"(define (domain {self.name})",
            f"  (:requirements {' '.join(self.requirements)})",
            str(self.types),
            str(self.constants),
            str(self.predicates),
        ]
        if self.has_costs:
            sections.append("  (:functions\n    (total-cost)\n  )")
        for action in self.actions:
            sections.append(str(action))
        sections.append(")")
        return "\n\n".join(sections) + "\n"
