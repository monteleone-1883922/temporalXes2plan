from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set, Tuple


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
        if self.kind == "marked":
            return f"(marked {self.attribute})"
        if self.kind == "attr_is":
            return f"({self.attribute}_is {self.value})"
        if self.kind == "attr_is_not":
            return f"({self.attribute}_is_not {self.value})"
        if self.kind == "attr_true":
            return f"({self.attribute}_true)"
        if self.kind == "attr_false":
            return f"({self.attribute}_false)"
        raise ValueError(f"Unknown kind: {self.kind!r}")

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
        if self.kind == "marked":
            inner = f"(marked {self.attribute})"
        elif self.kind == "attr_is":
            inner = f"({self.attribute}_is {self.value})"
        elif self.kind == "attr_is_not":
            inner = f"({self.attribute}_is_not {self.value})"
        elif self.kind == "attr_true":
            inner = f"({self.attribute}_true)"
        elif self.kind == "attr_false":
            inner = f"({self.attribute}_false)"
        else:
            raise ValueError(f"Unknown kind: {self.kind!r}")
        return f"(not {inner})" if self.clear else inner


# ---------------------------------------------------------------------------
# PDDL domain declaration types
# ---------------------------------------------------------------------------

@dataclass
class PDDLType:
    """A PDDL type declaration with optional parent type."""
    name: str
    parent: Optional[str] = None


@dataclass
class PDDLObject:
    """A PDDL constant or object instance."""
    name: str
    type_name: str


@dataclass
class PDDLPredicate:
    """A PDDL predicate *declaration* with typed parameters (used in :predicates block)."""
    name: str
    parameters: List[Tuple[str, str]] = field(default_factory=list)


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


@dataclass
class PDDLAction(PDDLBaseAction):
    """A PDDL instantaneous action."""
    preconditions: Set[PDDLCondition] = field(default_factory=set)
    effects: List[PDDLEffect] = field(default_factory=list)


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


@dataclass
class PDDLDomain:
    """Complete PDDL domain representation."""
    name: str
    requirements: List[str] = field(default_factory=list)
    types: List[PDDLType] = field(default_factory=list)
    constants: List[PDDLObject] = field(default_factory=list)
    predicates: List[PDDLPredicate] = field(default_factory=list)
    actions: List[PDDLBaseAction] = field(default_factory=list)
