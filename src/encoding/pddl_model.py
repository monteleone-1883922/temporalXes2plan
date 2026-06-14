from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    """A PDDL predicate declaration with typed parameters."""
    name: str
    parameters: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class PDDLBaseAction:
    """Common base for all PDDL action types.

    Holds only the fields shared by instantaneous and durative actions.

    Attributes:
        base_cost: Fixed action cost (reserved for future use, e.g. global frequency).
        additional_cost: XOR branch cost derived from statistical fallback: -log(p).
    """
    name: str
    parameters: List[Tuple[str, str]] = field(default_factory=list)
    base_cost: Optional[float] = None
    additional_cost: Optional[float] = None


@dataclass
class PDDLAction(PDDLBaseAction):
    """A PDDL instantaneous action."""
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)


@dataclass
class PDDLDurativeAction(PDDLBaseAction):
    """A PDDL durative action with a duration constraint.

    Conditions are split into at-start, over-all, and at-end slots.
    Effects are split into at-start and at-end slots.
    All list fields default to empty; populate only the slots you need.

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
    conditions_at_start: List[str] = field(default_factory=list)
    conditions_over_all: List[str] = field(default_factory=list)
    conditions_at_end: List[str] = field(default_factory=list)
    effects_at_start: List[str] = field(default_factory=list)
    effects_at_end: List[str] = field(default_factory=list)


@dataclass
class PDDLDomain:
    """Complete PDDL domain representation."""
    name: str
    requirements: List[str] = field(default_factory=list)
    types: List[PDDLType] = field(default_factory=list)
    constants: List[PDDLObject] = field(default_factory=list)
    predicates: List[PDDLPredicate] = field(default_factory=list)
    actions: List[PDDLBaseAction] = field(default_factory=list)
