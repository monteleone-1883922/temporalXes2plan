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
class PDDLAction:
    """A PDDL instantaneous action."""
    name: str
    parameters: List[Tuple[str, str]] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)


@dataclass
class PDDLDomain:
    """Complete PDDL domain representation."""
    name: str
    requirements: List[str] = field(default_factory=list)
    types: List[PDDLType] = field(default_factory=list)
    constants: List[PDDLObject] = field(default_factory=list)
    predicates: List[PDDLPredicate] = field(default_factory=list)
    actions: List[PDDLAction] = field(default_factory=list)
