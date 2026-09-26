"""Backend-neutral potential and modifier expression trees."""

from dataclasses import dataclass
from typing import Any, Tuple

from gdpx.providers.specs import ModifierSpec, PotentialSpec


class PotentialExpression:
    """Marker base for a deterministic potential expression tree."""


@dataclass(frozen=True)
class PotentialLeaf(PotentialExpression):
    potential: PotentialSpec


@dataclass(frozen=True)
class ModifiedPotential(PotentialExpression):
    source: PotentialExpression
    modifier: ModifierSpec


@dataclass(frozen=True)
class SumPotential(PotentialExpression):
    terms: Tuple[PotentialExpression, ...]

    def __post_init__(self):
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.terms:
            raise ValueError("A sum potential requires at least one term.")


@dataclass(frozen=True)
class CommitteePotential(PotentialExpression):
    members: Tuple[PotentialExpression, ...]

    def __post_init__(self):
        object.__setattr__(self, "members", tuple(self.members))
        if not self.members:
            raise ValueError("A committee requires at least one member.")


def apply_modifiers(potential: PotentialSpec, modifiers) -> PotentialExpression:
    expression: PotentialExpression = PotentialLeaf(potential)
    for modifier in modifiers:
        expression = ModifiedPotential(expression, modifier)
    return expression

