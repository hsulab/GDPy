"""EMT potential provider."""

from dataclasses import dataclass
from typing import Any, Mapping

from ase.calculators.emt import EMT

from ..capabilities import CapabilityKind
from ..provider import Provider
from ..specs import freeze
from ..targets import AseCalculatorMaterialization


@dataclass(frozen=True)
class EmtPotential:
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", freeze(self.parameters))


class EmtPotentialFactory:
    def create(self, parameters, **context):
        copied = dict(parameters)
        copied.pop("backend", None)
        copied.pop("version", None)
        return EmtPotential(copied)


class EmtAseMaterializer:
    backend = "ase"

    target = "ase.calculator"

    def materialize(self, potential, target=None, **context):
        if not isinstance(potential, EmtPotential):
            raise TypeError(f"Expected EmtPotential, got {type(potential).__name__}.")
        return AseCalculatorMaterialization(EMT(**dict(potential.parameters)))


EMT_PROVIDER = Provider(
    name="emt",
    version="1",
    capabilities={
        CapabilityKind.POTENTIAL: {"default": EmtPotentialFactory()},
        CapabilityKind.MATERIALIZER: {"ase.calculator": EmtAseMaterializer()},
    },
)
