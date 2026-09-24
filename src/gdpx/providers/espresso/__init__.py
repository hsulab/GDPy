"""Quantum ESPRESSO potential provider."""

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from ..capabilities import CapabilityKind
from ..provider import Provider
from ..specs import freeze, thaw
from ..targets import AseCalculatorMaterialization


@dataclass(frozen=True)
class EspressoPotential:
    parameters: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "parameters", freeze(self.parameters))


class EspressoPotentialFactory:
    def create(self, parameters, **context):
        copied = copy.deepcopy(dict(parameters))
        copied.pop("backend", None)
        copied.pop("version", None)
        return EspressoPotential(copied)


class EspressoMaterializer:
    backend = "espresso"

    def materialize(self, potential, target=None, **context):
        if not isinstance(potential, EspressoPotential):
            raise TypeError(f"Expected EspressoPotential, got {type(potential).__name__}.")
        from .manager import EspressoManager

        parameters = thaw(potential.parameters)
        parameters["backend"] = "espresso"
        manager = EspressoManager()
        manager.register_calculator(parameters)
        return AseCalculatorMaterialization(manager.calc)


ESPRESSO_PROVIDER = Provider(
    "espresso",
    "1",
    {
        CapabilityKind.POTENTIAL: {"default": EspressoPotentialFactory()},
        CapabilityKind.MATERIALIZER: {"ase.calculator": EspressoMaterializer()},
    },
)

__all__ = ["ESPRESSO_PROVIDER", "EspressoPotential", "EspressoPotentialFactory"]
