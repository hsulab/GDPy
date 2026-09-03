"""ABACUS potential and execution provider."""

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from ..capabilities import CapabilityKind
from ..provider import Provider
from ..specs import freeze, thaw
from ..targets import AseCalculatorMaterialization, NativeInputMaterialization


@dataclass(frozen=True)
class AbacusPotential:
    parameters: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "parameters", freeze(self.parameters))


class AbacusPotentialFactory:
    def create(self, parameters, **context):
        copied = copy.deepcopy(dict(parameters))
        copied.pop("backend", None)
        copied.pop("version", None)
        return AbacusPotential(copied)


class AbacusMaterializer:
    def __init__(self, target):
        self.target = target

    def materialize(self, potential, target=None, **context):
        if not isinstance(potential, AbacusPotential):
            raise TypeError(f"Expected AbacusPotential, got {type(potential).__name__}.")
        from .manager import AbacusManager

        parameters = thaw(potential.parameters)
        parameters["backend"] = "abacus"
        manager = AbacusManager()
        manager.register_calculator(parameters)
        if self.target == "ase.calculator":
            return AseCalculatorMaterialization(manager.calc)
        return NativeInputMaterialization("abacus", {"manager": manager})


@dataclass(frozen=True)
class AbacusExecutorFactory:
    method: str
    target: str = "abacus.native"

    def create(self, parameters, **context):
        materialization = context["materialization"]
        if not isinstance(materialization, NativeInputMaterialization):
            raise TypeError("ABACUS executor requires abacus.native materialization.")
        from .driver import AbacusDriver

        manager = materialization.parameters["manager"]
        params = copy.deepcopy(dict(parameters))
        params["task"] = self.method
        ignore_convergence = params.pop("ignore_convergence", False)
        random_seed = params.pop("random_seed", None)
        executor = AbacusDriver(
            manager.calc,
            params,
            directory=getattr(manager.calc, "directory", "./"),
            ignore_convergence=ignore_convergence,
            random_seed=random_seed,
        )
        executor.pot_params = manager._implementation_config()
        return executor


ABACUS_PROVIDER = Provider(
    "abacus",
    "1",
    {
        CapabilityKind.POTENTIAL: {"default": AbacusPotentialFactory()},
        CapabilityKind.MATERIALIZER: {
            "ase.calculator": AbacusMaterializer("ase.calculator"),
            "abacus.native": AbacusMaterializer("abacus.native"),
        },
        CapabilityKind.EXECUTOR: {
            method: AbacusExecutorFactory(method) for method in ("scf", "min", "md")
        },
    },
)

__all__ = ["ABACUS_PROVIDER", "AbacusExecutorFactory", "AbacusPotential", "AbacusPotentialFactory"]
