"""LASP potential and native execution provider."""

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from ..capabilities import CapabilityKind
from ..provider import Provider
from ..specs import freeze, thaw
from ..targets import AseCalculatorMaterialization, NativeInputMaterialization


@dataclass(frozen=True)
class LaspPotential:
    parameters: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "parameters", freeze(self.parameters))


class LaspPotentialFactory:
    def create(self, parameters, **context):
        copied = copy.deepcopy(dict(parameters))
        copied.pop("backend", None)
        copied.pop("version", None)
        return LaspPotential(copied)


class LaspMaterializer:
    def __init__(self, target):
        self.target = target

    def materialize(self, potential, target=None, **context):
        if not isinstance(potential, LaspPotential):
            raise TypeError(f"Expected LaspPotential, got {type(potential).__name__}.")
        from .manager import LaspManager

        parameters = thaw(potential.parameters)
        parameters["backend"] = "lasp"
        manager = LaspManager()
        manager.register_calculator(parameters)
        if self.target == "ase.calculator":
            return AseCalculatorMaterialization(manager.calc)
        return NativeInputMaterialization("lasp", {"manager": manager})


@dataclass(frozen=True)
class LaspExecutorFactory:
    method: str
    target: str = "lasp.native"

    def create(self, parameters, **context):
        materialization = context["materialization"]
        if not isinstance(materialization, NativeInputMaterialization):
            raise TypeError("LASP executor requires lasp.native materialization.")
        from .driver import LaspDriver

        manager = materialization.parameters["manager"]
        params = copy.deepcopy(dict(parameters))
        params["task"] = self.method
        ignore_convergence = params.pop("ignore_convergence", False)
        random_seed = params.pop("random_seed", None)
        executor = LaspDriver(
            manager.calc,
            params,
            directory=getattr(manager.calc, "directory", "./"),
            ignore_convergence=ignore_convergence,
            random_seed=random_seed,
        )
        executor.pot_params = manager._implementation_config()
        return executor


LASP_PROVIDER = Provider(
    "lasp",
    "1",
    {
        CapabilityKind.POTENTIAL: {"default": LaspPotentialFactory()},
        CapabilityKind.MATERIALIZER: {
            "ase.calculator": LaspMaterializer("ase.calculator"),
            "lasp.native": LaspMaterializer("lasp.native"),
        },
        CapabilityKind.EXECUTOR: {
            method: LaspExecutorFactory(method) for method in ("spc", "min", "cmin", "md")
        },
    },
)

__all__ = ["LASP_PROVIDER", "LaspExecutorFactory", "LaspPotential", "LaspPotentialFactory"]
