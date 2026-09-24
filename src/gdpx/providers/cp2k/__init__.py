"""CP2K potential, materialization, and execution provider."""

from ..adapters import BackendMaterializer

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from gdpx.providers.capabilities import CapabilityKind
from gdpx.providers.provider import Provider
from gdpx.providers.specs import freeze, thaw
from gdpx.providers.targets import AseCalculatorMaterialization, NativeInputMaterialization


@dataclass(frozen=True)
class Cp2kPotential:
    parameters: Mapping[str, Any]


    def __post_init__(self):
        object.__setattr__(self, "parameters", freeze(self.parameters))


class Cp2kPotentialFactory:
    def create(self, parameters, **context):
        copied = copy.deepcopy(dict(parameters))
        if "interface" in copied:
            raise ValueError("Move parameters.interface to potential.backend; use interactive for shell/interactive interfaces.")
        copied.pop("backend", None)
        copied.pop("version", None)
        return Cp2kPotential(copied)


class Cp2kMaterializer:
    def __init__(self, target, backend="cp2k"):
        self.target = target
        self.backend = backend

    def materialize(self, potential, target=None, **context):
        if not isinstance(potential, Cp2kPotential):
            raise TypeError(f"Expected Cp2kPotential, got {type(potential).__name__}.")
        from .manager import Cp2kManager

        interface = self.backend
        parameters = thaw(potential.parameters)
        parameters["backend"] = interface
        manager = Cp2kManager()
        manager.register_calculator(parameters)
        if self.target == "ase.calculator":
            return AseCalculatorMaterialization(manager.calc)
        return NativeInputMaterialization(
            provider="cp2k",
            parameters={"manager": manager, "interface": interface},
        )


@dataclass(frozen=True)
class Cp2kExecutorFactory:
    method: str
    target: str = "cp2k.native"

    def create(self, parameters, **context):
        materialization = context["materialization"]
        if not isinstance(materialization, NativeInputMaterialization):
            raise TypeError("CP2K executor requires cp2k.native materialization.")
        manager = materialization.parameters["manager"]
        params = copy.deepcopy(dict(parameters))
        ignore_convergence = params.pop("ignore_convergence", False)
        random_seed = params.pop("random_seed", None)
        if self.method == "neb":
            from .path import Cp2kStringReactor

            params["task"] = "neb"
            executor = Cp2kStringReactor(
                manager.calc,
                params,
                directory=getattr(manager.calc, "directory", "./"),
                random_seed=random_seed,
            )
        else:
            from .driver import Cp2kDriver

            params["task"] = "ts" if self.method == "dimer" else self.method
            if self.method == "dimer":
                params.setdefault("controller", {"name": "dimer"})
            executor = Cp2kDriver(
                manager.calc,
                params,
                directory=getattr(manager.calc, "directory", "./"),
                ignore_convergence=ignore_convergence,
                random_seed=random_seed,
            )
        executor.pot_params = manager._implementation_config()
        return executor


CP2K_PROVIDER = Provider(
    name="cp2k",
    version="1",
    capabilities={
        CapabilityKind.POTENTIAL: {"default": Cp2kPotentialFactory()},
        CapabilityKind.MATERIALIZER: {
            "ase.calculator": BackendMaterializer("cp2k", {b: Cp2kMaterializer("ase.calculator", b) for b in ("cp2k", "interactive")}),
            "cp2k.native": Cp2kMaterializer("cp2k.native"),
        },
        CapabilityKind.EXECUTOR: {
            method: Cp2kExecutorFactory(method)
            for method in ("spc", "min", "md", "freq", "ts", "dimer", "neb")
        },
    },
)


__all__ = [
    "CP2K_PROVIDER", "Cp2kExecutorFactory", "Cp2kMaterializer",
    "Cp2kPotential", "Cp2kPotentialFactory",
]
