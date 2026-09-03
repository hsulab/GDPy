"""VASP potential materialization and native execution capabilities."""

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from ..capabilities import CapabilityKind
from ..provider import Provider
from ..specs import freeze, thaw
from ..targets import AseCalculatorMaterialization, NativeInputMaterialization


@dataclass(frozen=True)
class VaspPotential:
    parameters: Mapping[str, Any]
    interface: str = "vasp"

    def __post_init__(self):
        object.__setattr__(self, "parameters", freeze(self.parameters))


class VaspPotentialFactory:
    def create(self, parameters, **context):
        copied = copy.deepcopy(dict(parameters))
        interface = copied.pop("interface", copied.pop("backend", "vasp"))
        copied.pop("version", None)
        return VaspPotential(copied, interface)


class VaspMaterializer:
    def __init__(self, target, default_interface):
        self.target = target
        self.default_interface = default_interface

    def materialize(self, potential, target=None, **context):
        from .manager import VaspManager

        if not isinstance(potential, VaspPotential):
            raise TypeError(f"Expected VaspPotential, got {type(potential).__name__}.")
        interface = potential.interface
        if self.target == "vasp.native":
            interface = "vasp"
        elif interface == "vasp":
            interface = self.default_interface
        parameters = thaw(potential.parameters)
        parameters["backend"] = interface
        manager = VaspManager()
        manager.register_calculator(parameters)
        if self.target == "ase.calculator":
            return AseCalculatorMaterialization(manager.calc)
        return NativeInputMaterialization(
            provider="vasp",
            parameters={"manager": manager, "interface": interface},
        )


@dataclass(frozen=True)
class VaspExecutorFactory:
    method: str
    target: str = "vasp.native"

    def create(self, parameters, **context):
        materialization = context["materialization"]
        if not isinstance(materialization, NativeInputMaterialization):
            raise TypeError(f"VASP executor requires native input, got {type(materialization).__name__}.")
        manager = materialization.parameters["manager"]
        params = copy.deepcopy(dict(parameters))
        ignore_convergence = params.pop("ignore_convergence", False)
        random_seed = params.pop("random_seed", None)
        if self.method == "neb":
            from .path import VaspStringReactor

            params["task"] = "neb"
            executor = VaspStringReactor(
                manager.calc,
                params,
                directory=getattr(manager.calc, "directory", "./"),
                ignore_convergence=ignore_convergence,
                random_seed=random_seed,
            )
        else:
            from .driver import VaspDriver

            params["task"] = self.method
            executor = VaspDriver(
                manager.calc,
                params,
                directory=getattr(manager.calc, "directory", "./"),
                ignore_convergence=ignore_convergence,
                random_seed=random_seed,
            )
        executor.pot_params = manager.as_dict()
        return executor


VASP_PROVIDER = Provider(
    name="vasp",
    version="1",
    capabilities={
        CapabilityKind.POTENTIAL: {"default": VaspPotentialFactory()},
        CapabilityKind.MATERIALIZER: {
            "ase.calculator": VaspMaterializer("ase.calculator", "vasp_interactive"),
            "vasp.native": VaspMaterializer("vasp.native", "vasp"),
        },
        CapabilityKind.EXECUTOR: {
            method: VaspExecutorFactory(method) for method in ("spc", "min", "cmin", "md", "freq", "neb")
        },
    },
)
