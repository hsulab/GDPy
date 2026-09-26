"""ASE execution provider."""

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from ..capabilities import CapabilityKind
from ..provider import Provider
from ..specs import freeze, thaw
from ..targets import AseCalculatorMaterialization


@dataclass(frozen=True)
class AsePotential:
    method: str
    parameters: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "parameters", freeze(self.parameters))


class AsePotentialFactory:
    def create(self, parameters, **context):
        copied = copy.deepcopy(dict(parameters))
        copied.pop("backend", None)
        copied.pop("version", None)
        method = copied.pop("method", None)
        if not method:
            raise ValueError("ASE potential requires a `method` parameter.")
        return AsePotential(method, copied)


class AsePotentialMaterializer:
    backend = "ase"

    def materialize(self, potential, target=None, **context):
        if not isinstance(potential, AsePotential):
            raise TypeError(f"Expected AsePotential, got {type(potential).__name__}.")
        if potential.method == "lj":
            from ase.calculators.lj import LennardJones as calculator_class
        elif potential.method == "morse":
            from ase.calculators.morse import MorsePotential as calculator_class
        elif potential.method == "tip3p":
            from ase.calculators.tip3p import TIP3P as calculator_class
        else:
            raise ValueError(f"Unsupported ASE potential {potential.method!r}.")
        return AseCalculatorMaterialization(calculator_class(**thaw(potential.parameters)))


@dataclass(frozen=True)
class AseExecutorFactory:
    method: str
    target: str = "ase.calculator"

    def create(self, parameters: Mapping[str, Any], **context: Any):
        from .driver import AseDriver

        materialization = context["materialization"]
        if not isinstance(materialization, AseCalculatorMaterialization):
            raise TypeError(
                f"ASE executor requires AseCalculatorMaterialization, got {type(materialization).__name__}."
            )
        params = copy.deepcopy(dict(parameters))
        task = "ts" if self.method == "dimer" else self.method
        params["task"] = task
        if self.method == "dimer":
            params.setdefault("controller", {"name": "dimer_ts"})
        ignore_convergence = params.pop("ignore_convergence", False)
        random_seed = params.pop("random_seed", None)
        calculator = materialization.calculator
        executor_class = AseDriver
        if self.method == "neb":
            from .path import AseStringReactor

            executor_class = AseStringReactor
        return executor_class(
            calculator,
            params,
            directory=getattr(calculator, "directory", "./"),
            ignore_convergence=ignore_convergence,
            random_seed=random_seed,
        )


ASE_METHODS = ("spc", "min", "cmin", "md", "dimer", "neb")
ASE_PROVIDER = Provider(
    name="ase",
    version="1",
    capabilities={
        CapabilityKind.POTENTIAL: {"default": AsePotentialFactory()},
        CapabilityKind.MATERIALIZER: {"ase.calculator": AsePotentialMaterializer()},
        CapabilityKind.EXECUTOR: {method: AseExecutorFactory(method) for method in ASE_METHODS},
    },
)
