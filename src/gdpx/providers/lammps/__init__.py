"""LAMMPS execution provider consuming generic LAMMPS potential payloads."""

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from ..capabilities import CapabilityKind
from ..provider import Provider
from ..targets import LammpsPotentialMaterialization


@dataclass(frozen=True)
class LammpsExecutorFactory:
    method: str
    target: str = "lammps.potential"

    def create(self, parameters: Mapping[str, Any], **context: Any):
        from .execution import LmpDriver

        materialization = context["materialization"]
        if not isinstance(materialization, LammpsPotentialMaterialization):
            raise TypeError(
                "LAMMPS executor requires LammpsPotentialMaterialization, "
                f"got {type(materialization).__name__}."
            )
        if materialization.calculator is None:
            raise ValueError("LAMMPS materialization has no runtime calculator payload.")
        params = copy.deepcopy(dict(parameters))
        params["task"] = self.method
        ignore_convergence = params.pop("ignore_convergence", False)
        random_seed = params.pop("random_seed", None)
        calculator = materialization.calculator
        return LmpDriver(
            calculator,
            params,
            directory=getattr(calculator, "directory", "./"),
            ignore_convergence=ignore_convergence,
            random_seed=random_seed,
        )


LAMMPS_PROVIDER = Provider(
    name="lammps",
    version="1",
    capabilities={
        CapabilityKind.EXECUTOR: {
            method: LammpsExecutorFactory(method) for method in ("min", "md")
        }
    },
)
