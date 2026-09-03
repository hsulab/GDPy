"""Backend-neutral adapters for legacy potential-manager implementations.

This is the migration bridge for integrations that have not yet moved their
calculator construction out of ``gdpx.potential``.  Creating the potential is
pure; the legacy manager is instantiated only when an executor selects a
materialization target.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .capabilities import CapabilityKind
from .provider import Provider
from .specs import freeze, thaw
from .targets import AseCalculatorMaterialization, LammpsPotentialMaterialization


@dataclass(frozen=True)
class ManagedPotential:
    provider: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", freeze(self.parameters))


@dataclass(frozen=True)
class ManagedPotentialFactory:
    provider: str

    def create(self, parameters: Mapping[str, Any], **context: Any) -> ManagedPotential:
        copied = copy.deepcopy(dict(parameters))
        copied.pop("backend", None)
        copied.pop("version", None)
        return ManagedPotential(self.provider, copied)


@dataclass(frozen=True)
class ManagedPotentialMaterializer:
    provider: str
    backend: str

    def materialize(self, potential: ManagedPotential, target: str | None = None, **context: Any):
        if not isinstance(potential, ManagedPotential) or potential.provider != self.provider:
            raise TypeError(f"Expected ManagedPotential for {self.provider!r}.")
        from gdpx.potential import REGISTER

        manager = REGISTER[self.provider]()
        parameters = thaw(potential.parameters)
        parameters["backend"] = self.backend
        manager.register_calculator(parameters)
        if target == "ase.calculator":
            return AseCalculatorMaterialization(manager.calc)
        if target == "lammps.potential":
            calculator = manager.calc
            commands = []
            pair_style = getattr(calculator, "pair_style", None)
            pair_coeff = getattr(calculator, "pair_coeff", None)
            if pair_style:
                commands.append(f"pair_style {pair_style}")
            if pair_coeff:
                coefficients = pair_coeff if isinstance(pair_coeff, (list, tuple)) else (pair_coeff,)
                commands.extend(f"pair_coeff {value}" for value in coefficients)
            return LammpsPotentialMaterialization(
                commands=tuple(commands),
                metadata={"provider": self.provider},
                calculator=calculator,
            )
        raise ValueError(f"Unsupported materialization target {target!r}.")


# Calculator backends that have a stable executor-facing representation.
# Native file-I/O integrations remain on the legacy executor bridge until a
# dedicated native target contract is introduced for each engine.
MANAGED_TARGETS = {
    "deepmd_jax": {"ase.calculator": "ase"},
    "deepmd_jax_x": {"ase.calculator": "ase"},
    "beann": {"ase.calculator": "ase", "lammps.potential": "lammps"},
    "reann": {"ase.calculator": "ase"},
    "mace": {"ase.calculator": "ase", "lammps.potential": "lammps"},
    "nequip": {"ase.calculator": "ase", "lammps.potential": "lammps"},
    "mattersim": {"ase.calculator": "ase", "lammps.potential": "lammps"},
    "tace": {"ase.calculator": "ase"},
    "fairchem": {"ase.calculator": "ase"},
    "nnp": {"ase.calculator": "ase"},
    "classic": {"lammps.potential": "lammps"},
    "eam": {"ase.calculator": "ase", "lammps.potential": "lammps"},
    "reax": {"lammps.potential": "lammps"},
    "gp": {"ase.calculator": "ase"},
    "mixer": {"ase.calculator": "ase"},
    "xtb": {"ase.calculator": "xtb"},
    "dftd3": {"ase.calculator": "ase"},
    "dftd4": {"ase.calculator": "ase"},
    "bias": {"ase.calculator": "ase"},
    "plumed": {"ase.calculator": "ase"},
}


def managed_provider_fragments():
    for name, targets in MANAGED_TARGETS.items():
        yield Provider(
            name=name,
            version="1",
            capabilities={
                CapabilityKind.POTENTIAL: {"default": ManagedPotentialFactory(name)},
                CapabilityKind.MATERIALIZER: {
                    target: ManagedPotentialMaterializer(name, backend)
                    for target, backend in targets.items()
                },
            },
        )
