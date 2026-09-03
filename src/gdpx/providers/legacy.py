"""Lazy capability adapters for integrations awaiting physical migration."""

import copy
import importlib
from dataclasses import dataclass
from typing import Any, Mapping

from .capabilities import CapabilityKind
from .provider import Provider

POTENTIAL_NAMES = (
    "deepmd", "deepmd_jax", "deepmd_jax_x", "beann", "reann", "lasp", "mace", "nequip",
    "mattersim", "tace", "fairchem", "nnp", "cp2k", "espresso", "vasp", "ase", "classic",
    "eam", "emt", "reax", "gp", "grid", "mixer", "abacus", "xtb", "dftd3", "dftd4", "bias",
    "plumed",
)
EXECUTOR_NAMES = ("ase", "jax", "deepmd_jax", "lammps", "lasp", "abacus", "vasp", "cp2k", "replica", "grid")
EXECUTOR_METHODS = ("spc", "min", "cmin", "md", "freq", "ts", "dimer", "neb")
TRAINER_NAMES = ("deepmd", "deepmd_jax", "nequip", "mace", "beann", "reann", "nnp")


@dataclass(frozen=True)
class LegacyPotentialFactory:
    name: str

    def create(self, parameters: Mapping[str, Any], **context: Any) -> Any:
        from gdpx.potential.utils import potter_from_dict

        copied = copy.deepcopy(dict(parameters))
        version = copied.pop("version", context.get("version", "unknown"))
        return potter_from_dict({"name": self.name, "params": copied, "version": version})


@dataclass(frozen=True)
class LegacyExecutorFactory:
    backend: str
    method: str

    def create(self, parameters: Mapping[str, Any], **context: Any) -> Any:
        potential = context["potential"]
        if not hasattr(potential, "calc"):
            # Provider-v2 potentials are deliberately backend-neutral.  Old
            # executor implementations still need a potential manager, so the
            # compatibility adapter performs the late materialization here.
            from gdpx.potential import REGISTER
            from .specs import thaw

            manager = REGISTER[potential.provider]()
            calculator_parameters = thaw(potential.parameters)
            calculator_parameters["backend"] = self.backend
            manager.register_calculator(calculator_parameters)
            potential = manager
        copied = copy.deepcopy(dict(parameters))
        copied["backend"] = self.backend
        if self.method == "dimer":
            copied["task"] = "ts"
            copied.setdefault("controller", {"name": "dimer_ts"})
        else:
            copied["task"] = self.method
        legacy_execution = importlib.import_module("gdpx.execution.legacy")
        return legacy_execution.create_legacy_executor(potential, copied)


@dataclass(frozen=True)
class LegacyTrainerFactory:
    name: str

    def create(self, parameters: Mapping[str, Any], **context: Any) -> Any:
        from gdpx.factory.components import create_trainer

        return create_trainer({"name": self.name, **copy.deepcopy(dict(parameters))})


def register_legacy_providers(manager) -> None:
    """Register lazy-safe descriptors for all current built-in integrations."""
    names = set(POTENTIAL_NAMES) | set(EXECUTOR_NAMES) | set(TRAINER_NAMES)
    for name in sorted(names):
        capabilities = {}
        if name in POTENTIAL_NAMES:
            capabilities[CapabilityKind.POTENTIAL] = {"default": LegacyPotentialFactory(name)}
        if name in EXECUTOR_NAMES:
            capabilities[CapabilityKind.EXECUTOR] = {
                method: LegacyExecutorFactory(name, method) for method in EXECUTOR_METHODS
            }
        if name in TRAINER_NAMES:
            capabilities[CapabilityKind.TRAINER] = {"default": LegacyTrainerFactory(name)}
        manager.register(Provider(name=name, version="legacy", capabilities=capabilities))
