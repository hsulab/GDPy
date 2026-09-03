"""DeepMD model ownership and backend materialization."""

import copy
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

from .capabilities import CapabilityKind
from .legacy import LegacyTrainerFactory
from .provider import Provider
from .specs import Artifact, freeze, thaw
from .targets import AseCalculatorMaterialization, LammpsPotentialMaterialization


@dataclass(frozen=True)
class DeepMDPotential:
    parameters: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "parameters", freeze(self.parameters))


class DeepMDPotentialFactory:
    def create(self, parameters, **context):
        copied = copy.deepcopy(dict(parameters))
        copied.pop("backend", None)
        copied.pop("version", None)
        if "models" in copied and "model" not in copied:
            copied["model"] = copied.pop("models")
        return DeepMDPotential(copied)


class DeepMDMaterializer:
    def __init__(self, backend):
        self.backend = backend

    def materialize(self, potential, target=None, **context):
        from gdpx.potential.deepmd.deepmd import DeepmdManager

        if not isinstance(potential, DeepMDPotential):
            raise TypeError(f"Expected DeepMDPotential, got {type(potential).__name__}.")
        parameters = thaw(potential.parameters)
        parameters["backend"] = self.backend
        manager = DeepmdManager()
        manager.register_calculator(parameters)
        models = manager.calc_params.get("model", ())
        artifacts = tuple(Artifact(pathlib.Path(path), "model") for path in models)
        if self.backend == "ase":
            return AseCalculatorMaterialization(manager.calc, artifacts)
        commands = (
            f"pair_style {manager.calc.pair_style}",
            f"pair_coeff {manager.calc.pair_coeff}",
        )
        return LammpsPotentialMaterialization(
            commands=commands,
            artifacts=artifacts,
            metadata={"provider": "deepmd"},
            calculator=manager.calc,
        )


DEEPMD_PROVIDER = Provider(
    name="deepmd",
    version="1",
    capabilities={
        CapabilityKind.POTENTIAL: {"default": DeepMDPotentialFactory()},
        CapabilityKind.MATERIALIZER: {
            "ase.calculator": DeepMDMaterializer("ase"),
            "lammps.potential": DeepMDMaterializer("lammps"),
        },
        CapabilityKind.TRAINER: {"default": LegacyTrainerFactory("deepmd")},
    },
)
