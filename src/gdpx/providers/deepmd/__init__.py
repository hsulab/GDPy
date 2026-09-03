"""DeepMD model ownership and backend materialization."""

import copy
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

from ..capabilities import CapabilityKind
from ..adapters import ExecutorFactory, TrainerFactory, add_capabilities, manager_provider
from ..provider import Provider
from ..specs import Artifact, freeze, thaw
from ..targets import AseCalculatorMaterialization, LammpsPotentialMaterialization


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
        from .deepmd import DeepmdManager

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
        CapabilityKind.TRAINER: {
            "default": TrainerFactory(
                "gdpx.providers.deepmd.training.deepmd", "DeepmdTrainer"
            )
        },
    },
)
DEEPMD_PROVIDER = add_capabilities(
    DEEPMD_PROVIDER,
    CapabilityKind.DATASET_CODEC,
    {"default": TrainerFactory("gdpx.providers.deepmd.data", "DeepmdDataloader")},
)

DEEPMD_JAX_PROVIDER = manager_provider(
    "deepmd_jax", "gdpx.providers.deepmd.deepmd_jax", "DeepmdJaxManager",
    {"ase.calculator": "ase"},
    trainer=("gdpx.providers.deepmd.training.deepmd_jax", "DeepmdJaxTrainer"),
)
DEEPMD_JAX_PROVIDER = add_capabilities(
    DEEPMD_JAX_PROVIDER,
    CapabilityKind.EXECUTOR,
    {"md": ExecutorFactory(
        "gdpx.providers.deepmd.jax_driver", "DeepmdJaxDriver", "md", "ase.calculator"
    )},
)
DEEPMD_JAX_PROVIDER = add_capabilities(
    DEEPMD_JAX_PROVIDER,
    CapabilityKind.DATASET_CODEC,
    {"default": TrainerFactory("gdpx.providers.deepmd.data", "DeepmdDataloader")},
)
DEEPMD_JAX_X_PROVIDER = manager_provider(
    "deepmd_jax_x", "gdpx.providers.deepmd.deepmd_jax_x", "DeepmdJaxXManager",
    {"ase.calculator": "ase"},
)

from .deepmd import DeepmdManager
from .deepmd_jax import DeepmdJaxManager
from .deepmd_jax_x import DeepmdJaxXManager

__all__ = [
    "DeepmdManager", "DeepmdJaxManager", "DeepmdJaxXManager",
    "DEEPMD_PROVIDER", "DEEPMD_JAX_PROVIDER", "DEEPMD_JAX_X_PROVIDER",
]
