"""Small reusable adapters for provider-owned manager and trainer implementations.

The adapters contain no global registry or name-based dispatch.  A provider
declares the exact implementation class it owns, keeping plugin boundaries
explicit while older manager classes complete their API migration.
"""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from typing import Any, Mapping

from .capabilities import CapabilityKind
from .provider import Provider
from .specs import freeze, thaw
from .targets import AseCalculatorMaterialization, LammpsPotentialMaterialization


def _load(module: str, attribute: str):
    return getattr(importlib.import_module(module), attribute)


@dataclass(frozen=True)
class ProviderPotential:
    provider: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", freeze(self.parameters))


@dataclass(frozen=True)
class ProviderPotentialFactory:
    provider: str

    def create(self, parameters: Mapping[str, Any], **context: Any) -> ProviderPotential:
        copied = copy.deepcopy(dict(parameters))
        copied.pop("backend", None)
        copied.pop("version", None)
        return ProviderPotential(self.provider, copied)


@dataclass(frozen=True)
class ManagerMaterializer:
    provider: str
    module: str
    manager_class: str
    backend: str

    def materialize(self, potential: ProviderPotential, target: str | None = None, **context: Any):
        if not isinstance(potential, ProviderPotential) or potential.provider != self.provider:
            raise TypeError(f"Expected ProviderPotential for {self.provider!r}.")
        manager = _load(self.module, self.manager_class)()
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
                values = pair_coeff if isinstance(pair_coeff, (list, tuple)) else (pair_coeff,)
                commands.extend(f"pair_coeff {value}" for value in values)
            return LammpsPotentialMaterialization(
                commands=tuple(commands),
                metadata={"provider": self.provider},
                calculator=calculator,
            )
        raise ValueError(f"Unsupported materialization target {target!r}.")


@dataclass(frozen=True)
class TrainerFactory:
    module: str
    trainer_class: str

    def create(self, parameters: Mapping[str, Any], **context: Any):
        return _load(self.module, self.trainer_class)(**copy.deepcopy(dict(parameters)))


@dataclass(frozen=True)
class ExecutorFactory:
    """Instantiate one explicitly declared driver or reactor class."""

    module: str
    executor_class: str
    method: str
    target: str

    def create(self, parameters: Mapping[str, Any], **context: Any):
        materialization = context["materialization"]
        calculator = materialization.calculator
        params = copy.deepcopy(dict(parameters))
        params["task"] = self.method
        ignore_convergence = params.pop("ignore_convergence", False)
        random_seed = params.pop("random_seed", None)
        return _load(self.module, self.executor_class)(
            calculator,
            params,
            directory=getattr(calculator, "directory", "./"),
            ignore_convergence=ignore_convergence,
            random_seed=random_seed,
        )


def add_capabilities(
    provider: Provider,
    kind: CapabilityKind,
    implementations: Mapping[str, object],
) -> Provider:
    capabilities = {key: dict(value) for key, value in provider.capabilities.items()}
    capabilities.setdefault(kind, {}).update(implementations)
    return Provider(provider.name, provider.version, capabilities)


def manager_provider(
    name: str,
    manager_module: str,
    manager_class: str,
    targets: Mapping[str, str],
    trainer: tuple[str, str] | None = None,
) -> Provider:
    """Build a descriptor from implementation paths declared by one provider."""
    capabilities: dict[CapabilityKind, dict[str, object]] = {
        CapabilityKind.POTENTIAL: {"default": ProviderPotentialFactory(name)},
        CapabilityKind.MATERIALIZER: {
            target: ManagerMaterializer(name, manager_module, manager_class, backend)
            for target, backend in targets.items()
        },
    }
    if trainer is not None:
        capabilities[CapabilityKind.TRAINER] = {
            "default": TrainerFactory(*trainer)
        }
    return Provider(name=name, version="2", capabilities=capabilities)
