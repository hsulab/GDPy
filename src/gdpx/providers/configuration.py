"""Versioned provider runtime configuration and legacy translation."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from .errors import ProviderConfigurationError
from .specs import ModifierSpec, PotentialSpec, freeze, thaw

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ComponentConfig:
    provider: str
    method: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ProviderConfigurationError("Component provider cannot be empty.")
        object.__setattr__(self, "parameters", freeze(self.parameters))

    def to_dict(self) -> dict:
        data = {"provider": self.provider}
        if self.method is not None:
            data["method"] = self.method
        data["parameters"] = thaw(self.parameters)
        return data


@dataclass(frozen=True)
class RuntimeConfig:
    potential: ComponentConfig
    executor: ComponentConfig
    modifiers: Tuple[ComponentConfig, ...] = ()
    scheduler: Optional[ComponentConfig] = None
    options: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProviderConfigurationError(
                f"Unsupported runtime schema {self.schema_version}; expected {SCHEMA_VERSION}."
            )
        object.__setattr__(self, "modifiers", tuple(self.modifiers))
        object.__setattr__(self, "options", freeze(self.options))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        raw = copy.deepcopy(dict(value))
        version = raw.pop("schema_version", None)
        if version not in (None, 1, SCHEMA_VERSION):
            raise ProviderConfigurationError(f"Unsupported runtime schema {version}.")
        if version == SCHEMA_VERSION or _is_v2(raw):
            return _parse_v2(raw)
        return _translate_legacy(raw)

    def to_dict(self) -> dict:
        data = {
            "schema_version": SCHEMA_VERSION,
            "potential": self.potential.to_dict(),
            "modifiers": [item.to_dict() for item in self.modifiers],
            "executor": self.executor.to_dict(),
        }
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.to_dict()
        data.update(thaw(self.options))
        return data

    def potential_spec(self) -> PotentialSpec:
        return PotentialSpec(
            self.potential.provider,
            self.potential.parameters,
            method=self.potential.method or "default",
        )

    def modifier_specs(self) -> Tuple[ModifierSpec, ...]:
        return tuple(
            ModifierSpec(item.provider, item.method or "", item.parameters) for item in self.modifiers
        )


def _is_v2(raw: Mapping[str, Any]) -> bool:
    potential = raw.get("potential")
    executor = raw.get("executor")
    return isinstance(potential, Mapping) and "provider" in potential and isinstance(executor, Mapping)


def _component(value: Any, label: str, *, require_method: bool = False) -> ComponentConfig:
    if not isinstance(value, Mapping):
        raise ProviderConfigurationError(f"{label.capitalize()} configuration must be a mapping.")
    data = copy.deepcopy(dict(value))
    provider = data.pop("provider", None)
    method = data.pop("method", None)
    parameters = data.pop("parameters", {})
    if data:
        raise ProviderConfigurationError(f"Unknown {label} fields: {', '.join(sorted(data))}.")
    if not isinstance(parameters, Mapping):
        raise ProviderConfigurationError(f"{label.capitalize()} parameters must be a mapping.")
    if require_method and not method:
        raise ProviderConfigurationError(f"{label.capitalize()} method is required.")
    return ComponentConfig(str(provider or ""), method, parameters)


def _parse_v2(raw: dict) -> RuntimeConfig:
    try:
        potential = _component(raw.pop("potential"), "potential")
        executor = _component(raw.pop("executor"), "executor", require_method=True)
    except KeyError as error:
        raise ProviderConfigurationError(f"Missing runtime section {error.args[0]!r}.") from error
    modifiers_value = raw.pop("modifiers", ())
    if not isinstance(modifiers_value, (list, tuple)):
        raise ProviderConfigurationError("Modifiers must be a sequence.")
    modifiers = tuple(_component(item, "modifier", require_method=True) for item in modifiers_value)
    scheduler_value = raw.pop("scheduler", None)
    scheduler = None if scheduler_value is None else _component(scheduler_value, "scheduler")
    return RuntimeConfig(potential, executor, modifiers, scheduler, options=raw)


def _translate_legacy(raw: dict) -> RuntimeConfig:
    warnings.warn(
        "Legacy potter/driver configuration is deprecated; use schema_version 2 with potential/executor sections.",
        DeprecationWarning,
        stacklevel=3,
    )
    potential_value = raw.pop("potential", raw.pop("potter", None))
    if not isinstance(potential_value, Mapping):
        raise ProviderConfigurationError("Legacy runtime configuration requires `potter` or `potential`.")
    potential_data = copy.deepcopy(dict(potential_value))
    provider = potential_data.pop("name", potential_data.pop("provider", None))
    parameters = potential_data.pop("params", potential_data.pop("parameters", {}))
    version = potential_data.pop("version", None)
    if version is not None:
        parameters = dict(parameters, version=version)
    if potential_data:
        parameters = dict(parameters, **potential_data)

    driver_value = raw.pop("driver", {})
    if not isinstance(driver_value, Mapping):
        raise ProviderConfigurationError("Legacy driver configuration must be a mapping.")
    driver = copy.deepcopy(dict(driver_value))
    executor_provider = driver.pop("backend", "external")
    potential_backend = parameters.get("backend", provider)
    if executor_provider == "external":
        executor_provider = potential_backend
    method = driver.pop("task", "min")
    controller = driver.get("controller")
    if method == "ts" and isinstance(controller, Mapping) and controller.get("name") in {"dimer", "dimer_ts"}:
        method = "dimer"

    scheduler_value = raw.pop("scheduler", None)
    scheduler = None
    if scheduler_value is not None:
        if not isinstance(scheduler_value, Mapping):
            raise ProviderConfigurationError("Legacy scheduler configuration must be a mapping.")
        scheduler_data = copy.deepcopy(dict(scheduler_value))
        scheduler_provider = scheduler_data.pop("backend", scheduler_data.pop("provider", "local"))
        scheduler = ComponentConfig(str(scheduler_provider).lower(), parameters=scheduler_data)

    return RuntimeConfig(
        potential=ComponentConfig(str(provider or ""), parameters=parameters),
        executor=ComponentConfig(str(executor_provider), method=str(method), parameters=driver),
        scheduler=scheduler,
        options=raw,
    )
