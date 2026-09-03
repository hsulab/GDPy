"""Versioned provider runtime configuration."""

from __future__ import annotations

import copy
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
        if "backend" in self.parameters:
            raise ProviderConfigurationError(
                "Component parameter `backend` is not supported; select a provider and materialization target instead."
            )
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
        legacy_fields = sorted({"potter", "driver", "computer", "backend"}.intersection(raw))
        if version != SCHEMA_VERSION:
            suffix = f" Legacy fields found: {', '.join(legacy_fields)}." if legacy_fields else ""
            raise ProviderConfigurationError(
                f"Runtime configuration requires schema_version: {SCHEMA_VERSION}; got {version!r}."
                f"{suffix} Migrate to potential/executor component sections."
            )
        if legacy_fields:
            raise ProviderConfigurationError(
                f"Legacy runtime fields are not supported: {', '.join(legacy_fields)}; "
                "use potential/executor component sections."
            )
        return _parse_v2(raw)

    def to_dict(self) -> dict:
        data = {
            "schema_version": SCHEMA_VERSION,
            "potential": self.potential.to_dict(),
            "modifiers": [item.to_dict() for item in self.modifiers],
            "executor": self.executor.to_dict(),
        }
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.to_dict()
        data["options"] = thaw(self.options)
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
    options = raw.pop("options", {})
    if not isinstance(options, Mapping):
        raise ProviderConfigurationError("Runtime options must be a mapping.")
    allowed_options = {"batch_size", "worker", "share_workdir", "retain_info"}
    unknown_options = set(options) - allowed_options
    if unknown_options:
        raise ProviderConfigurationError(
            f"Unknown runtime options: {', '.join(sorted(unknown_options))}."
        )
    if raw:
        raise ProviderConfigurationError(f"Unknown runtime fields: {', '.join(sorted(raw))}.")
    return RuntimeConfig(potential, executor, modifiers, scheduler, options=options)
