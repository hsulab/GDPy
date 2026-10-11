"""Versioned provider runtime configuration."""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Tuple

from .errors import ProviderConfigurationError
from .specs import ModifierSpec, PotentialSpec, freeze, thaw

SCHEMA_VERSION = 4
EXECUTOR_PARAMETER_SECTIONS = frozenset({"setup", "output", "stop"})
EXECUTOR_PARAMETER_GLOBALS = frozenset({"random_seed"})


def resolve_executor_parameters(parameters: Mapping[str, Any], method: Optional[str] = None) -> dict[str, Any]:
    """Translate structured public executor parameters for existing providers."""
    data = copy.deepcopy(thaw(parameters))
    sections = EXECUTOR_PARAMETER_SECTIONS.intersection(data)
    if not sections:
        return data
    extra = set(data) - EXECUTOR_PARAMETER_SECTIONS - EXECUTOR_PARAMETER_GLOBALS
    if extra:
        raise ProviderConfigurationError(
            "Do not mix flat and structured executor parameters; move "
            f"{', '.join(sorted(extra))} under setup, output, or stop."
        )
    for name in EXECUTOR_PARAMETER_SECTIONS:
        if name in data and not isinstance(data[name], Mapping):
            raise ProviderConfigurationError(f"Executor parameters.{name} must be a mapping.")

    setup = copy.deepcopy(dict(data.get("setup", {})))
    output = copy.deepcopy(dict(data.get("output", {})))
    stop = copy.deepcopy(dict(data.get("stop", {})))
    resolved: dict[str, Any] = {}

    def merge(values: Mapping[str, Any], source: str) -> None:
        duplicate = set(resolved).intersection(values)
        if duplicate:
            raise ProviderConfigurationError(
                f"Duplicate executor parameter in {source}: {', '.join(sorted(duplicate))}."
            )
        resolved.update(copy.deepcopy(dict(values)))

    merge({name: data[name] for name in EXECUTOR_PARAMETER_GLOBALS if name in data}, "parameters")

    velocities = setup.pop("velocities", None)
    regulator = setup.pop("regulator", None)
    optimizer = setup.pop("optimizer", None)
    if regulator is not None and optimizer is not None:
        raise ProviderConfigurationError("Executor setup cannot define both regulator and optimizer.")
    merge(setup, "setup")

    if velocities is not None:
        if not isinstance(velocities, Mapping):
            raise ProviderConfigurationError("Executor setup.velocities must be a mapping.")
        velocities = copy.deepcopy(dict(velocities))
        initialize = velocities.pop("initialize", "if_missing")
        if initialize not in ("if_missing", "always"):
            raise ProviderConfigurationError(
                "Executor setup.velocities.initialize must be 'if_missing' or 'always'."
            )
        translated = {"ignore_atoms_velocities": initialize == "always"}
        for public, internal in {
            "seed": "velocity_seed",
            "remove_translation": "remove_translation",
            "remove_rotation": "remove_rotation",
        }.items():
            if public in velocities:
                translated[internal] = velocities.pop(public)
        if "temperature" in velocities:
            translated["temp"] = velocities.pop("temperature")
        if velocities:
            raise ProviderConfigurationError(
                f"Unknown executor velocity fields: {', '.join(sorted(velocities))}."
            )
        merge(translated, "setup.velocities")

    choice = regulator if regulator is not None else optimizer
    if choice is not None:
        label = "regulator" if regulator is not None else "optimizer"
        if not isinstance(choice, Mapping):
            raise ProviderConfigurationError(f"Executor setup.{label} must be a mapping.")
        choice = copy.deepcopy(dict(choice))
        name = choice.pop("name", None)
        parameters_ = choice.pop("parameters", {})
        targets = choice.pop("targets", {}) if label == "regulator" else {}
        if not isinstance(name, str) or not name:
            raise ProviderConfigurationError(f"Executor setup.{label}.name must be a nonempty string.")
        if not isinstance(parameters_, Mapping) or not isinstance(targets, Mapping):
            raise ProviderConfigurationError(f"Executor setup.{label} parameters and targets must be mappings.")
        if choice:
            raise ProviderConfigurationError(
                f"Unknown executor {label} fields: {', '.join(sorted(choice))}."
            )
        merge({"controller": {"name": name, "params": copy.deepcopy(dict(parameters_))}}, f"setup.{label}")
        if label == "regulator":
            unknown_targets = set(targets) - {"temperature", "pressure"}
            if unknown_targets:
                raise ProviderConfigurationError(
                    f"Unknown regulator targets: {', '.join(sorted(unknown_targets))}."
                )
            translated_targets = {}
            if "temperature" in targets:
                translated_targets["temp"] = targets["temperature"]
            if "pressure" in targets:
                translated_targets["press"] = targets["pressure"]
            merge(translated_targets, "setup.regulator.targets")

    unknown_output = set(output) - {"trajectory", "checkpoint"}
    if unknown_output:
        raise ProviderConfigurationError(f"Unknown executor output fields: {', '.join(sorted(unknown_output))}.")
    trajectory = output.get("trajectory", {})
    checkpoint = output.get("checkpoint", {})
    if not isinstance(trajectory, Mapping) or set(trajectory) - {"period"}:
        raise ProviderConfigurationError("Executor output.trajectory accepts only period.")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) - {"period", "keep"}:
        raise ProviderConfigurationError("Executor output.checkpoint accepts only period and keep.")
    translated_output = {}
    if "period" in trajectory:
        translated_output["dump_period"] = trajectory["period"]
    if "period" in checkpoint:
        translated_output["ckpt_period"] = checkpoint["period"]
    if "keep" in checkpoint:
        translated_output["ckpt_number"] = checkpoint["keep"]
    merge(translated_output, "output")
    merge(stop, "stop")

    if method == "md":
        ensemble = resolved.get("ensemble", "nve")
        has_regulator = regulator is not None
        has_temperature = "temp" in resolved and has_regulator
        has_pressure = "press" in resolved and has_regulator
        if ensemble == "nve" and has_regulator:
            raise ProviderConfigurationError("NVE executor setup must not define a regulator.")
        if ensemble == "nvt" and (not has_temperature or has_pressure):
            raise ProviderConfigurationError("NVT executor setup requires only a temperature regulator target.")
        if ensemble == "npt" and (not has_temperature or not has_pressure):
            raise ProviderConfigurationError("NPT executor setup requires temperature and pressure regulator targets.")
    return resolved


def _broadcast_target(parameters: Any, parts: tuple[str, ...], label: str):
    """Resolve a parameter path, allowing only a missing final mapping key."""
    parent = parameters
    for position, part in enumerate(parts):
        last = position == len(parts) - 1
        if isinstance(parent, Mapping):
            key: Any = part
            if not last and key not in parent:
                raise ProviderConfigurationError(f"Unknown {label} broadcast parent: {'.'.join(parts)}.")
        elif isinstance(parent, list):
            if not part.isdecimal() or str(int(part)) != part or int(part) >= len(parent):
                raise ProviderConfigurationError(f"Invalid {label} broadcast list index: {'.'.join(parts)}.")
            key = int(part)
        else:
            raise ProviderConfigurationError(
                f"{label.capitalize()} broadcast path crosses a non-container: {'.'.join(parts)}."
            )
        if last:
            return parent, key
        parent = parent[key]


def _prepare_parameter_broadcast(
    component: Mapping[str, Any], label: str
) -> tuple[dict[str, Any], list[tuple[tuple[str, ...], list[Any]]]]:
    """Remove and validate one component-local parameter broadcast."""
    component = copy.deepcopy(dict(component))
    marker = object()
    broadcast = component.pop("broadcast", marker)
    if broadcast is marker or broadcast is None:
        return component, []
    if not isinstance(broadcast, Mapping) or not broadcast:
        raise ProviderConfigurationError(
            f"{label.capitalize()} broadcast must be a nonempty mapping of parameter paths to value lists."
        )

    parameters = component.get("parameters")
    if parameters is None:
        parameters = {}
        component["parameters"] = parameters
    elif not isinstance(parameters, Mapping):
        raise ProviderConfigurationError(f"{label.capitalize()} parameters must be a mapping.")
    else:
        parameters = copy.deepcopy(dict(parameters))
        component["parameters"] = parameters

    paths: list[tuple[str, ...]] = []
    dimensions: list[tuple[tuple[str, ...], list[Any]]] = []
    for path, values in broadcast.items():
        if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
            raise ProviderConfigurationError(f"Invalid {label} broadcast parameter path: {path!r}.")
        parts = tuple(path.split("."))
        if any(parts[: len(other)] == other or other[: len(parts)] == parts for other in paths):
            raise ProviderConfigurationError(f"Overlapping {label} broadcast parameter path: {path}.")
        if not isinstance(values, list) or not values:
            raise ProviderConfigurationError(
                f"{label.capitalize()} broadcast values for {path} must be a nonempty list."
            )
        _broadcast_target(parameters, parts, label)
        paths.append(parts)
        dimensions.append((parts, values))
    return component, dimensions


def _expand_runtime_mapping(value: Mapping[str, Any]) -> tuple[RuntimeConfig, ...]:
    """Expand one runtime mapping's component-local parameter broadcasts."""
    raw = copy.deepcopy(dict(value))
    dimensions: list[tuple[tuple[str, ...], list[Any], str]] = []

    executor = raw.get("executor")
    if isinstance(executor, Mapping):
        executor, executor_dimensions = _prepare_parameter_broadcast(executor, "executor")
        raw["executor"] = executor
        dimensions.extend(
            (("executor", "parameters", *parts), values, "executor") for parts, values in executor_dimensions
        )

    modifiers = raw.get("modifiers")
    if isinstance(modifiers, (list, tuple)):
        clean_modifiers = []
        for index, modifier in enumerate(modifiers):
            if not isinstance(modifier, Mapping):
                clean_modifiers.append(modifier)
                continue
            label = f"modifier {index}"
            modifier, modifier_dimensions = _prepare_parameter_broadcast(modifier, label)
            clean_modifiers.append(modifier)
            dimensions.extend(
                (("modifiers", str(index), "parameters", *parts), values, label)
                for parts, values in modifier_dimensions
            )
        raw["modifiers"] = clean_modifiers

    if not dimensions:
        return (RuntimeConfig.from_mapping(raw),)

    configs = []
    for combination in itertools.product(*(values for _, values, _ in dimensions)):
        expanded = copy.deepcopy(raw)
        for (parts, _, label), item in zip(dimensions, combination):
            parent, key = _broadcast_target(expanded, parts, label)
            parent[key] = copy.deepcopy(item)
        configs.append(RuntimeConfig.from_mapping(expanded))
    return tuple(configs)


def expand_runtime_configs(value: Any) -> tuple[RuntimeConfig, ...]:
    """Resolve a runtime mapping or flat list, expanding component broadcasts."""
    if isinstance(value, Mapping):
        return _expand_runtime_mapping(value)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ProviderConfigurationError("At least one runtime configuration is required.")
        configs = []
        for item in value:
            if not isinstance(item, Mapping):
                raise ProviderConfigurationError(
                    "Every runtime configuration must be a mapping."
                )
            configs.extend(_expand_runtime_mapping(item))
        return tuple(configs)
    raise ProviderConfigurationError(
        f"Runtime configuration must be a mapping or flat list, got {type(value).__name__}."
    )


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
                "Component parameter `backend` is not supported; use potential.backend or modifiers[].backend instead."
            )
        object.__setattr__(self, "parameters", freeze(self.parameters))

    def to_dict(self) -> dict:
        data = {"provider": self.provider}
        if self.method is not None:
            data["method"] = self.method
        data["parameters"] = thaw(self.parameters)
        return data


@dataclass(frozen=True)
class PotentialConfig(ComponentConfig):
    backend: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.backend is not None and (not isinstance(self.backend, str) or not self.backend):
            raise ProviderConfigurationError("Potential backend must be a nonempty string.")
        if self.provider in ("vasp", "cp2k") and "interface" in self.parameters:
            raise ProviderConfigurationError(
                "Move parameters.interface to potential.backend; "
                "use interactive for shell/interactive interfaces."
            )
        if self.provider == "vasp" and "dispersion" in self.parameters:
            raise ProviderConfigurationError("Move VASP parameters.dispersion to a dftd3 modifier with backend: ase.")
        if (self.provider, self.backend) in (("cp2k", "cp2k_shell"), ("vasp", "vasp_interactive")):
            raise ProviderConfigurationError(
                f"Backend {self.backend!r} was renamed to interactive; use potential.backend: interactive."
            )
        if self.provider == "vasp" and self.backend == "vasp_interactive_disp":
            raise ProviderConfigurationError("Use potential.backend: interactive and a dftd3 modifier with backend: ase.")

    def to_dict(self):
        data = super().to_dict()
        if self.backend is not None:
            data["backend"] = self.backend
        return data


@dataclass(frozen=True)
class ModifierConfig(ComponentConfig):
    backend: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.backend is not None and (not isinstance(self.backend, str) or not self.backend):
            raise ProviderConfigurationError("Modifier backend must be a nonempty string.")

    def to_dict(self):
        data = super().to_dict()
        if self.backend is not None:
            data["backend"] = self.backend
        return data


@dataclass(frozen=True)
class SchedulerConfig(ComponentConfig):
    """A submission backend and the transport used to reach its host."""

    transport: Optional[ComponentConfig] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.provider == "local":
            raise ProviderConfigurationError(
                "Scheduler provider `local` was removed in schema 3; use `direct` "
                "with `transport.provider: local`."
            )
        if self.provider == "remote":
            raise ProviderConfigurationError(
                "Scheduler provider `remote` was removed in schema 3; select the actual "
                "scheduler and add `transport.provider: ssh`."
            )
        if self.transport is not None and not isinstance(self.transport, ComponentConfig):
            object.__setattr__(self, "transport", _component(self.transport, "transport"))

    def to_dict(self) -> dict:
        data = super().to_dict()
        if self.transport is not None:
            data["transport"] = self.transport.to_dict()
        return data


@dataclass(frozen=True)
class DispatchConfig:
    """Worker orchestration policy for one resolved runtime."""

    worker: Literal["batch", "single"] = "batch"
    batch_size: int = 1
    share_workdir: bool = False
    retain_info: bool = False
    # Preserve old job fingerprints when reading manifests written before
    # batching compatible variants became automatic. This does not affect policy.
    _legacy_group_variants: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.worker not in ("batch", "single"):
            raise ProviderConfigurationError(
                f"Unknown dispatch worker {self.worker!r}; expected 'batch' or 'single'."
            )
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ProviderConfigurationError(
                f"Dispatch batch_size must be a positive integer; got {self.batch_size!r}."
            )
        for name in ("share_workdir", "retain_info", "_legacy_group_variants"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ProviderConfigurationError(
                    f"Dispatch {name} must be a boolean; got {value!r}."
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DispatchConfig:
        if not isinstance(value, Mapping):
            raise ProviderConfigurationError("Dispatch configuration must be a mapping.")
        data = copy.deepcopy(dict(value))
        allowed = {"worker", "batch_size", "share_workdir", "retain_info", "group_variants"}
        unknown = set(data) - allowed
        if unknown:
            raise ProviderConfigurationError(
                f"Unknown dispatch fields: {', '.join(sorted(unknown))}."
            )
        if "group_variants" in data:
            legacy = data.pop("group_variants")
            if not isinstance(legacy, bool):
                raise ProviderConfigurationError("Historical dispatch group_variants must be a boolean.")
            data["_legacy_group_variants"] = legacy
        return cls(**data)

    def to_dict(self) -> dict:
        data = {
            "worker": self.worker,
            "batch_size": self.batch_size,
            "share_workdir": self.share_workdir,
            "retain_info": self.retain_info,
        }
        if self._legacy_group_variants:
            data["group_variants"] = True
        return data


@dataclass(frozen=True)
class RuntimeConfig:
    potential: PotentialConfig
    executor: ComponentConfig
    modifiers: Tuple[ModifierConfig, ...] = ()
    scheduler: Optional[SchedulerConfig] = None
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProviderConfigurationError(
                f"Unsupported runtime schema {self.schema_version}; expected {SCHEMA_VERSION}."
            )
        if not isinstance(self.potential, PotentialConfig):
            object.__setattr__(self, "potential", _component(self.potential.to_dict(), "potential"))
        object.__setattr__(self, "modifiers", tuple(
            item if isinstance(item, ModifierConfig) else _component(item.to_dict(), "modifier")
            for item in self.modifiers
        ))
        if not isinstance(self.dispatch, DispatchConfig):
            object.__setattr__(self, "dispatch", DispatchConfig.from_mapping(self.dispatch))
        if self.scheduler is not None:
            object.__setattr__(self, "scheduler", scheduler_component(self.scheduler))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        raw = copy.deepcopy(dict(value))
        version = raw.pop("schema_version", SCHEMA_VERSION)
        legacy_fields = sorted({"potter", "driver", "computer", "backend"}.intersection(raw))
        if version == 3:
            raise ProviderConfigurationError(
                "Runtime schema 3 is unsupported; replace top-level `options` with "
                "`dispatch` and set `schema_version: 4`."
            )
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
        return _parse_v4(raw)

    def to_dict(self) -> dict:
        data = {
            "schema_version": SCHEMA_VERSION,
            "potential": self.potential.to_dict(),
            "modifiers": [item.to_dict() for item in self.modifiers],
            "executor": self.executor.to_dict(),
        }
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.to_dict()
        data["dispatch"] = self.dispatch.to_dict()
        return data

    def potential_spec(self) -> PotentialSpec:
        return PotentialSpec(
            self.potential.provider,
            self.potential.parameters,
            method=self.potential.method or "default",
            backend=self.potential.backend,
        )

    def modifier_specs(self) -> Tuple[ModifierSpec, ...]:
        return tuple(
            ModifierSpec(item.provider, item.method or "default", item.parameters, backend=item.backend) for item in self.modifiers
        )


def _component(value: Any, label: str, *, require_method: bool = False) -> ComponentConfig:
    if not isinstance(value, Mapping):
        raise ProviderConfigurationError(f"{label.capitalize()} configuration must be a mapping.")
    data = copy.deepcopy(dict(value))
    provider = data.pop("provider", None)
    method = data.pop("method", None)
    parameters = data.pop("parameters", {})
    backend = data.pop("backend", None) if label in ("potential", "modifier") else None
    if data:
        raise ProviderConfigurationError(f"Unknown {label} fields: {', '.join(sorted(data))}.")
    if not isinstance(parameters, Mapping):
        raise ProviderConfigurationError(f"{label.capitalize()} parameters must be a mapping.")
    if require_method and not method:
        raise ProviderConfigurationError(f"{label.capitalize()} method is required.")
    if label == "executor":
        resolve_executor_parameters(parameters, method)
    if label == "potential":
        return PotentialConfig(str(provider or ""), method, parameters, backend)
    if label == "modifier":
        return ModifierConfig(str(provider or ""), method, parameters, backend)
    return ComponentConfig(str(provider or ""), method, parameters)


def scheduler_component(value: Any) -> SchedulerConfig:
    """Parse a scheduler component with its optional nested transport."""
    if isinstance(value, SchedulerConfig):
        return value
    if isinstance(value, ComponentConfig):
        value = value.to_dict()
    from gdpx.user_config import UserConfigError, resolve_scheduler_preset
    try:
        value = resolve_scheduler_preset(value)
    except UserConfigError as error:
        raise ProviderConfigurationError(str(error)) from error
    if not isinstance(value, Mapping):
        raise ProviderConfigurationError("Scheduler configuration must be a mapping.")
    data = copy.deepcopy(dict(value))
    transport_value = data.pop("transport", None)
    component = _component(data, "scheduler")
    if component.provider == "local":
        raise ProviderConfigurationError(
            "Scheduler provider `local` was removed in schema 3; use `direct` "
            "and select `transport.provider: local` when it must run on this machine."
        )
    if component.provider == "remote":
        raise ProviderConfigurationError(
            "Scheduler provider `remote` was removed in schema 3; select the actual "
            "scheduler and add `transport.provider: ssh`."
        )
    transport = None if transport_value is None else _component(transport_value, "transport")
    return SchedulerConfig(
        component.provider,
        component.method,
        component.parameters,
        transport,
    )


def _parse_v4(raw: dict) -> RuntimeConfig:
    try:
        potential = _component(raw.pop("potential"), "potential")
        executor = _component(raw.pop("executor"), "executor", require_method=True)
    except KeyError as error:
        raise ProviderConfigurationError(f"Missing runtime section {error.args[0]!r}.") from error
    modifiers_value = raw.pop("modifiers", ())
    if not isinstance(modifiers_value, (list, tuple)):
        raise ProviderConfigurationError("Modifiers must be a sequence.")
    modifiers = tuple(_component(item, "modifier") for item in modifiers_value)
    scheduler_value = raw.pop("scheduler", None)
    scheduler = None if scheduler_value is None else scheduler_component(scheduler_value)
    if "options" in raw:
        raise ProviderConfigurationError(
            "Runtime field `options` was removed in schema 4; rename it to `dispatch`."
        )
    dispatch_value = raw.pop("dispatch", {})
    dispatch = DispatchConfig.from_mapping(dispatch_value)
    if raw:
        raise ProviderConfigurationError(f"Unknown runtime fields: {', '.join(sorted(raw))}.")
    return RuntimeConfig(potential, executor, modifiers, scheduler, dispatch=dispatch)
