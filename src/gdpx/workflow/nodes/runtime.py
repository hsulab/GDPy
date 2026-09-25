"""Declarative provider components and resolved runtime workflow variables."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

from gdpx.execution import Runtime, resolve_runtime
from gdpx.providers import SCHEMA_VERSION, ComponentConfig, RuntimeConfig, expand_runtime_configs
from gdpx.providers.configuration import scheduler_component
from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.workflow.session.variable import Variable


def _component(value, label: str) -> ComponentConfig:
    if isinstance(value, Variable):
        value = value.value
    if isinstance(value, ComponentConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a ComponentConfig or mapping.")
    return ComponentConfig(**dict(value))


@registers.variable.register
class PotentialVariable(Variable):
    def __init__(self, provider, method="default", parameters=None, directory="./"):
        component = ComponentConfig(provider, method, parameters or {})
        super().__init__(component, directory=directory)

    def as_dict(self) -> dict:
        return self.value.to_dict()


@registers.variable.register
class ExecutorVariable(Variable):
    def __init__(self, provider, method, parameters=None, broadcast=None, directory="./"):
        component = ComponentConfig(provider, method, parameters or {})
        self.broadcast = copy.deepcopy(broadcast)
        super().__init__(component, directory=directory)

    def as_dict(self) -> dict:
        data = self.value.to_dict()
        if self.broadcast is not None:
            data["broadcast"] = copy.deepcopy(self.broadcast)
        return data


@registers.variable.register
class RuntimeVariable(Variable):
    def __init__(
        self,
        potential,
        executor,
        modifiers=(),
        scheduler=None,
        dispatch=None,
        schema_version=SCHEMA_VERSION,
        directory="./",
    ):
        potential_config = _component(potential, "potential")
        executor_broadcast = None
        if isinstance(executor, ExecutorVariable):
            executor_broadcast = copy.deepcopy(executor.broadcast)
            executor_config = executor.value
        elif isinstance(executor, Mapping):
            executor_data = copy.deepcopy(dict(executor))
            executor_broadcast = executor_data.pop("broadcast", None)
            executor_config = _component(executor_data, "executor")
        else:
            executor_config = _component(executor, "executor")
        modifier_configs = tuple(_component(item, "modifier") for item in modifiers)
        if isinstance(scheduler, Variable):
            scheduler = scheduler.value
        scheduler_config = None if scheduler is None else scheduler_component(scheduler)
        base_config = RuntimeConfig(
            potential=potential_config,
            executor=executor_config,
            modifiers=modifier_configs,
            scheduler=scheduler_config,
            dispatch=dispatch or {},
            schema_version=schema_version,
        )
        source = base_config.to_dict()
        if executor_broadcast is not None:
            source["executor"]["broadcast"] = executor_broadcast
        self.configs = expand_runtime_configs(source)
        self.config = self.configs[0]
        resolved = tuple(resolve_runtime(config) for config in self.configs)
        value = resolved if executor_broadcast is not None else resolved[0]
        super().__init__(value, directory=directory)

    @classmethod
    def from_mapping(cls, value: Mapping, directory="./") -> RuntimeVariable:
        source = copy.deepcopy(dict(value))
        executor = source.get("executor")
        broadcast = None
        if isinstance(executor, Mapping):
            executor = copy.deepcopy(dict(executor))
            broadcast = executor.pop("broadcast", None)
            source["executor"] = executor
        config = RuntimeConfig.from_mapping(source)
        executor = config.executor.to_dict()
        if broadcast is not None:
            executor["broadcast"] = broadcast
        return cls(
            config.potential,
            executor,
            modifiers=config.modifiers,
            scheduler=config.scheduler,
            dispatch=config.dispatch,
            schema_version=config.schema_version,
            directory=directory,
        )

    def as_dict(self):
        if isinstance(self.value, Runtime):
            return self.config.to_dict()
        return [config.to_dict() for config in self.configs]


@registers.variable.register
class RuntimeChainVariable(Variable):
    def __init__(self, runtimes: Sequence, directory="./"):
        variants = []
        config_variants = []
        for value in runtimes:
            if isinstance(value, RuntimeVariable):
                values = value.value if isinstance(value.value, tuple) else (value.value,)
                variants.append(tuple(values))
                config_variants.append(tuple(value.configs))
            elif isinstance(value, Runtime):
                variants.append((value,))
                config_variants.append((value.config,))
            elif isinstance(value, Mapping):
                variable = RuntimeVariable.from_mapping(value, directory=directory)
                values = variable.value if isinstance(variable.value, tuple) else (variable.value,)
                variants.append(tuple(values))
                config_variants.append(tuple(variable.configs))
            else:
                raise TypeError(f"Unsupported runtime chain item {type(value).__name__}.")
        if not variants:
            raise ValueError("A runtime chain cannot be empty.")
        width = max(len(items) for items in variants)
        invalid = [
            (index, len(items))
            for index, items in enumerate(variants)
            if len(items) not in (1, width)
        ]
        if invalid:
            details = ", ".join(f"step {index}: {size}" for index, size in invalid)
            raise ValueError(
                f"Runtime chain broadcast widths must be 1 or {width}; got {details}."
            )
        chains = tuple(
            tuple(items[0] if len(items) == 1 else items[index] for items in variants)
            for index in range(width)
        )
        config_chains = tuple(
            tuple(items[0] if len(items) == 1 else items[index] for items in config_variants)
            for index in range(width)
        )
        self.configs = config_chains[0] if width == 1 else config_chains
        super().__init__(chains[0] if width == 1 else chains, directory=directory)

    def as_dict(self):
        if isinstance(self.value, tuple) and all(isinstance(item, Runtime) for item in self.value):
            return [config.to_dict() for config in self.configs]
        return [[config.to_dict() for config in chain] for chain in self.configs]
