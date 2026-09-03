"""Declarative provider components and resolved runtime workflow variables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from gdpx.execution import Runtime, resolve_runtime
from gdpx.providers import ComponentConfig, RuntimeConfig
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
    def __init__(self, provider, method, parameters=None, directory="./"):
        component = ComponentConfig(provider, method, parameters or {})
        super().__init__(component, directory=directory)

    def as_dict(self) -> dict:
        return self.value.to_dict()


@registers.variable.register
class RuntimeVariable(Variable):
    def __init__(
        self,
        potential,
        executor,
        modifiers=(),
        scheduler=None,
        options=None,
        schema_version=2,
        directory="./",
    ):
        potential_config = _component(potential, "potential")
        executor_config = _component(executor, "executor")
        modifier_configs = tuple(_component(item, "modifier") for item in modifiers)
        scheduler_config = None if scheduler is None else _component(scheduler, "scheduler")
        self.config = RuntimeConfig(
            potential=potential_config,
            executor=executor_config,
            modifiers=modifier_configs,
            scheduler=scheduler_config,
            options=options or {},
            schema_version=schema_version,
        )
        super().__init__(resolve_runtime(self.config), directory=directory)

    @classmethod
    def from_mapping(cls, value: Mapping, directory="./") -> "RuntimeVariable":
        config = RuntimeConfig.from_mapping(value)
        return cls(
            config.potential,
            config.executor,
            modifiers=config.modifiers,
            scheduler=config.scheduler,
            options=config.options,
            schema_version=config.schema_version,
            directory=directory,
        )

    def as_dict(self) -> dict:
        return self.config.to_dict()


@registers.variable.register
class RuntimeChainVariable(Variable):
    def __init__(self, runtimes: Sequence, directory="./"):
        resolved = []
        configs = []
        for value in runtimes:
            if isinstance(value, RuntimeVariable):
                resolved.append(value.value)
                configs.append(value.config)
            elif isinstance(value, Runtime):
                resolved.append(value)
                configs.append(value.config)
            elif isinstance(value, Mapping):
                variable = RuntimeVariable.from_mapping(value, directory=directory)
                resolved.append(variable.value)
                configs.append(variable.config)
            else:
                raise TypeError(f"Unsupported runtime chain item {type(value).__name__}.")
        if not resolved:
            raise ValueError("A runtime chain cannot be empty.")
        self.configs = tuple(configs)
        super().__init__(tuple(resolved), directory=directory)

    def as_dict(self) -> list[dict]:
        return [config.to_dict() for config in self.configs]
