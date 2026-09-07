"""Compose provider capabilities into executable runtimes."""

import importlib
from typing import Any, Mapping, Union

from gdpx.providers.capabilities import CapabilityKind
from gdpx.providers.configuration import RuntimeConfig
from gdpx.providers.errors import MaterializationError, MissingCapabilityError
from gdpx.providers.specs import Materialization, thaw
from gdpx.providers.targets import AseCalculatorMaterialization

from .runtime import Runtime


class RuntimeResolver:
    def __init__(self, providers: Any) -> None:
        self.providers = providers

    def resolve(self, value: Union[RuntimeConfig, Mapping[str, Any]]) -> Runtime:
        config = value if isinstance(value, RuntimeConfig) else RuntimeConfig.from_mapping(value)
        potential_factory = self.providers.require(
            config.potential.provider,
            CapabilityKind.POTENTIAL,
            config.potential.method or "default",
        )
        potential = potential_factory.create(thaw(config.potential.parameters))
        executor_factory = self.providers.require(
            config.executor.provider, CapabilityKind.EXECUTOR, config.executor.method
        )
        target = getattr(executor_factory, "target", None)
        modifier_instances = ()
        if target is None:
            materialization = Materialization(
                target=f"{config.executor.provider}.legacy",
                payload=getattr(potential, "calc", potential),
            )
            executor = executor_factory.create(thaw(config.executor.parameters), potential=potential)
        else:
            try:
                materializer = self.providers.require(
                    config.potential.provider, CapabilityKind.MATERIALIZER, target
                )
            except MissingCapabilityError as error:
                raise MaterializationError(
                    f"Potential {config.potential.provider!r}/{config.potential.method or 'default'} "
                    f"cannot materialize target {target!r} required by executor "
                    f"{config.executor.provider!r}/{config.executor.method!r}."
                ) from error
            try:
                materialization = materializer.materialize(potential, target)
            except MaterializationError:
                raise
            except Exception as error:
                raise MaterializationError(
                    f"Failed to materialize {config.potential.provider!r} for target {target!r}: {error}"
                ) from error
            modifier_instances = self._create_modifiers(config)
            materialization = self._apply_modifiers(materialization, target, modifier_instances)
            executor = executor_factory.create(
                thaw(config.executor.parameters),
                potential=potential,
                materialization=materialization,
            )
        scheduler_config = config.scheduler
        if scheduler_config is None:
            from gdpx.execution.schedulers.local import LocalScheduler

            scheduler = LocalScheduler()
        else:
            scheduler_provider = scheduler_config.provider
            scheduler_method = scheduler_config.method or "default"
            scheduler_parameters = thaw(scheduler_config.parameters)
            scheduler_factory = self.providers.require(
                scheduler_provider, CapabilityKind.SCHEDULER, scheduler_method
            )
            scheduler = scheduler_factory.create(scheduler_parameters, providers=self.providers)
        return Runtime(
            potential=config.potential_spec(),
            materialization=materialization,
            executor=executor,
            modifiers=config.modifier_specs(),
            config=config,
            provider_potential=potential,
            modifier_instances=modifier_instances if target is not None else (),
            scheduler=scheduler,
        )

    def _create_modifiers(self, config):
        instances = []
        for component in config.modifiers:
            factory = self.providers.require(
                component.provider, CapabilityKind.MODIFIER, component.method
            )
            instances.append(factory.create(thaw(component.parameters)))
        return tuple(instances)

    @staticmethod
    def _apply_modifiers(materialization, target, modifiers):
        if not modifiers:
            return materialization
        if target != "ase.calculator" or not isinstance(materialization, AseCalculatorMaterialization):
            raise MaterializationError(
                f"Modifiers are not supported by materialization target {target!r}; "
                "select an ASE executor or install a target-specific modifier provider."
            )
        from gdpx.providers.ase.backend import EnhancedCalculator

        calculator = EnhancedCalculator([materialization.calculator, *modifiers])
        return AseCalculatorMaterialization(calculator, materialization.artifacts)


def resolve_runtime(value, providers=None) -> Runtime:
    if providers is None:
        provider_module = importlib.import_module("gdpx.providers.manager")
        providers = provider_module.get_provider_manager()
    return RuntimeResolver(providers).resolve(value)
