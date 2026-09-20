from typing import Union

from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.configuration import scheduler_component
from gdpx.providers.specs import thaw

from .scheduler import BaseScheduler


def canonicalise_scheduler(config: Union[dict, BaseScheduler]) -> BaseScheduler:
    """Cannonicalise the scheduler based on the input configuration."""
    if isinstance(config, dict):
        value = config or {"provider": "direct", "parameters": {}}
        component = scheduler_component(value)
        providers = get_provider_manager()
        scheduler_factory = providers.require(
            component.provider,
            CapabilityKind.SCHEDULER,
            component.method or "default",
        )
        scheduler = scheduler_factory.create(thaw(component.parameters), providers=providers)
        if component.transport is not None:
            transport_factory = providers.require(
                component.transport.provider,
                CapabilityKind.TRANSPORT,
                component.transport.method or "default",
            )
            scheduler = transport_factory.create(
                thaw(component.transport.parameters),
                providers=providers,
                scheduler=scheduler,
            )
    else:
        if not isinstance(config, BaseScheduler):
            raise TypeError(f"Scheduler must be a mapping or BaseScheduler, got {type(config).__name__}.")
        scheduler = config

    return scheduler
