"""Typed factories for configurable domain components."""

import copy
from collections.abc import Iterable, Mapping

import numpy as np

def _params(config, expected_type):
    if isinstance(config, expected_type):
        return config, None
    if not isinstance(config, Mapping):
        raise TypeError(f"Configuration must be a mapping or {expected_type.__name__}.")
    return None, copy.deepcopy(dict(config))


def create_selector(config):
    from gdpx.analysis.selectors import create_selector as create

    return create(config)


def create_comparator(config):
    from gdpx.analysis.comparators import create_comparator as create

    return create(config)


def create_describer(config):
    from gdpx.analysis.descriptors import create_describer as create

    return create(config)


def create_trainer(config):
    from gdpx.providers import ComponentConfig, get_provider_manager

    component = config if isinstance(config, ComponentConfig) else ComponentConfig(**copy.deepcopy(dict(config)))
    trainer = get_provider_manager().create_training(component)

    # Retain a CLI-compatible component specification on the constructed
    # trainer.  Use effective settings so provider defaults survive when a
    # workflow worker writes the standalone ``gdp train`` input.
    parameters = trainer.as_dict()
    parameters.pop("name", None)
    trainer.component_config = ComponentConfig(
        provider=component.provider,
        method=component.method or "default",
        parameters=parameters,
    )
    return trainer


def create_exploration(config):
    from gdpx.exploration.factory import create_exploration as create

    return create(config)
