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
    from gdpx.providers.trainer_registry import REGISTER as registry
    from gdpx.providers.training import BasePotentialTrainer

    existing, params = _params(config, BasePotentialTrainer)
    if existing is not None:
        return existing
    name = params.pop("name")
    class_name = "".join(part.capitalize() for part in name.split("_")) + "Trainer"
    return registry[class_name](**params)


def create_expedition(config):
    from gdpx.exploration.factory import create_expedition as create

    return create(config)
