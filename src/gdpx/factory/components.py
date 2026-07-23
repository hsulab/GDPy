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
    from gdpx.selector import REGISTER as registry
    from gdpx.selector.composition import ComposedSelector
    from gdpx.selector.selector import BaseSelector

    if isinstance(config, Mapping) and "selection" in config:
        config = config["selection"]
    if isinstance(config, list):
        selectors = [create_selector(item) for item in copy.deepcopy(config)]
        if not selectors:
            raise ValueError("At least one selector definition is required.")
        return selectors[0] if len(selectors) == 1 else ComposedSelector(selectors)
    existing, params = _params(config, BaseSelector)
    if existing is not None:
        return existing
    method = params.pop("method", "random")
    return registry[method](**params)


def create_comparator(config):
    from gdpx.comparator import REGISTER as registry
    from gdpx.comparator.comparator import BaseComparator

    existing, params = _params(config, BaseComparator)
    if existing is not None:
        return existing
    method = params.pop("method")
    return registry[method](**params)


def create_describer(config):
    from gdpx.describer import REGISTER as registry
    from gdpx.describer.describer import BaseDescriber

    existing, params = _params(config, BaseDescriber)
    if existing is not None:
        return existing
    name = params.pop("name", "soap")
    return registry[name](**params)


def create_trainer(config):
    from gdpx.trainer import REGISTER as registry
    from gdpx.trainer.trainer import BasePotentialTrainer

    existing, params = _params(config, BasePotentialTrainer)
    if existing is not None:
        return existing
    name = params.pop("name")
    class_name = "".join(part.capitalize() for part in name.split("_")) + "Trainer"
    return registry[class_name](**params)


def create_expedition(config):
    from gdpx.expedition import REGISTER as registry
    from gdpx.expedition.expedition import BaseExpedition
    from gdpx.factory.builder import canonicalise_builder

    existing, params = _params(config, BaseExpedition)
    if existing is not None:
        return existing
    method = params.pop("method")
    random_seed = params.get("random_seed")
    if random_seed is None:
        random_seed = int(np.random.randint(0, 1_000_000_000_000))
    if "builder" in params and params["builder"] is not None:
        params["builder"] = canonicalise_builder(params["builder"])
        params["builder"].set_rng(seed=random_seed)
    expedition = registry[method](**params)
    if isinstance(expedition, Iterable) and not isinstance(expedition, BaseExpedition):
        return list(expedition)
    return expedition
