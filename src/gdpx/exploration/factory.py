"""Construction of adaptive exploration strategies."""

import copy
from collections.abc import Iterable, Mapping

import numpy as np

from gdpx.structures.builders import canonicalise_builder

from . import REGISTER
from .expedition import BaseExpedition


def create_expedition(config):
    if isinstance(config, BaseExpedition):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("Exploration configuration must be a mapping or BaseExpedition.")
    parameters = copy.deepcopy(dict(config))
    method = parameters.pop("method")
    random_seed = parameters.get("random_seed")
    if random_seed is None:
        random_seed = int(np.random.randint(0, 1_000_000_000_000))
    if parameters.get("builder") is not None:
        parameters["builder"] = canonicalise_builder(parameters["builder"])
        parameters["builder"].set_rng(seed=random_seed)
    expedition = REGISTER[method](**parameters)
    if isinstance(expedition, Iterable) and not isinstance(expedition, BaseExpedition):
        return list(expedition)
    return expedition
