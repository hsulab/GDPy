"""Construction of adaptive exploration strategies."""

import copy
from collections.abc import Iterable, Mapping

import numpy as np

from gdpx.structures.builders import canonicalise_builder

from . import REGISTER
from .expedition import BaseExpedition


RECIPE_METHODS = {
    "genetic_algorithm",
    "monte_carlo",
    "basin_hopping",
    "concurrent_hopping",
    "simulated_annealing",
}


def _recipe_parameters(method: str, parameters: dict) -> dict:
    """Return constructor parameters from the public recipe envelope."""
    if method not in RECIPE_METHODS:
        return parameters

    if "recipe" not in parameters:
        legacy_keys = ", ".join(sorted(parameters)) or "none"
        migration = (
            "Move random_seed and method-specific settings under 'recipe', and move the GA builder "
            "to 'recipe.population.random_generator'."
            if method == "genetic_algorithm"
            else "Move random_seed, builder, and all method-specific settings under 'recipe'."
        )
        raise ValueError(
            f"Exploration method {method!r} requires a 'recipe' mapping. "
            f"{migration} "
            f"(legacy top-level keys: {legacy_keys})."
        )
    if len(parameters) != 1:
        legacy_keys = ", ".join(sorted(key for key in parameters if key != "recipe"))
        raise ValueError(
            f"Exploration method {method!r} only accepts 'recipe' beside 'method'; "
            f"move these top-level keys into 'recipe': {legacy_keys}."
        )

    recipe = parameters["recipe"]
    if not isinstance(recipe, Mapping):
        raise TypeError(f"The recipe for exploration method {method!r} must be a mapping.")
    return copy.deepcopy(dict(recipe))


def create_expedition(config):
    if isinstance(config, BaseExpedition):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("Exploration configuration must be a mapping or BaseExpedition.")
    parameters = copy.deepcopy(dict(config))
    try:
        method = parameters.pop("method")
    except KeyError as error:
        raise ValueError("Exploration configuration requires a top-level 'method'.") from error
    parameters = _recipe_parameters(method, parameters)
    random_seed = parameters.get("random_seed")
    if random_seed is None:
        random_seed = int(np.random.randint(0, 1_000_000_000_000))
    parameters["random_seed"] = random_seed
    if parameters.get("builder") is not None:
        parameters["builder"] = canonicalise_builder(parameters["builder"])
        parameters["builder"].set_rng(seed=random_seed)
    expedition = REGISTER[method](**parameters)
    if isinstance(expedition, Iterable) and not isinstance(expedition, BaseExpedition):
        return list(expedition)
    return expedition
