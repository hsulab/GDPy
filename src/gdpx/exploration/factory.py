"""Construction of adaptive exploration strategies."""

import copy
import itertools
from collections.abc import Iterable, Mapping

import numpy as np

from gdpx.structures.builders import canonicalise_builder

from . import REGISTER
from .exploration import BaseExploration


RECIPE_METHODS = {
    "simulated_annealing",
}


def _recipe_parameters(method: str, parameters: dict) -> dict:
    """Return constructor parameters from the public recipe envelope."""
    if method not in RECIPE_METHODS:
        return parameters

    if "recipe" not in parameters:
        legacy_keys = ", ".join(sorted(parameters)) or "none"
        migration = "Move random_seed, builder, and all method-specific settings under 'recipe'."
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


def create_exploration(config):
    if isinstance(config, BaseExploration):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("Exploration configuration must be a mapping or BaseExploration.")
    parameters = copy.deepcopy(dict(config))
    try:
        method = parameters.pop("method")
    except KeyError as error:
        raise ValueError("Exploration configuration requires a top-level 'method'.") from error
    if method in {"genetic_algorithm", "basin_hopping", "concurrent_hopping"}:
        selected = "basin_hopping" if method == "concurrent_hopping" else method
        raise ValueError(f"Use method: global_optimisation with strategy.method: {selected}; "
                         "remove the recipe wrapper and move algorithm settings under strategy.")
    if method == "global_optimisation":
        from .population.exploration import reject_legacy_settings, validate_strategy
        reject_legacy_settings(parameters)
        validate_strategy(parameters.get("strategy"))
        if "population" not in parameters:
            raise ValueError("global_optimisation requires population settings.")
    if method == "monte_carlo" and "recipe" in parameters:
        raise ValueError(
            "monte_carlo no longer uses a recipe wrapper; move the builder and ensemble "
            "under system, operators under strategy, and convergence to the top level."
        )
    broadcast = parameters.pop('broadcast', None)
    if 'broadcast' in config:
        if method not in RECIPE_METHODS | {"global_optimisation", "monte_carlo"}:
            raise ValueError(
                f"Broadcast is only supported for global_optimisation, monte_carlo, "
                f"and recipe-based explorations, not {method!r}."
            )
        if not isinstance(broadcast, Mapping) or not broadcast:
            raise ValueError('broadcast must be a nonempty mapping of recipe paths to value lists.')
    parameters = _recipe_parameters(method, parameters)
    if broadcast is None:
        return _create_exploration(method, parameters)
    explorations = []
    for recipe in _broadcast_recipes(parameters, broadcast):
        result = _create_exploration(method, recipe)
        explorations.extend(result if isinstance(result, list) else [result])
    return explorations


def _broadcast_target(recipe, parts):
    """Resolve a recipe path; only a final mapping key may be absent."""
    parent = recipe
    for position, part in enumerate(parts):
        last = position == len(parts) - 1
        if isinstance(parent, Mapping):
            key = part
            if not last and key not in parent:
                raise ValueError(f'Unknown broadcast parent: {".".join(parts)}')
        elif isinstance(parent, list):
            if not part.isdecimal() or str(int(part)) != part or int(part) >= len(parent):
                raise ValueError(f'Invalid broadcast list index: {".".join(parts)}')
            key = int(part)
        else:
            raise ValueError(f'Broadcast path crosses a non-container: {".".join(parts)}')
        if last:
            return parent, key
        parent = parent[key]


def _broadcast_recipes(recipe, broadcast):
    paths, alternatives = [], []
    for path, values in broadcast.items():
        if not isinstance(path, str) or not path or any(not part for part in path.split('.')):
            raise ValueError(f'Invalid broadcast recipe path: {path!r}')
        parts = path.split('.')
        if any(parts[:len(other)] == other or other[:len(parts)] == parts for other in paths):
            raise ValueError(f'Overlapping broadcast recipe path: {path}')
        if not isinstance(values, list) or not values:
            raise ValueError(f'Broadcast values for {path} must be a nonempty list.')
        _broadcast_target(recipe, parts)
        paths.append(parts)
        alternatives.append(values)
    for combination in itertools.product(*alternatives):
        expanded = copy.deepcopy(recipe)
        for parts, value in zip(paths, combination):
            parent, key = _broadcast_target(expanded, parts)
            parent[key] = copy.deepcopy(value)
        yield expanded


def _create_exploration(method, parameters):
    """Construct one resolved recipe, preserving method-specific broadcasting."""
    random_seed = parameters.get("random_seed")
    if random_seed is None:
        random_seed = int(np.random.randint(0, 1_000_000_000_000))
    parameters["random_seed"] = random_seed
    if parameters.get("builder") is not None:
        parameters["builder"] = canonicalise_builder(parameters["builder"])
        parameters["builder"].set_rng(seed=random_seed)
    elif method == "monte_carlo" and isinstance(parameters.get("system"), Mapping):
        system = parameters["system"]
        if system.get("builder") is not None:
            system["builder"] = canonicalise_builder(system["builder"])
            system["builder"].set_rng(seed=random_seed)
    exploration = REGISTER[method](**parameters)
    if isinstance(exploration, Iterable) and not isinstance(exploration, BaseExploration):
        return list(exploration)
    return exploration
