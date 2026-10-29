"""Population comparators shared by global-search methods."""
import copy
import inspect
from collections.abc import Mapping

from .basic import AtomsComparator, NNMatComparator
from .interatomic_distance import InteratomicDistanceComparator
from .mobile import MobileComparator

COMPARATORS = {
    "atoms": AtomsComparator,
    "interatomic_distance": InteratomicDistanceComparator,
    "nnmat": NNMatComparator,
}


def create_population_comparator(config, periodic=True, rng=None):
    if isinstance(config, (list, tuple)):
        return create_population_comparator({"method": "combined", "comparators": list(config)}, periodic, rng)
    if not isinstance(config, Mapping):
        if hasattr(config, "looks_like"):
            return config
        raise TypeError("system.comparator must be a mapping or comparator instance.")
    params = copy.deepcopy(dict(config))
    if "n_top" in params:
        raise ValueError("system.comparator.n_top is removed; use group_expression within tag != 0 atoms.")
    group_expression = params.pop("group_expression", None)
    if "name" in params:
        raise ValueError("system.comparator.name is not supported; use method.")
    if {"pbc", "mic", "rng"} & params.keys():
        raise ValueError("Population comparator periodicity and RNG are system-owned; use system.periodic.")
    method = params.pop("method", None)
    if method == "combined":
        if group_expression is not None:
            raise ValueError("Set group_expression on each structural child comparator.")
        children = params.get("comparators")
        if not isinstance(children, (list, tuple)) or not children:
            raise ValueError("combined requires a nonempty list of comparators.")
        params["comparators"] = [create_population_comparator(child, periodic, rng) for child in children]
    if method not in COMPARATORS:
        from gdpx.analysis.comparators import REGISTER
        if method not in REGISTER:
            raise ValueError(f"Unknown population comparator method {method!r}.")
        cls = REGISTER[method]
    else:
        cls = COMPARATORS[method]
    signature = inspect.signature(cls).parameters
    for key, value in dict(pbc=periodic, mic=periodic, rng=rng).items():
        if key in signature:
            params[key] = value
    comparator = cls(**params)
    if method in {"combined", "energy"}:
        if group_expression is not None:
            raise ValueError("Energy comparison uses the complete candidate energy, without group_expression.")
        return comparator
    return MobileComparator(comparator, group_expression, preserve_energy=method not in {
        "atoms", "nnmat", "oganov_fingerprint", "displacement",
    })
