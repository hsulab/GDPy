"""Population comparators shared by global-search methods."""
import copy
import inspect
from collections.abc import Mapping

from .basic import AtomsComparator, NNMatComparator
from .interatomic_distance import InteratomicDistanceComparator
from .ofp import OFPComparator

COMPARATORS = {
    "atoms": AtomsComparator,
    "interatomic_distance": InteratomicDistanceComparator,
    "nnmat": NNMatComparator,
    "ofp": OFPComparator,
}


def create_population_comparator(config, periodic=True, rng=None):
    if not isinstance(config, Mapping):
        if hasattr(config, "looks_like"):
            return config
        raise TypeError("system.comparator must be a mapping or comparator instance.")
    params = copy.deepcopy(dict(config))
    if "name" in params:
        raise ValueError("system.comparator.name is not supported; use method.")
    if {"pbc", "mic", "rng"} & params.keys():
        raise ValueError("Population comparator periodicity and RNG are system-owned; use system.periodic.")
    method = params.pop("method", None)
    if method not in COMPARATORS:
        from gdpx.analysis.comparators import REGISTER
        if method not in REGISTER:
            raise ValueError(f"Unknown population comparator method {method!r}.")
        cls = REGISTER[method]
    else:
        cls = COMPARATORS[method]
    signature = inspect.signature(cls.__init__).parameters
    for key, value in dict(pbc=periodic, mic=periodic, rng=rng).items():
        if key in signature:
            params[key] = value
    return cls(**params)
