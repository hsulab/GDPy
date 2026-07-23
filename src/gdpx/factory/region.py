"""Session-independent region construction."""

import copy
from collections.abc import Mapping

from gdpx.region import REGISTER as REGION_REGISTER
from gdpx.region.region import BaseRegion


def create_region(config=None, **overrides) -> BaseRegion:
    """Create a region from a mapping, or return an existing region."""
    if isinstance(config, BaseRegion):
        if overrides:
            raise TypeError("Overrides cannot be applied to an existing region.")
        return config
    if config is None:
        params = {}
    elif isinstance(config, Mapping):
        params = copy.deepcopy(dict(config))
    else:
        raise TypeError(f"Region must be a mapping or BaseRegion, got {type(config).__name__}.")
    params.update(copy.deepcopy(overrides))
    method = params.pop("method", "auto")
    class_name = "".join(part.capitalize() for part in method.strip().split("_")) + "Region"
    return REGION_REGISTER[class_name](**params)
