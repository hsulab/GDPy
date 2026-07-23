"""Session-independent dataset loader construction."""

import copy
from collections.abc import Mapping

from gdpx.dataloader import REGISTER as DATALOADER_REGISTER
from gdpx.dataloader.dataset import AbstractDataloader


def create_dataloader(config=None, **overrides):
    """Create a dataloader from a mapping, or return an existing loader."""
    if isinstance(config, AbstractDataloader):
        if overrides:
            raise TypeError("Overrides cannot be applied to an existing dataloader.")
        return config
    if config is None:
        params = {}
    elif isinstance(config, Mapping):
        params = copy.deepcopy(dict(config))
    else:
        raise TypeError(f"Dataloader must be a mapping, got {type(config).__name__}.")
    params.update(copy.deepcopy(overrides))
    name = params.pop("name", params.pop("type", None))
    if name is None:
        raise ValueError("Dataloader configuration requires `name` or `type`.")
    class_name = "".join(part.capitalize() for part in name.strip().split("_")) + "Dataloader"
    return DATALOADER_REGISTER[class_name](**params)
