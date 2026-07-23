#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
from typing import Any, Optional

from gdpx.builder import REGISTER as BUILDER_REGISTER
from gdpx.builder.builder import StructureBuilder
from gdpx.utils.parser import parse_input_file


def canonicalise_builder(config: Any) -> Optional[StructureBuilder]:
    """"""
    # Check if it is already a StructureBuilder, then return it directly,
    # which will keep its state, especially, the random state.
    if isinstance(config, StructureBuilder):
        return config

    # Check if it is a structure file path, a configuration file path or just a pure string
    supported_configtypes = [".json", ".yaml"]
    if isinstance(config, str):
        config = pathlib.Path(config)
        if config.suffix in supported_configtypes:
            config = parse_input_file(config)
        else:
            if config.exists():  # a structure filepath
                ...
            else:
                config = config.name

    # Convert everything into a dict
    if isinstance(config, str):
        raise NotImplementedError(f"Cannot convert `{config}` to builder.")
    elif isinstance(config, pathlib.Path):
        config = dict(method="direct", frames=str(config))
    elif isinstance(config, dict):
        ...
    elif isinstance(config, type(None)):
        ...
    else:
        raise Exception(
            f"Unknown config `{config}` with type `{type(config)}`."
        )

    if config is not None:
        config_to_use = copy.deepcopy(config)
        method = config_to_use.pop("method", "direct")
        builder = BUILDER_REGISTER[method](**config_to_use)
    else:
        builder = None

    return builder


if __name__ == "__main__":
    ...
