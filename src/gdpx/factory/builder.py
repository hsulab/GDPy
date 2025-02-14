#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Any, Optional

from gdpx.builder.builder import StructureBuilder
from gdpx.core.register import registers
from gdpx.utils.command import parse_input_file


def canonicalise_builder(config: Any) -> Optional[StructureBuilder]:
    """"""
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
        method = config.pop("method", "direct")
        builder = registers.create(
            "builder", method, convert_name=False, **config
        )
    else:
        builder = None

    return builder


if __name__ == "__main__":
    ...
