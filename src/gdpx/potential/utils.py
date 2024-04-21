#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib

from typing import Callable, Optional, List

import omegaconf

from ..core.variable import Variable
from ..core.register import registers
from ..utils.command import parse_input_file
from .manager import AbstractPotentialManager


def potter_from_dict(inp_dict: dict):
    """"""
    name = inp_dict.get("name", None)
    potter = registers.create(
        "manager",
        name,
        convert_name=True,
    )
    potter.register_calculator(inp_dict.get("params", {}))
    potter.version = inp_dict.get("version", "unknown")

    return potter

def convert_input_to_potter(inp) -> "AbstractPotentialManager":
    """Convert an input to a potter and adjust its behaviour."""
    potter = None
    if isinstance(inp, AbstractPotentialManager):
        potter = inp
    elif isinstance(inp, Variable):
        potter = inp.value
    elif isinstance(inp, dict) or isinstance(inp, omegaconf.dictconfig.DictConfig):
        # DictConfig must be cast to dict as sometimes cannot be overwritten.
        if isinstance(inp, omegaconf.dictconfig.DictConfig):
            inp = omegaconf.OmegaConf.to_object(inp)
        potter_params = copy.deepcopy(inp)
        potter = potter_from_dict(potter_params)
    elif isinstance(inp, str) or isinstance(inp, pathlib.Path):
        if pathlib.Path(inp).exists():
            potter_params = parse_input_file(input_fpath=inp)
            potter = potter_from_dict(potter_params)
        else:
            raise RuntimeError(f"The potter configuration `{inp}` does not exist.")
    else:
        raise RuntimeError(f"Unknown {inp} of type {type(inp)} for the potter.")

    return potter


if __name__ == "__main__":
    ...
