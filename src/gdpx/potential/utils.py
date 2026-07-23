#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
from typing import Any, Optional, Union

import omegaconf

from gdpx.backend.ase import CommitteeCalculator
from gdpx.utils.parser import parse_input_file

from .manager import BasePotentialManager


def canonicalise_input_models(model: Union[str, list[str]], must_exist: bool = True) -> list[str]:
    """Convert input models to a list of resolved path strings.

    Args:
        model: The input model(s).

    Returns:
        The resolved path strings. An error is raised if the model does not exist.

    """
    if not isinstance(model, list):
        assert isinstance(model, str)
        model_ = [model]
    else:
        model_ = model

    models = []
    for m in model_:
        m = pathlib.Path(m).resolve()
        if must_exist and not m.exists():
            raise FileNotFoundError(f"The model {str(m)} does not exist.")
        models.append(str(m))

    return models


def build_a_committee_calculator(calc_cls, params_list: list[dict], estimate_uncertainty: bool = False):
    """Build a committee calculator.

    If there is one set of parameters or estimate_uncertainty is False,
    then a single calculator is built.

    """
    num_calculators = len(params_list)
    assert num_calculators >= 1, "At least one calculator must be provided."
    use_committee = num_calculators > 1 and estimate_uncertainty
    if use_committee:
        calc = CommitteeCalculator(calcs=[calc_cls(**params) for params in params_list])
    else:
        calc = calc_cls(**params_list[0])

    return calc


def canonicalise_plumed_for_lammps(params: dict) -> dict:
    """"""
    inp = params.get("inp", "./plumed.inp")
    if isinstance(inp, str) or isinstance(inp, pathlib.Path):
        inp = pathlib.Path(inp)
        if inp.exists():  # read input file and clean up comments and empty lines
            input_lines = []
            with open(inp, "r") as fopen:
                lines = fopen.readlines()
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "#" in line:
                            line = line[: line.index("#")]
                        else:
                            line = line
                        input_lines.append(line + "\n")
            params.update(inp=input_lines)
        else:
            raise FileNotFoundError(f"{inp} does not exist.")
    elif isinstance(inp, list):
        input_lines = inp
    else:
        raise Exception(f"Plumed input {inp} {type(inp)} is invalid.")

    new_params = dict(
        inp=input_lines,
    )

    return new_params


def potter_from_dict(inp_dict: dict) -> "BasePotentialManager":
    """"""
    name = inp_dict.get("name", None)
    if name is None:
        raise Exception(f"The input dictionary `{inp_dict}` does not define a valid potter.")
    from gdpx.potential import REGISTER as manager_registry

    potter = manager_registry[name]()
    potter.register_calculator(inp_dict.get("params", {}))
    potter.version = inp_dict.get("version", "unknown")

    return potter


def convert_input_to_potter(inp: Any) -> Optional["BasePotentialManager"]:
    """Convert an input to a potter and adjust its behaviour."""
    potter = None
    if isinstance(inp, BasePotentialManager):
        potter = inp
    elif isinstance(inp, dict) or isinstance(inp, omegaconf.dictconfig.DictConfig):
        # DictConfig must be cast to dict as sometimes it cannot be overwritten.
        if isinstance(inp, omegaconf.dictconfig.DictConfig):
            inp = omegaconf.OmegaConf.to_object(inp)
        assert isinstance(inp, dict)
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
