#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union

from gdpx.providers.ase.backend import CommitteeCalculator


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


if __name__ == "__main__":
    ...
