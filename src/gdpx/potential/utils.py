#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union

from ..calculators.mixer import CommitteeCalculator


def canonicalise_input_models(model: Union[str, list[str]]) -> list[str]:
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
        if not m.exists():
            raise FileNotFoundError(f"The model {str(m)} does not exist.")
        models.append(str(m))

    return models


def build_a_committee_calculator(
    calc_cls, params_list: list[dict], estimate_uncertainty: bool = False
):
    """Build a committee calculator.

    If there is one set of parameters or estimate_uncertainty is False, 
    then a single calculator is built.

    """
    num_calculators = len(params_list)
    assert num_calculators >= 1, "At least one calculator must be provided."
    use_committee = num_calculators > 1 and estimate_uncertainty
    if use_committee:
        calc = CommitteeCalculator(
            calcs=[calc_cls(**params) for params in params_list]
        )
    else:
        calc = calc_cls(**params_list[0])

    return calc


if __name__ == "__main__":
    ...
