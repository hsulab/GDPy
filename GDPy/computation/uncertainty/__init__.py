#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib
from typing import Callable

from GDPy.computation.uncertainty.committee import CommitteeUncertaintyEstimator

"""This module for uncertainty estimation.

Neural Network - Committee
Gaussian Process - Bayesian

"""


def create_estimator(est_params_: dict, calc_params_: dict, create_calc_func: Callable):
    """"""
    assert len(est_params_), "Only one estimator can be created each time."
    est_params = copy.deepcopy(est_params_)
    calc_params = copy.deepcopy(calc_params_)

    # NOTE: We only have committee now :(
    for name, cur_params in est_params.items():
        if name == "committee":
            models = cur_params.get("models", None) # model paths
            assert models, "We need models for committee uncertainty."
            calculators = []
            for m in models:
                m = str(pathlib.Path(m).resolve())
                cur_calc_params = copy.deepcopy(calc_params)
                cur_calc_params["model"] = m
                calculators.append(create_calc_func(cur_calc_params))
            estimator = CommitteeUncertaintyEstimator(calculators=calculators)
        else:
            raise NotImplementedError(f"{name} is not implemented.")

    return estimator


if __name__ == "__main__":
    pass