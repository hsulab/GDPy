#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List


"""Create bias on potential energy surface.
"""


def create_single_bias(params: dict):
    """"""
    # NOTE: only have afir now...
    method = params.pop("method", None)
    if method == "afir":
        from GDPy.computation.bias.afir import AFIRBias as Bias
    else:
        ...
    bias = Bias(**params)

    return bias

def create_bias_list(params_list: List[dict]):
    """"""
    bias_list = []
    for params in params_list:
        bias = create_single_bias(params)
        bias_list.append(bias)

    return bias_list


if __name__ == "__main__":
    ...