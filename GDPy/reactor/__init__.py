#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

""" This submodule is for exploring, sampling, 
    and performing (chemical) reactions with
    various advanced algorithms.
"""


def create_reaction_explorer(params_: dict):
    """"""
    params = copy.deepcopy(params_)
    method = params.pop("method", "afir")

    if method == "afir":
        from GDPy.reaction.afir import AFIRSearch
        rxn = AFIRSearch(**params)
    else:
        raise NotImplementedError(f"{method} is not implemented.")

    return rxn


if __name__ == "__main__":
    ...