#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def create_modifier(method: str, params: dict):
    """"""
    # TODO: check if params are valid
    if method == "perturb":
        from GDPy.builder.perturb import perturb as op_cls
    elif method == "insert_adsorbate_graph":
        from GDPy.builder.graph import insert_adsorbate_graph as op_cls
    else:
        raise NotImplementedError(f"Unimplemented modifier {method}.")

    return op_cls

if __name__ == "__main__":
    ...