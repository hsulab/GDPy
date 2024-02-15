#!/usr/bin/env python3
# -*- coding: utf-8 -*-


try:
    import jax
except Exception as e:
    ...

from ..core.register import registers

from .distance import compute_distance_bias
registers.colvar.register("distance")(compute_distance_bias)

from .position import position
registers.colvar.register("position")(position)


def initiate_colvar(params):
    """"""
    name = params.pop("name", )

    cvfunc = registers.colvar[name]
    cvfunc = jax.tree_util.Partial(cvfunc, **params)

    return cvfunc


if __name__ == "__main__":
    ...