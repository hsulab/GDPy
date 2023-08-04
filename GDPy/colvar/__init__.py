#!/usr/bin/env python3
# -*- coding: utf-8 -*-


try:
    import jax
except Exception as e:
    ...

from ..core.register import Register
colvar_register = Register("colvar")

from .position import position
colvar_register.register("position")(position)


def initiate_colvar(params):
    """"""
    name = params.pop("name", )

    cvfunc = colvar_register[name]
    cvfunc = jax.tree_util.Partial(cvfunc, **params)

    return cvfunc


if __name__ == "__main__":
    ...