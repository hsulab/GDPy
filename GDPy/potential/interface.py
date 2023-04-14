#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GDPy.core.variable import Variable
from GDPy.potential.register import PotentialRegister


class Potter(Variable):

    def __init__(self, **kwargs):
        """"""
        manager = PotentialRegister()

        name = kwargs.get("name", None)
        potter = manager.create_potential(pot_name=name)
        potter.register_calculator(kwargs.get("params", {}))
        potter.version = kwargs.get("version", "unknown")

        super().__init__(potter)

        return

if __name__ == "__main__":
    ...