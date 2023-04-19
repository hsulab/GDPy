#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import NoReturn, List

from ase import Atoms
from ase.io import read, write

from GDPy.core.variable import Variable
from GDPy.core.operation import Operation
from GDPy.core.register import registers


@registers.variable.register
class DriverVariable(Variable):

    def __init__(self, **kwargs):
        """"""
        initial_value = copy.deepcopy(kwargs)
        super().__init__(initial_value)

        return

if __name__ == "__main__":
    ...