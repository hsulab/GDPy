#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..core.register import registers
from ..core.variable import Variable


@registers.variable.register
class DescriberVariable(Variable):

    def __init__(self, directory="./", *args, **kwargs):
        """"""
        method = kwargs.pop("method", "soap")
        describer = registers.create("describer", method, convert_name=False, **kwargs)

        super().__init__(initial_value=describer, directory=directory)

        return


if __name__ == "__main__":
    ...