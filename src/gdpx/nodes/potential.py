#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.session.registry import workflow_registers as registers
from gdpx.session.variable import Variable
from gdpx.potential.utils import convert_input_to_potter


@registers.variable.register
class PotterVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        potter = convert_input_to_potter(kwargs)

        super().__init__(initial_value=potter, directory=directory)

        return


if __name__ == "__main__":
    ...
