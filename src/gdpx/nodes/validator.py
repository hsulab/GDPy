#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import registers
from gdpx.core.variable import Variable


@registers.variable.register
class ValidatorVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        # Instantiate a validator
        method = kwargs.pop("method", "minima")
        validator = registers.create("validator", method, convert_name=False, **kwargs)

        # Save the instance
        super().__init__(initial_value=validator, directory=directory)

        return


if __name__ == "__main__":
    ...
  
