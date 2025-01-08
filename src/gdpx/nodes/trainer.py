#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import registers
from gdpx.core.variable import Variable


@registers.variable.register
class TrainerVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        name = kwargs.pop("name", None)
        trainer = registers.create("trainer", name, convert_name=True, **kwargs)

        super().__init__(initial_value=trainer, directory=directory)

        return


if __name__ == "__main__":
    ...
  
