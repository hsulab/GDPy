#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import registers
from gdpx.session.variable import Variable


@registers.variable.register
class PotterVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        name = kwargs.get("name", None)
        potter = registers.create(
            "manager",
            name,
            convert_name=False,
            # **kwargs.get("params", {})
        )
        potter.register_calculator(kwargs.get("params", {}))

        super().__init__(initial_value=potter, directory=directory)

        return


if __name__ == "__main__":
    ...
