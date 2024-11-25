#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ..core.register import registers
from ..core.variable import Variable


class RegionVariable(Variable):

    def __init__(self, directory="./", *args, **kwargs):
        """"""
        name = kwargs.pop("method", "auto")
        region = registers.create("region", name, convert_name=True, **kwargs)

        super().__init__(initial_value=region, directory=directory)

        return


if __name__ == "__main__":
    ...
  
