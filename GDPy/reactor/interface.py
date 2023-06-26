#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List, Mapping

from ..core.register import registers
from ..core.variable import Variable
from ..core.operation import Operation
from ..data.array import AtomsArray2D


@registers.variable.register
class ReactorVariable(Variable):

    def __init__(self, potter, directory="./", *args, **kwargs):
        """"""
        # - save state by all nodes
        self.potter = self._load_potter(potter)

        # - create a reactor
        reactor = self.potter.create_reactor(kwargs)
        super().__init__(initial_value=reactor, directory=directory)

        return

    def _load_potter(self, inp):
        """"""
        potter = None
        if isinstance(inp, Variable):
            potter = inp.value
        elif isinstance(inp, dict):
            potter_params = copy.deepcopy(inp)
            name = potter_params.get("name", None)
            potter = registers.create(
                "manager", name, convert_name=True,
            )
            potter.register_calculator(potter_params.get("params", {}))
            potter.version = potter_params.get("version", "unknown")
        else:
            raise RuntimeError(f"Unknown {inp} for the potter.")

        return potter


@registers.operation.register
class react(Operation):

    def __init__(self, structures, reactor, directory="./") -> None:
        """"""
        super().__init__(input_nodes=[structures, reactor], directory=directory)

        return
    
    def forward(self, stru_dict: Mapping[str,AtomsArray2D], reactor):
        """"""
        super().forward()

        # - read input structures
        ini_atoms = stru_dict["IS"][0][-1]
        fin_atoms = stru_dict["FS"][0][-1]

        # - align structures
        #computers[0].directory = self.directory
        reactor.directory = self.directory
        reactor.run([ini_atoms, fin_atoms])

        return


if __name__ == "__main__":
    ...