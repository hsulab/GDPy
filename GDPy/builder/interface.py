#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from typing import NoReturn, List

from ase import Atoms
from ase.io import read, write

from GDPy.core.variable import Variable
from GDPy.core.operation import Operation
from GDPy.core.register import registers


def create_modifier(method: str, params: dict):
    """"""
    # TODO: check if params are valid
    if method == "perturb":
        from GDPy.builder.perturb import perturb as op_cls
    elif method == "insert_adsorbate_graph":
        from GDPy.builder.graph import insert_adsorbate_graph as op_cls
    else:
        raise NotImplementedError(f"Unimplemented modifier {method}.")

    return op_cls

@registers.variable.register
class BuilderVariable(Variable):

    """Read structures from file."""

    def __init__(self, filename, index=":", format=None):
        """"""
        initial_value = read(filename, index, format)

        super().__init__(initial_value)

        return

@registers.operation.register
class build(Operation):

    """Build structures without substrate structures.
    """

    def __init__(self, *builders) -> NoReturn:
        super().__init__(builders)
    
    def forward(self, *args, **kwargs) -> List[Atoms]:
        """"""
        super().forward()

        for i, frames in enumerate(args):
            write(self.directory/f"output-{i}.xyz", frames)
            self.pfunc(f"b{i} nframes: {len(frames)}")
        
        bundle = list(itertools.chain(*args))
        self.pfunc(f"nframes: {len(bundle)}")

        return bundle

@registers.operation.register
class modify(Operation):

    def __init__(self, substrate, modifier) -> NoReturn:
        super().__init__([substrate, modifier])
    
    def forward(self):
        """"""
        super().forward()

        return


if __name__ == "__main__":
    ...