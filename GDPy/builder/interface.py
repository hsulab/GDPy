#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NoReturn, List

from ase import Atoms
from ase.io import read, write

from GDPy.core.operation import Operation


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

class build(Operation):

    """Build structures without substrate structures.
    """

    def __init__(self, builder) -> NoReturn:
        super().__init__([builder])
    
    def forward(self, frames) -> List[Atoms]:
        """"""
        super().forward()

        write(self.directory/"output.xyz", frames)

        return frames

class modify(Operation):

    def __init__(self, modifier) -> NoReturn:
        super().__init__([modifier])


if __name__ == "__main__":
    ...