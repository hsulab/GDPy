#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NoReturn

from ase import Atoms


class PESModifier():

    """Base class of any PES modifier.

    This object modifies the PES based on the state of structure.

    """

    def __init__(self) -> NoReturn:
        """"""
        ...

        return
    
    def compute(self, atoms: Atoms):
        """"""

        return


if __name__ == "__main__":
    ...