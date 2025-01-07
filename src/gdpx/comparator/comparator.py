#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

from ase import Atoms

from gdpx.core.component import BaseComponent


class AbstractComparator(BaseComponent):

    def compare_composition(self, a1: Atoms, a2: Atoms) -> bool:
        """"""
        # TODO: compare PBC?
        is_similar = False
        na1, na2 = len(a1), len(a2)
        if na1 == na2:
            c1, c2 = a1.get_cell(complete=True), a2.get_cell(complete=True)
            if np.allclose(c1, c2):
                s1, s2 = a1.get_chemical_formula(), a2.get_chemical_formula()
                if s1 == s2:
                    is_similar = True
            else:
                ...
        else:
            ...

        return is_similar

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        """"""

        raise NotImplementedError()

    def __call__(self, a1, a2) -> bool:
        """"""

        return self.looks_like(a1, a2)


if __name__ == "__main__":
    ...

