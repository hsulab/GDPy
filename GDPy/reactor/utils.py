#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from ase.formula import Formula


def convert_index_to_formula(atoms, group_indices: List[List[int]]):
    """"""
    formulae = []
    for g in group_indices:
        symbols = [atoms[i].symbol for i in g]
        formulae.append(
            Formula.from_list(symbols).format("hill")
        )
    #formulae = sorted(formulae)

    return formulae


if __name__ == "__main__":
    ...