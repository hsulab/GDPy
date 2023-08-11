#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Callable

import numpy as np
import jax.numpy as jnp

from ase.calculators.morse import MorsePotential
from ase.neighborlist import neighbor_list

from .bias import AbstractBias



def morse(positioins, cvfunc: Callable):
    """"""

    return


class NucleiBias(AbstractBias):

    implemented_properties = ["energy", "free_energy", "forces"]

    default_parameters = dict()

    def __init__(self, colvar: dict=None, restart=None, label=None, atoms=None, directory=".", **kwargs):
        """"""
        super().__init__(colvar=colvar, restart=restart, label=label, atoms=atoms, directory=directory, **kwargs)

        return

    def calculate(self, atoms=None, properties=["energy"], system_changes=["positions"]):
        """"""
        super().calculate(atoms, properties, system_changes)

        positions = jnp.array(atoms.get_positions())
        ret = self._compute_bias(
            positions, self.colvar,
            **self._compute_params
        )
        self.results["energy"] = np.asarray(ret[0])
        self.results["forces"] = -np.array(ret[1]) # copy forces

        return


if __name__ == "__main__":
    ...