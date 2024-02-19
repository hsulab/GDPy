#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NoReturn, List

import numpy as np
import jax.numpy as jnp

from ase import Atoms
from ase.calculators.calculator import Calculator


class AbstractBias(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    default_parameters = dict()

    def __init__(self, colvars: List[dict]=None, restart=None, label=None, atoms=None, directory=".", **kwargs):
        """"""
        super().__init__(restart=restart, label=label, atoms=atoms, directory=directory, **kwargs)

        # - check colvar
        colvars_ = []
        if isinstance(colvars, list):
            for colvar in colvars:
                colvars_.append(initiate_colvar(colvar))
        elif isinstance(colvars, dict):
            colvars_.append(initiate_colvar(colvars))
        else:
            ...
        self.colvars = colvars_

        # - NOTE: set bias function and parameters in subclass!
        ...

        return

    def calculate(self, atoms=None, properties=["energy"], system_changes=["positions"]):
        """"""
        super().calculate(atoms, properties, system_changes)

        positions = jnp.array(atoms.get_positions())
        bias_energy, bias_forces = 0., np.zeros(positions.shape)
        for colvar, bias_params in zip(self.colvars, self.bias_params):
            ret = self._compute_bias(
                positions, colvar, **bias_params
            )
            bias_energy += np.asarray(ret[0])
            bias_forces += -np.array(ret[1])
        self.results["energy"] = bias_energy
        self.results["forces"] = bias_forces

        return


if __name__ == "__main__":
    ...