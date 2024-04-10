#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
from typing import List, Callable

import numpy as np

import jax
import jax.numpy as jnp

from ase.calculators.calculator import Calculator, all_changes


@jax.jit
def harmonic(positions, cvfunc: Callable, k: float, s: float):
    """Harmonic bias potential."""
    colvars = cvfunc(positions) # NOTE: Must be 1D CV.
    bias = k*jnp.square(colvars - s)

    return jnp.sum(bias)

@jax.jit
def upper_harmonic(positions, cvfunc: Callable, k: float, s: float):
    """Harmonic bias potential."""
    colvars = cvfunc(positions) # NOTE: Must be 1D CV.
    colvars = jnp.where(colvars > s, colvars, s)

    bias = k*jnp.square(colvars - s)

    return jnp.sum(bias)

@jax.jit
def lower_harmonic(positions, cvfunc: Callable, k: float, s: float):
    """Harmonic bias potential."""
    colvars = cvfunc(positions) # NOTE: Must be 1D CV.
    colvars = jnp.where(colvars < s, colvars, s)

    bias = k*jnp.square(colvars - s)

    return jnp.sum(bias)



@dataclasses.dataclass
class HarmonicSetting:

    #: Spring constant, unit of eV.
    k: float = 100.

    #: Function origin, unit of colvar.
    s: float = 0.

    #: Whether construct a upper wall.
    upper: bool = False

    #: Whether construct a lower wall.
    lower: bool = False


class HarmonicBias(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    default_parameters = dict()

    """A harmonic bias.

    E_bias = k*(s_i-s_0)**2

    """

    def __init__(self, colvar: dict=None, restart=None, label=None, atoms=None, directory=".", **kwargs):
        """"""
        super().__init__(restart=restart, label=label, atoms=atoms, directory=directory, **kwargs)

        # - check bias params
        self._setting = HarmonicSetting(**self.parameters)
        assert not (self._setting.upper and self._setting.lower)

        bias_func = harmonic
        if self._setting.upper:
            bias_func = upper_harmonic
        if self._setting.lower:
            bias_func = lower_harmonic
        self.compute_harmonic = jax.value_and_grad(bias_func, argnums=0)

        # - check colvar
        self.colvar = initiate_colvar(colvar)

        return
    
    def calculate(self, atoms=None, properties=["energy"], system_changes=["positions"]):
        """"""
        super().calculate(atoms, properties, system_changes)

        positions = jnp.array(atoms.get_positions())
        ret = self.compute_harmonic(
            positions, self.colvar,
            k=self._setting.k, s=self._setting.s, 
        )
        self.results["energy"] = np.asarray(ret[0])
        self.results["forces"] = -np.array(ret[1]) # copy forces

        return


if __name__ == "__main__":
    ...
