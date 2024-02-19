#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools
import pathlib

import numpy as np

import jax
import jax.numpy as jnp

from ase.calculators.calculator import Calculator

from . import registers


@jax.jit
def gaussian_bias(x, x_t, sigma, omega):
    """Compute bias potential.

    Args:
        x: (1, num_dim)
        x_t: (num_bias, num_dim)
        sigma: (1, num_dim)
        omega: scalar

    """
    x2 = (x - x_t)**2/2./sigma**2
    v = omega*jnp.exp(-jnp.sum(x2, axis=1))

    return v.sum(axis=0)


@functools.partial(jax.jit, static_argnums=(1,))
def compute_bias(positions, cvfunc, cvparams, ref_colvars, sigma, omega):
    """"""
    colvar = cvfunc(positions, cvparams)

    bias = gaussian_bias(colvar, ref_colvars, sigma, omega)

    return bias, colvar


class GaussianCalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    default_parameters = dict(
        width = 0.1,
        height = 0.2,
    )

    _history = []

    _fname = "HILLS"

    def __init__(self, colvar, restart=None, label=None, atoms=None, directory=".", **kwargs):
        """"""
        super().__init__(restart=restart, label=label, atoms=atoms, directory=directory, **kwargs)

        self._colvar = colvar
        print(self._colvar)

        _width = np.array(self.parameters["width"])
        if len(_width.shape) == 0:
            _width = _width.reshape(-1)
        self._width = _width
        self._height = np.array(self.parameters["height"])

        return
    
    def _save_history(self, ):
        """"""

        return
    
    def calculate(self, atoms=None, properties=["energy"], system_changes=["positions","numbers","cell"]):
        """"""
        super().calculate(atoms, properties, system_changes)

        sigma, omega = self._width, self._height
        
        positions = atoms.positions
        cvfunc, cvparams = self._colvar.cvfunc, self._colvar.params

        if False: # Combined function...
            num_his = len(self._history)
            if num_his == 0:
                curr_colvar = cvfunc(positions, cvparams)
                natoms = len(atoms)
                e, f = omega, np.zeros((natoms, 3))
            else:
                ref_colvars = np.vstack(self._history)
                (e, curr_colvar), f = jax.value_and_grad(compute_bias, argnums=0, has_aux=True)(
                    positions, cvfunc, cvparams, ref_colvars, sigma, omega
                )
            self._history.append(curr_colvar)
        else:
            curr_colvar = cvfunc(atoms, cvparams)
            print(f"colvar: {curr_colvar}")
            ...

        # ---
        fpath = pathlib.Path(self.directory)/self._fname
        fmode = "w"
        if fpath.exists():
            fmode = "a"
        
        with open(fpath, fmode) as fopen:
            content = f"{curr_colvar[0][0]:<8.4f}  {sigma[0]:<8.4f}  {omega:<8.4f}\n"
            fopen.write(content)

        self.results["energy"] = np.asarray(e)
        self.results["forces"] = np.array(f)

        return


if __name__ == "__main__":
    ...