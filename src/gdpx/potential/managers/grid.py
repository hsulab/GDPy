#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import pathlib
import warnings

import numpy as np

from ..manager import AbstractPotentialManager


class GridCalculator():

    directory = "./"

    def __init__(self, dpath, unit="eV", *args, **kwargs) -> None:
        """"""
        data = np.loadtxt(dpath) # Need 3 columns, x, y and z.
        x, y, z = self._preprocess_data(data) # C-order 2D data

        self._x, self._y = x, y
        self._z = z / 96.485
        self._points = np.vstack([x.flatten(), y.flatten()]).T
        self._gradients = np.gradient(z, np.unique(data[:, 0]), np.unique(data[:, 1]))

        return
    
    def _preprocess_data(self, data):
        """"""
        nx = np.size(np.unique(data[:, 0]))
        ny = np.size(np.unique(data[:, 1]))

        # Some people give input where the first column varies fastest. 
        # That is Fortran ordering, and you are liable to get things confused 
        # if you don't take this into account.
        order = "C"
        if data[0, 0] != data[1, 0]:
            order = "F"
        x = data[:, 0].reshape(nx, ny, order=order)
        y = data[:, 1].reshape(nx, ny, order=order)
        z = data[:, 2].reshape(nx, ny, order=order)
        #print(f"x: {x}")
        #print(f"y: {y}")

        # - basic input sanity check
        xdiff = np.diff(x, axis=0)
        ydiff = np.diff(y, axis=1)
        if (not np.all(np.abs(xdiff - xdiff[0]) < 1e-8)) or (not np.all(np.abs(ydiff - ydiff[0]) < 1e-8)):
            warnings.warn(
                "WARNING! The input data is not coming from an equally spaced grid. imshow will be subtly wrong as a result, as each pixel is assumed to represent the same area.",
                UserWarning
            )
        
        return x, y, z
    
    def reset(self):
        """"""

        return
    
    def get_potential_energy(self):
        """"""

        return
    
    def get_forces(self):
        """Negative gradients."""

        return


class GridManager(AbstractPotentialManager):

    name = "grid"

    implemented_backends = ["grid"]
    valid_combinations = (
        ("grid", "grid"),
    )

    """"""

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__()

        return
    
    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        calc_params = copy.deepcopy(calc_params)

        dpath = calc_params.get("data", None) # Must exist!
        dpath = pathlib.Path(dpath).resolve()
        self.calc_params["data"] = str(dpath)

        self.calc = GridCalculator(dpath=dpath)

        return


if __name__ == "__main__":
    ...