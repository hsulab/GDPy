#!/usr/bin/env python3
# -*- coding: utf-8 -*


import dataclasses
import pathlib
from typing import Union

import numpy as np
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt
try:
    plt.style.use("presentation")
except:
    ...

from .string import AbstractStringReactor, StringReactorSetting


def interpolate_string(a, b, t, npts: int=5, concat: bool=False):
    """Inteerpolate a 2D string."""
    # - use transition state
    if concat:
        npts_1 = int(npts/2.)
        npts_2 = npts - npts_1
        pts = np.vstack(
            [
                np.hstack([np.linspace(a[0], t[0], npts_1), np.linspace(t[0], b[0], npts_2)]),
                np.hstack([np.linspace(a[1], t[1], npts_1), np.linspace(t[1], b[1], npts_2)])
            ]
        ).T
    else:
        pts = np.vstack(
            [np.linspace(a[0], b[0], npts), np.linspace(a[1], b[1], npts)]
        ).T

    return pts

def optimise_string(pts, gridpoints, z, gradx, grady, steps=100, stepsize=0.01):
    """"""
    npts = pts.shape[0]
    for i in range(steps):
        # - find gradient interpolation
        Dx = griddata(gridpoints, gradx.flatten(), (pts[:, 0], pts[:, 1]), method="linear")
        Dy = griddata(gridpoints, grady.flatten(), (pts[:, 0], pts[:, 1]), method="linear")
        h = np.amax(np.sqrt(np.square(Dx) + np.square(Dy)))
        
        # - evolve
        pts -= stepsize * np.vstack([Dx, Dy]).T / h

        # - reparameterize
        arclength = np.hstack([0, np.cumsum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))])
        arclength /= arclength[-1]
        pts = np.vstack(
            [
                interp1d(arclength, pts[:, 0])(np.linspace(0, 1, npts)), 
                interp1d(arclength, pts[:, 1])(np.linspace(0, 1, npts))
            ]
        ).T

        # - save history?
        if i % 10 == 0:
            print(i, np.sum(griddata(gridpoints, z.flatten(), (pts[:, 0], pts[:, 1]), method="linear")))

    return pts


@dataclasses.dataclass
class ZeroStringReactorSetting(StringReactorSetting):

    backend: str = "grid"

    def __post_init__(self):
        """"""

        return
    
    def get_run_params(self, *args, **kwargs):
        """"""

        return


class ZeroStringReactor(AbstractStringReactor):

    name: str = "zero"

    def __init__(self, calc=None, params={}, ignore_convergence=False, directory="./", *args, **kwargs) -> None:
        """"""
        self.calc = calc
        if self.calc is not None:
            self.calc.reset()

        self.ignore_convergence = ignore_convergence

        self.directory = directory
        self.cache_nebtraj = self.directory/self.traj_name

        # - parse params
        self.setting = ZeroStringReactorSetting(**params)
        self._debug(self.setting)

        return
    
    def run(self, structures, read_cache: bool=True, *args, **kwargs):
        """"""
        self.directory.mkdir(parents=True, exist_ok=True)

        # - get points
        start_points = [a.get_positions() for a in structures]
        start_points = np.array(start_points).reshape(-1, 3)

        a, b = start_points
        nimages = self.setting.nimages
        images = np.vstack(
            [np.linspace(a[0], b[0], nimages), np.linspace(a[1], b[1], nimages)]
        ).T

        images = optimise_string(
            images, self.calc._points, self.calc._z, 
            self.calc._gradients[0], self.calc._gradients[1],
            steps=100, stepsize=0.01
        )
        energies = griddata(
            self.calc._points, self.calc._z.flatten(), 
            (images[:, 0], images[:, 1]), method="linear"
        )[:, np.newaxis]

        np.savetxt(
            self.directory/"pmf.dat", np.hstack([np.linspace(0, 1, nimages)[:, np.newaxis], images, energies]), 
            fmt="%8.4f"
        )

        # - vis
        # --
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(
            np.linspace(0, 1, nimages), energies, marker="o"
        )
        ax.set_ylabel("Free Energy [eV]")
        ax.set_xlabel("Reaction Coordinate")
        fig.savefig(self.directory/"pmf.png")

        # --
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.set_xlabel("CV 1")
        ax.set_ylabel("CV 2")
        #ax.set_xlim(0., 0.33)
        #ax.set_ylim(0., 0.50)
        x, y, z = self.calc._x, self.calc._y, self.calc._z
        cntr = ax.contourf(
            x, y, z, levels=10,
            #x, np.where(y < 0., y+1.0, y), z, levels=10,
            #cmap="RdBu"
        )
        fig.colorbar(cntr, ax=ax, label="Free Energy [eV]")
        cn = ax.contour(
            x, y, z, #cmap="RdBu"
            #x, np.where(y < 0., y, y+1.0), z, levels=10,
        )
        plt.clabel(cn, inline=True, fontsize=10)

        ax.plot(
            images[:, 0], images[:,1], linestyle="--"
        )
        fig.savefig(self.directory/"fes.png", transparent=False)

        return
    
    def read_convergence(self, *args, **kwargs) -> bool:
        """"""
        converged = super().read_convergence(*args, **kwargs)

        return


if __name__ == "__main__":
    ...
