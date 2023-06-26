#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import Union

import numpy as np

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    print("Used default matplotlib style.")

from ..utils.command import CustomTimer
from .validator import AbstractValidator
from .utils import wrap_traj
from ..builder.group import create_a_group

def plot_msd(wdir, names, lagtimes, timeseries, prefix=""):
    """"""
    # - get self-diffusivity
    from scipy.stats import linregress

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    fig.suptitle("MSD")

    #ax.set_title("$51Å\times 44Å$")

    for name, x, y in zip(names, lagtimes, timeseries):

        start_step, end_step = 20, 60
        linear_model = linregress(
            x[start_step:end_step], y[start_step:end_step]
        )

        slope = linear_model.slope
        error = linear_model.rvalue
        # dim_fac is 3 as we computed a 3D msd with 'xyz'
        D = slope * 1/(2*3)
        print(f"{name} D: ", D)

        #ax.plot(x, y, label=f"{name} K $D={D:>.2e}$")
        ax.plot(x, y, label=f"{name}")

        ax.text(np.median(x), np.median(y), f"$D={D:>.2e}$")


    ax.set_ylabel("MSD [Å^2]")
    ax.set_xlabel("Time [ps]")

    ax.legend()

    plt.savefig(wdir/ f"{prefix}msd.png")

    return

class DiffusionCoefficientValidator(AbstractValidator):

    """Estimate the diffusion coefficient.

    Compute a windowed MSD where the MSD is averaged over all possible lag-times
    tau < tau_max.

    """

    def __init__(self, group, timeintv: float, lagmax: int, start: int=None, end: int=None, directory: str | pathlib.Path = "./", *args, **kwargs):
        """"""
        super().__init__(directory, *args, **kwargs)

        self.start = start
        self.end = end

        self.lagmax = lagmax
        self.timeintv = timeintv

        self.group = group

        return
    
    def run(self, dataset: dict, worker=None, *args, **kwargs):
        """"""
        super().run()

        self._print("process reference ->")
        reference = dataset.get("reference")
        if reference is not None:
            mdtraj = reference[0]._images
            group_indices = create_a_group(mdtraj[0], self.group)

            cache_msd = self.directory/"ref-msd.npy"
            if not cache_msd.exists():
                data = self._compute_msd(
                    mdtraj, group_indices, lagmax=self.lagmax, 
                    start=self.start, end=self.end, timeintv=self.timeintv, 
                    prefix="ref-"
                )
            else:
                data = np.load(cache_msd)

            lagtimes = [x[0] for x in [data]]
            timeseries = [x[1] for x in [data]]
            plot_msd(self.directory, names=["MSD"], lagtimes=lagtimes, timeseries=timeseries, prefix="ref-")

        self._print("process prediction ->")
        prediction = dataset.get("prediction")
        if prediction is not None:
            mdtraj = prediction[0]._images
            group_indices = create_a_group(mdtraj[0], self.group)
            self._debug(f"group_indices: {group_indices}")

            cache_msd = self.directory/"pre-msd.npy"
            if not cache_msd.exists():
                data = self._compute_msd(
                    mdtraj, group_indices, lagmax=self.lagmax, 
                    start=self.start, end=self.end, timeintv=self.timeintv, 
                    prefix="pre-"
                )
            else:
                data = np.load(cache_msd)

            lagtimes = [x[0] for x in [data]]
            timeseries = [x[1] for x in [data]]
            plot_msd(self.directory, names=["MSD"], lagtimes=lagtimes, timeseries=timeseries, prefix="pre-")

        return
    
    def _compute_msd(
            self, frames, group_indices, lagmax, start, end, 
            timeintv: float, prefix=""
        ):
        """Compute MSD ...

        Args:
            group_func: Function used to get a group of atoms.

        """
        # -
        frames = wrap_traj(frames)
        frames = frames[start:end:]
        #print("nframes: ", len(frames))

        positions = []
        for atoms in frames:
            positions.append(atoms.get_positions()[group_indices,:])
        positions = np.array(positions)

        nframes, natoms, _ = positions.shape
        #print("shape: ", positions.shape)

        # -
        msds_by_particle = np.zeros((lagmax, natoms))

        lagtimes = np.arange(1, lagmax)
        for lag in lagtimes:
            disp = positions[:-lag, :, :] - positions[lag:, :, :]
            #print("disp: ", disp.shape)
            sqdist = np.square(disp).sum(axis=-1)
            msds_by_particle[lag, :] = np.mean(sqdist, axis=0)
            timeseries = msds_by_particle.mean(axis=1)

        timeintv = timeintv / 1000. # fs to ps
        lagtimes = np.arange(lagmax)*timeintv

        return lagtimes, timeseries


if __name__ == "__main__":
    ...