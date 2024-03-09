#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import Union, List

import numpy as np
from joblib import Parallel, delayed

from ase import Atoms

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    #print("Used default matplotlib style.")
    ...

from ..utils.command import CustomTimer
from .validator import AbstractValidator
from .utils import wrap_traj
from ..builder.group import create_a_group
from ..data.array import AtomsNDArray


def plot_msd(wdir, names, lagtimes, timeseries, start_step=20, end_step=60, prefix=""):
    """"""
    # - get self-diffusivity
    from scipy.stats import linregress

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    fig.suptitle("MSD")

    #ax.set_title("$51Å\times 44Å$")
    if names is None:
        names = [str(i) for i in range(len(lagtimes))]

    for name, x, y in zip(names, lagtimes, timeseries):

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


def compute_msd(
        frames, group_indices, lagmax, start, end, 
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


class DiffusionCoefficientValidator(AbstractValidator):

    """Estimate the diffusion coefficient.

    Compute a windowed MSD where the MSD is averaged over all possible lag-times
    tau < tau_max.

    """

    def __init__(
            self, group, timeintv: float, lagmax: int, start: int=None, end: int=None, 
            d_start: int = 0, d_end: int = 20,
            directory: Union[str, pathlib.Path] = "./", *args, **kwargs
        ):
        """"""
        super().__init__(directory, *args, **kwargs)

        self.start = start
        self.end = end
        
        # - diffusion coefficient linear fitting
        self.d_start = d_start
        self.d_end = d_end

        self.lagmax = lagmax
        self.timeintv = timeintv

        self.group = group

        return

    def _process_data(self, data) -> List[List[Atoms]]:
        """"""
        data = AtomsNDArray(data)
        self._debug(f"data: {data}")

        if data.ndim == 1:
            data = [data.tolist()]
        elif data.ndim == 2: # assume it is from minimisations...
            data = data.tolist()
        else:
            raise RuntimeError(f"Invalid shape {data.shape}.")

        return data
    
    def run(self, dataset: dict, worker=None, *args, **kwargs):
        """"""
        super().run()

        # - find some optional parameters
        labels = kwargs.get("labels", None)

        # - 
        self._print("process reference ->")
        reference = dataset.get("reference")
        if reference is not None:
            self._irun(reference, "ref-", labels)

        self._print("process prediction ->")
        prediction = dataset.get("prediction")
        if prediction is not None:
            self._irun(prediction, "pre-", labels)

        return
    
    def _irun(self, data, prefix="", labels=None):
        """Test the first trajectory.

        TODO: Several trajectories.

        """
        mdtrajs = self._process_data(data)
        group_indices = create_a_group(mdtrajs[0][0], self.group)
        self._debug(f"group_indices: {group_indices}")

        cache_msd = self.directory/f"{prefix}msd.npy"
        if not cache_msd.exists():
            data = Parallel(n_jobs=self.njobs)(
                delayed(compute_msd)(
                    [a for a in frames if a is not None], # AtomsNDArray may have None...
                    group_indices, lagmax=self.lagmax, 
                    start=self.start, end=self.end, timeintv=self.timeintv, 
                    prefix=prefix
                ) for frames in mdtrajs
            )
            np.save(cache_msd, data)
        else:
            data = np.load(cache_msd)

        lagtimes = [x[0] for x in data]
        timeseries = [x[1] for x in data]
        plot_msd(
            self.directory, names=labels, lagtimes=lagtimes, timeseries=timeseries, 
            start_step=self.d_start, end_step=self.d_end, prefix=prefix
        )

        return
    



if __name__ == "__main__":
    ...