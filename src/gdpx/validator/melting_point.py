#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import NoReturn, List, Union

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

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
from ..data.array import AtomsNDArray
from ..builder.group import create_a_group

"""Measure melting point.

"""

def jpcc2020_func(T, Tm, x1, x2, x3, x4):
    """JPCC2020"""

    return x1/(1+np.exp(-x2*(T-Tm)))+x3*T+x4

def sigmoid_func(T, Tm, x1, x2, x4):
    """"""

    return x1/(1+np.exp(-x2*(T-Tm)))+x4

def get_distance_matrix(atoms: Atoms, indices=None):
    """"""
    if indices is None:
        return atoms.get_all_distances(mic=False,vector=False)
    else:
        selected_positions = atoms.positions[indices, :]
        return distance_matrix(selected_positions, selected_positions)

def _icalc_local_lindemann_index(frames, group, n_jobs=1):
    """Calculate Lindemann Index of each atom.

    Returns:
        An array with shape (natoms,)

    """
    # NOTE: Use unwrapped positions when under PBC?
    frames = wrap_traj(frames) # align structures

    group_indices = create_a_group(frames[0], group)
    natoms = len(group_indices)

    with CustomTimer("Lindemann Index"):
        distances = Parallel(n_jobs=n_jobs)(
            delayed(get_distance_matrix)(atoms, group_indices) 
            for atoms in frames
        )
    distances = np.array(distances)

    dis2 = np.square(distances)
    dis2_avg = np.average(dis2, axis=0)

    dis_avg = np.average(distances, axis=0)
    masked_dis_avg = dis_avg + np.eye(natoms) # avoid 0. in denominator
    #print(masked_dis_avg)

    q = np.sum( np.sqrt(dis2_avg - dis_avg**2) / masked_dis_avg, axis=1) / (natoms-1)

    return q


class MeltingPointValidator(AbstractValidator):

    """Estimate the melting point from a series of MD simulations.
    
    For nanoparticles, the lindeman index is used. MD simulations with various 
    initial temperatures are performed. For bulk, phase co-existence approach is used.
    Different initial temperatures are set for solid and liquid, where the steady state is
    found.

    """

    def __init__(self, group, run_fit: bool=True, start=0, temperatures: List[float]=None, fitting="sigmoid", directory: Union[str, pathlib.Path] = "./", *args, **kwargs):
        """"""
        super().__init__(directory, *args, **kwargs)

        self.group = group

        self.run_fit = run_fit

        self.start = start

        if temperatures is None:
            temperatures = []
        self.temperatures = temperatures
        
        assert fitting in ["sigmoid", "jpcc2020"], "Unsupported fitting function."
        self.fitting = fitting

        return
    
    def _process_data(self, data) -> List[List[Atoms]]:
        """"""
        data = AtomsNDArray(data)

        if data.ndim == 1:
            data = [data.tolist()]
        elif data.ndim == 2: # assume it is from extract_cache...
            data = data.tolist()
        elif data.ndim == 3: # assume it is from a compute node...
            data_ = []
            for d in data[:]: # TODO: add squeeze method?
                data_.extend(d)
            data = data_
        else:
            raise RuntimeError(f"Invalid shape {data.shape}.")

        return data

    def run(self, dataset: dict, worker=None, *args, **kwargs):
        """"""
        super().run()

        self._print("process reference ->")
        reference = dataset.get("reference")
        if reference is not None:
            reference = self._process_data(reference)
            data = self._compute_melting_point(reference, prefix="ref-")
            self._plot_figure(data[:, 1], data[:, 0], prefix="ref-", run_fit=self.run_fit)

        self._print("process prediction ->")
        prediction = dataset.get("prediction")
        if prediction is not None:
            prediction = self._process_data(prediction)
            data = self._compute_melting_point(prediction, prefix="pre-")
            self._plot_figure(data[:, 1], data[:, 0], prefix="pre-", run_fit=self.run_fit)

        return
    
    def _compute_melting_point(self, trajectories, prefix=""):
        """"""
        start, intv, end = self.start, None, None
        temperatures = self.temperatures
        assert len(trajectories) == len(temperatures), "Inconsitent number of trajectories and temperatures."

        qnames = [str(t) for t in temperatures]

        cached_data_path = self.directory / f"{prefix}qmat.txt"
        if not cached_data_path.exists():
            self._debug(f"nprocessors: {self.njobs}")
            with CustomTimer("joblib", func=self._print):
                qmat = Parallel(n_jobs=1)(
                    delayed(_icalc_local_lindemann_index)(
                        [a for a in curr_frames[start::] if a is not None], 
                        self.group, n_jobs=self.njobs
                    ) for curr_frames in trajectories
                )
            qmat = np.array(qmat)

            np.savetxt(
                cached_data_path, qmat.T,
                header=(
                    "{:>11s}"+"{:>12s}"*(len(qnames)-1)
                ).format(*qnames),
                fmt="%12.4f"
            )
        else:
            qmat = np.loadtxt(cached_data_path).T
        
        # - postprocess data
        t = temperatures
        q = np.average(qmat, axis=1)

        sorted_indices = [
            x[0] for x in sorted(enumerate(temperatures), key=lambda x: x[1])
        ]
        self._debug(sorted_indices)

        data = np.vstack(
            (
                [t[i] for i in sorted_indices],
                [q[i] for i in sorted_indices]
            )
        ).T

        np.savetxt(
            self.directory/f"{prefix}data.txt", data,
            header="{:>11s}  {:>12s}".format("temperature", "<q>"),
            fmt="%12.4f"
        )

        return data

    def _plot_figure(self, q, t: List[float], prefix="", run_fit=True):
        """"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))

        # - text
        fig.suptitle("Lindemann Index")

        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("$<q(T)>$")

        # - 
        ax.scatter(t, q)

        # - fitted curve
        if run_fit:
            if self.fitting == "sigmoid":
                func = sigmoid_func
                initial_guess = [np.median(t), np.max(q), 1., np.min(q)]
            elif self.fitting == "jpcc2020":
                func = jpcc2020_func
                initial_guess = [np.median(t), np.max(q), 1., 0., np.min(q)]
            coefs, cov = curve_fit(func, t, q, initial_guess, method="dogbox")
            self._debug(coefs)

            t_ = np.arange(np.min(t), np.max(t), 2.)
            q_ = func(t_, *coefs)
            ax.plot(t_, q_, label=f"$T_m={coefs[0]:>8.2f}$")

            ax.legend(loc="upper left")

        plt.savefig(self.directory/f"{prefix}mp.png")

        return
    
    def _compare_results(self):
        """Compare reference and prediction."""

        return


if __name__ == "__main__":
    ...