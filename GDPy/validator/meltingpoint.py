#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import NoReturn, List

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    print("Used default matplotlib style.")

from GDPy.core.register import registers
from GDPy.core.operation import Operation
from GDPy.validator.validator import AbstractValidator
from GDPy.utils.command import CustomTimer

"""Measure melting point.

"""

def mp_func(T, Tm, x1, x2, x3, x4):
    """JPCC2020"""

    return x1/(1+np.exp(-x2*(T-Tm)))+x3*T+x4

def mp2_func(T, Tm, x1, x2, x4):
    """"""

    return x1/(1+np.exp(-x2*(T-Tm)))+x4

def get_distance_matrix(atoms: Atoms):
    """"""

    return atoms.get_all_distances(mic=False,vector=False)

def _icalc_local_lindemann_index(frames):
    """Calculate Lindemann Index of each atom.

    Returns:
        An array with shape (natoms,)

    """
    nframes = len(frames)
    natoms = len(frames[0])

    with CustomTimer("Lindemann Index"):
        distances = Parallel(n_jobs=8)(
            delayed(get_distance_matrix)(atoms) 
            for atoms in frames
        )
        #distances = np.zeros((nframes,natoms,natoms))
        #for i, atoms in enumerate(frames):
        #    # NOTE: Use unwrapped positions when under PBC.
        #    dis_mat = atoms.get_all_distances(mic=False,vector=False)
        #    distances[i,:,:] = dis_mat
        #print(distances.shape)
        #print(distances)
    distances = np.array(distances)

    dis2 = np.square(distances)
    dis2_avg = np.average(dis2, axis=0)

    dis_avg = np.average(distances, axis=0)
    masked_dis_avg = dis_avg + np.eye(natoms) # avoid 0. in denominator
    #print(masked_dis_avg)

    q = np.sum( np.sqrt(dis2_avg - dis_avg**2) / masked_dis_avg, axis=1) / (natoms-1)

    return q

@registers.validator.register
class MeltingpointValidator(AbstractValidator):

    """Estimate the melting point from a series of MD simulations.
    
    For nanoparticles, the lindeman index is used. MD simulations with various 
    initial temperatures are performed. For bulk, phase co-existence approach is used.
    Different initial temperatures are set for solid and liquid, where the steady state is
    found.

    """

    def run(self, trajectories: List[List[Atoms]], temperatures: List[float]):
        """"""
        super().run()
        start, intv = 121, 0
        qnames = [str(t) for t in temperatures]

        cached_data_path = self.directory / "ret.txt"
        if not cached_data_path.exists():
            with CustomTimer("joblib"):
                qmat = Parallel(n_jobs=1)(
                    delayed(_icalc_local_lindemann_index)(curr_frames[start::]) 
                    for curr_frames in trajectories
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
        
        t = temperatures
        q = np.average(qmat, axis=1)
        #self.pfunc(t)
        #self.pfunc(q)
        print(t)
        print(q)

        sorted_indices = [
            x[0] for x in sorted(enumerate(temperatures), key=lambda x: x[1])
        ]
        print(sorted_indices)

        data = np.vstack(
            (
                [t[i] for i in sorted_indices],
                [q[i] for i in sorted_indices]
            )
        ).T

        np.savetxt(
            self.directory/"data.txt", data,
            header="{:>11s}  {:>12s}".format("temperature", "<q>"),
            fmt="%12.4f"
        )

        self._plot_figure(q, t)

        return

    def _plot_figure(self, q, t: List[float]):
        """"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
        fig.suptitle("Melting Temperatures by DP")

        #func = mp_func
        #initial_guess = [np.median(t), np.max(q), 1., 0., np.min(q)]

        func = mp2_func
        initial_guess = [np.median(t), np.max(q), 1., np.min(q)]

        coefs, cov = curve_fit(func, t, q, initial_guess, method="dogbox")
        print(coefs)

        # - fitted curve
        t_ = np.arange(np.min(t), np.max(t), 2.)
        q_ = func(t_, *coefs)

        ax.scatter(t, q)
        ax.plot(t_, q_, label=f"$T_m={coefs[0]:>8.2f}$")

        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("$<q(T)>$")

        # - add Huttig and Tamman
        y = [0., 0.2]
        x = [1358*0.3]*len(y)
        ax.plot(x, y, ls="--", label="Hüttig", zorder=100)
        x = [1358*0.5]*len(y)
        ax.plot(x, y, ls="--", label="Tamman", zorder=100)

        ax.legend(loc="upper left")

        plt.savefig(self.directory/"mp.png")

        return

@registers.operation.register
class mptest(Operation):

    def __init__(self, extract_node) -> NoReturn:
        """"""
        super().__init__([extract_node])

        return
    
    def forward(self, trajectories: List[List[Atoms]]):
        """"""
        super().forward()

        workers = self.input_nodes[0].workers

        temperatures = []
        for w in workers:
            temp = w.driver.as_dict().get("temp")
            temperatures.append(temp)
        
        validator = MeltingpointValidator("./", {})
        validator.directory = self.directory

        validator.run(trajectories, temperatures)

        return


if __name__ == "__main__":
    # - Cu10
    #temperatures = list(range(50,300,50)) + list(range(300,1250,50))
    # - Cu616
    temperatures = list(range(860,900,10)) + list(range(300,1250,50))
    fpaths = [
        # - Cu10
        #"./mdruns/2_scan_2/w0/cand0/traj.dump", # 50
        #"./mdruns/2_scan_2/w1/cand0/traj.dump",
        #"./mdruns/2_scan_2/w2/cand0/traj.dump",
        #"./mdruns/2_scan_2/w3/cand0/traj.dump",
        #"./mdruns/2_scan_2/w4/cand0/traj.dump", # 250
        # - Cu616
        "./mdruns/2_scan_2/w0/cand0/traj.dump",
        "./mdruns/2_scan_2/w1/cand0/traj.dump",
        "./mdruns/2_scan_2/w2/cand0/traj.dump",
        "./mdruns/2_scan_2/w3/cand0/traj.dump",
        "./mdruns/1_scan/w0/cand0/traj.dump",
        "./mdruns/1_scan/w1/cand0/traj.dump",
        "./mdruns/1_scan/w2/cand0/traj.dump",
        "./mdruns/1_scan/w3/cand0/traj.dump",
        "./mdruns/1_scan/w4/cand0/traj.dump",
        "./mdruns/1_scan/w5/cand0/traj.dump",
        "./mdruns/1_scan/w6/cand0/traj.dump",
        "./mdruns/1_scan/w7/cand0/traj.dump",
        "./mdruns/1_scan/w8/cand0/traj.dump",
        "./mdruns/1_scan/w9/cand0/traj.dump",
        "./mdruns/1_scan/w10/cand0/traj.dump",
        "./mdruns/1_scan/w11/cand0/traj.dump",
        "./mdruns/1_scan/w12/cand0/traj.dump",
        "./mdruns/1_scan/w13/cand0/traj.dump",
        "./mdruns/1_scan/w14/cand0/traj.dump",
        "./mdruns/1_scan/w15/cand0/traj.dump",
        "./mdruns/1_scan/w16/cand0/traj.dump",
        "./mdruns/1_scan/w17/cand0/traj.dump",
        "./mdruns/1_scan/w18/cand0/traj.dump",
    ]
    #trajectories = []
    #for p in fpaths:
    #    traj = read(p, ":")
    #    print("nframes: ", len(traj))
    #    trajectories.append(traj)
    trajectories = Parallel(n_jobs=8)(
        delayed(read)(p, ":") for p in fpaths
    )

    validator = MeltingpointValidator("./", {})
    validator.run(trajectories, temperatures)
    ...