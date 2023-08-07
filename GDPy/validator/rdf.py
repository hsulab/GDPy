#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
from typing import List

import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    print("Used default matplotlib style.")

from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList

from .validator import AbstractValidator
from .utils import wrap_traj
from ..data.array import AtomsNDArray

def smooth_curve(bins, points):
    """"""
    spl = make_interp_spline(bins, points, k=3)
    bins = np.linspace(bins.min(), bins.max(), 300)
    points= spl(bins)

    for i, d in enumerate(points):
        if d < 1e-6:
            points[i] = 0.0

    return bins, points

def calc_rdf(dat_fpath, frames, first_indices, second_indices, avg_density, nbins=60, cutoff=6.0):
    binwidth = cutoff/nbins
    bincentres = np.linspace(binwidth/2., cutoff+binwidth/2., nbins+1)
    #print(bincentres)
    left_edges = np.copy(bincentres) - binwidth/2.
    right_edges = np.copy(bincentres) + binwidth/2.
    bins = np.linspace(0., cutoff+binwidth, nbins+2)

    dis_hist = []
    for atoms in frames:
        cell = atoms.get_cell()
        #symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        natoms = len(atoms)
        nl = NeighborList(
            [(cutoff+binwidth)/2.]*natoms, skin=0., sorted=False,
            self_interaction=False, bothways=True,
        )
        nl.update(atoms)
        for x in first_indices:
            nei_indices, nei_offsets = nl.get_neighbors(x)
            distances, sel_indices = [], []
            for nei_ind, nei_off in zip(nei_indices, nei_offsets):
                #if symbols[nei_ind] == "O":
                if nei_ind in second_indices:
                    sel_indices.append(nei_ind)
                    dis = np.linalg.norm(
                        positions[x] -
                        (positions[nei_ind] + np.dot(cell, nei_off))
                    )
                    distances.append(dis)
            hist_, edges_ = np.histogram(distances, bins)
            dis_hist.append(hist_)

    # - reformat data
    dis_hist = np.array(dis_hist)

    vshells = 4.*np.pi*left_edges**2*binwidth # NOTE: VMD likely uses this formula
    #vshells = 4./3.*np.pi*binwidth*(3*left_edges**2+3*left_edges*binwidth+binwidth**2)
    vshells[0] = 1. # avoid zero in division

    rdf = dis_hist/vshells/avg_density

    rdf_avg = np.average(rdf, axis=0)
    rdf_min = np.min(rdf, axis=0)
    rdf_max = np.max(rdf, axis=0)
    rdf_svar = np.sqrt(np.var(rdf, axis=0))

    data = np.vstack((bincentres,rdf_avg,rdf_svar,rdf_min,rdf_max)).T
    np.savetxt(
        dat_fpath, data, 
        fmt="%8.4f  %8.4f  %8.4f  %8.4f  %8.4f", 
        header=("{:<8s}  "*5).format("r", "rdf", "svar", "min", "max")
    )

    return data


def plot_rdf(fig_path, data, ref_data=None, title="RDF"):
    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))

    plt.suptitle("Radial Distribution Function")
    ax.set_xlabel("r [Å]")
    ax.set_ylabel("g(r)")

    ax.set_title(title)

    bincentres, rdf = data[:,0], data[:,1]
    bincentres_, rdf_ = smooth_curve(bincentres, rdf)
    ax.plot(bincentres_, rdf_, label="prediction")

    if ref_data is not None:
        bincentres, rdf = ref_data[:,0], ref_data[:,1]
        bincentres_, rdf_ = smooth_curve(bincentres, rdf)
        ax.plot(bincentres_, rdf_, ls="-.", label="reference")

    plt.legend()

    plt.savefig(fig_path)

    return


class RdfValidator(AbstractValidator):

    def __init__(self, pair: List[str], cutoff: float=6., nbins: int=60, directory = "./", *args, **kwargs) -> None:
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        self.pair = pair
        assert len(self.pair), f"{self.__class__.__name__} requires two elements."
        self.cutoff = cutoff
        self.nbins = nbins

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

        return data[0] # TODO: support several trajectories

    def run(self, dataset, worker=None, *args, **kwargs):
        """Process reference and prediction data separately.

        TODO:

            Support average over several trajectories.

        """
        # - get custom volume
        volume = kwargs.get("volume", None)

        # - process dataset
        self._print("process reference ->")
        reference = dataset.get("reference", None)
        if reference is not None:
            ref_frames = self._process_data(reference)
            self._debug(f"reference  nframes: {len(ref_frames)}")
            ref_data = self._compute_rdf(
                self.directory/("-".join(self.pair)+"_ref.dat"), 
                ref_frames, self.pair, self.cutoff, self.nbins,
                volume=volume
            )
        else:
            ref_data = None

        self._print("process prediction ->")
        prediction = dataset.get("prediction", None)
        if prediction is not None:
            pre_frames = self._process_data(prediction)
            self._debug(f"prediction nframes: {len(pre_frames)}")
            pre_data = self._compute_rdf(
                self.directory/("-".join(self.pair)+"_pre.dat"), 
                pre_frames, self.pair, self.cutoff, self.nbins,
                volume=volume
            )
        else:
            pre_data = None
        
        assert (ref_data is not None or pre_data is not None), "Neither reference nor prediction is given."

        # - compare results
        self._compare_results(ref_data, pre_data)

        return
    
    def _compute_rdf(self, dat_fpath, frames: List[Atoms], pair, cutoff, nbins, volume: float=None):
        """"""
        # - treat pbc
        if any(frames[0].pbc):
            frames = wrap_traj(frames)
        
        # - volume
        if volume is None:
            volume = frames[0].get_volume()

        # NOTE: the atom order should be consistent in the entire trajectory
        #       i.e. this does not work for variable-composition system
        chemical_symbols = frames[0].get_chemical_symbols()
        sym_dict = {k: [] for k in set(chemical_symbols)}
        for k, v in itertools.groupby(enumerate(chemical_symbols), key=lambda x:x[1]):
            sym_dict[k].extend([x[0] for x in v])

        first_indices  = copy.deepcopy(sym_dict.get(pair[0], []))
        second_indices = copy.deepcopy(sym_dict.get(pair[1], []))
        assert len(first_indices) > 0, f"Cant found {pair[0]}."
        assert len(second_indices) > 0, f"Cant found {pair[1]}."
        self._debug(f"first : {first_indices}")
        self._debug(f"second: {second_indices}")

        if pair[0] == pair[1]:
            avg_density = len(first_indices)/volume
        else:
            avg_density = (len(first_indices)+len(second_indices))/volume

        if not dat_fpath.exists():
            data = calc_rdf(
                dat_fpath, frames, first_indices, second_indices, avg_density, nbins, cutoff
            )
        else:
            data = np.loadtxt(dat_fpath)

        return data
    
    def _compare_results(self, reference, prediction):
        """"""
        plot_rdf(
            self.directory/"rdf.png", 
            prediction, reference, title="-".join(self.pair)
        )

        return


if __name__ == "__main__":
    ...