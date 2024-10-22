#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import List, Optional

import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

import matplotlib.pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    ...

from joblib import delayed, Parallel

from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList, neighbor_list

from .validator import AbstractValidator
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

def calc_rdf(
        wdir: pathlib.Path, frames, pairs, volume: Optional[float]=None, 
        nbins: int=60, cutoff: float=6.0, n_jobs=1
    ) -> dict:
    """Calculate radial distribution.

    Args:
        wdir: Working directory that stores RDF results.
        frames: A List of Atoms objects.
        pairs: Target species pairs, for example, ["Cu-Cu", "Cu-O"].
        volume: System volume.
        nbins: Number of bins for histogram.
        cutoff: Cut-off radius in Angstrom.

    """
    if not wdir.exists():
        wdir.mkdir(parents=True)

    # --- parse system
    # NOTE: We assume the system volume does not change along the trajectory!
    # if volume is None:
    #     volume = frames[0].get_volume()

    # NOTE: the atom order should be consistent in the entire trajectory
    #       i.e. this does not work for variable-composition system
    chemical_symbols = frames[0].get_chemical_symbols()
    sym_dict = {k: [] for k in set(chemical_symbols)}
    for k, v in itertools.groupby(enumerate(chemical_symbols), key=lambda x:x[1]):
        sym_dict[k].extend([x[0] for x in v])

    pair_dict = {}
    for pair in pairs:
        p0, p1 = pair.split("-")
        first_indices  = copy.deepcopy(sym_dict.get(p0, []))
        second_indices = copy.deepcopy(sym_dict.get(p1, []))
        assert len(first_indices) > 0, f"Cant found {p0}."
        assert len(second_indices) > 0, f"Cant found {p1}."

        num_first, num_second = len(first_indices), len(second_indices)
        if p0 == p1:
            num_pairs= (num_first)*(num_second-1)
        else:
            num_pairs = num_first*num_second
        pair_dict[pair] = num_pairs

    # ---
    binwidth = cutoff/nbins
    bincentres = np.linspace(binwidth/2., cutoff+binwidth/2., nbins+1)
    left_edges = np.copy(bincentres) - binwidth/2.
    right_edges = np.copy(bincentres) + binwidth/2.
    bins = np.linspace(0., cutoff+binwidth, nbins+2)

    # ---
    def compute_distance_histogram(atoms, pairs, cutoff, bins, binwidth):
        """"""
        i, j, d = neighbor_list("ijd", atoms, cutoff=cutoff+binwidth)

        symbols = atoms.get_chemical_symbols()
        num_pairs = len(d)

        distance_dict = {k: [] for k in pairs}
        for p in range(num_pairs):
            pair = f"{symbols[i[p]]}-{symbols[j[p]]}"
            distance_dict[pair].append(d[p])

        dis_hist = {}
        for k, v in distance_dict.items():
            hist_, edges_ = np.histogram(v, bins)
            dis_hist[k] = hist_

        return dis_hist

    ret = Parallel(n_jobs=n_jobs)(
        delayed(compute_distance_histogram)(atoms, pairs, cutoff, bins, binwidth) for atoms in frames
    )

    dis_hist = {k: [] for k in pairs}
    for curr_dis_hist in ret:
        for k, v in curr_dis_hist.items():
            dis_hist[k].append(v)

    # get pair density
    density_dict = {k: [] for k in pairs}
    for atoms in frames:
        for k, num_pairs in pair_dict.items():
            if volume is None:
                density_dict[k].append(num_pairs/atoms.get_volume())
            else:
                density_dict[k].append(num_pairs/volume)

    # - reformat data
    results = {}
    for k, v in dis_hist.items():
        curr_dis_hist = np.array(v)
        avg_density = np.array(density_dict[k])[:, np.newaxis]

        # NOTE: VMD likely uses this formula
        vshells = 4.*np.pi*left_edges**2*binwidth 
        # vshells = 4./3.*np.pi*binwidth*(3*left_edges**2+3*left_edges*binwidth+binwidth**2)
        vshells[0] = 1. # avoid zero in division

        rdf = curr_dis_hist/vshells/avg_density

        rdf_avg = np.average(rdf, axis=0)
        rdf_min = np.min(rdf, axis=0)
        rdf_max = np.max(rdf, axis=0)
        rdf_svar = np.sqrt(np.var(rdf, axis=0))

        data = np.vstack((bincentres, rdf_avg, rdf_svar, rdf_min, rdf_max)).T
        np.savetxt(
            wdir/f"{k}.dat", data, 
            fmt="%8.4f  %8.4f  %8.4f  %8.4f  %8.4f", 
            header=("{:<8s}  "*5).format("r", "rdf", "svar", "min", "max")
        )
        results[k] = data

    return results


def plot_rdf(fig_path, data=None, ref_data=None, title="RDF"):
    """"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))

    plt.suptitle("Radial Distribution Function")
    ax.set_xlabel("r [Å]")
    ax.set_ylabel("g(r)")

    ax.set_title(title)

    if data is not None:
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

    def __init__(self, pairs: List[str], cutoff: float=6., nbins: int=60, directory = "./", *args, **kwargs) -> None:
        """Radial Distribution.

        Args:
            paris: A List of species pairs [Cu-Cu, ..., ...].

        """
        super().__init__(directory=directory, *args, **kwargs)

        self.pairs = pairs
        #assert len(self.pair), f"{self.__class__.__name__} requires two elements."

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
                self.directory/"reference", 
                ref_frames, self.pairs, self.cutoff, self.nbins,
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
                self.directory/"prediction", 
                pre_frames, self.pairs, self.cutoff, self.nbins,
                volume=volume
            )
        else:
            pre_data = None
        
        assert (ref_data is not None or pre_data is not None), "Neither reference nor prediction is given."

        # compare results
        self._compare_results(ref_data, pre_data)

        return
    
    def _compute_rdf(self, wdir, frames: List[Atoms], pairs, cutoff, nbins, volume: float=None):
        """"""
        if not wdir.exists():
            data = calc_rdf(
                wdir, frames, pairs, volume, nbins, cutoff, n_jobs=self.njobs
            )
        else:
            data = {}
            saved_files = list(wdir.glob("*.dat"))
            for p in saved_files:
                data[p.name[:-4]] = np.loadtxt(p)

        return data
    
    def _compare_results(self, reference, prediction):
        """"""
        for pair in self.pairs:
            p, r = None, None
            if prediction is not None:
                p = prediction.get(pair, None)
            if reference is not None:
                r = reference.get(pair, None)
            if not (p is None and r is None):
                plot_rdf(
                    self.directory/f"{pair}_rdf.png", 
                    p, r, title=pair
                )
            else:
                if p is not None:
                    plot_rdf(
                        self.directory/f"{pair}_rdf.png", 
                        p, None, title=pair
                    )
                else:  # if r is not None:
                    plot_rdf(
                        self.directory/f"{pair}_rdf.png", 
                        None, r, title=pair
                    )

        return


if __name__ == "__main__":
    ...
