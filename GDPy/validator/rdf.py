#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
plt.style.use("presentation")

from ase.io import read, write

from GDPy.validator.validator import AbstractValidator

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
    #print(bins)

    first_indices = [174, 175]
    #first_indices = [176, 177, 178, 179]

    from ase.neighborlist import NeighborList
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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,12))

    plt.suptitle("Radial Distribution Function")
    ax.set_xlabel("r [Å]")
    ax.set_ylabel("g(r)")

    ax.set_title(title)

    bincentres, rdf = data[:,0], data[:,1]
    bincentres_, rdf_ = smooth_curve(bincentres, rdf)
    ax.plot(bincentres_, rdf_, label="MLIP")

    if ref_data is not None:
        bincentres, rdf = ref_data[:,0], ref_data[:,1]
        bincentres_, rdf_ = smooth_curve(bincentres, rdf)
        ax.plot(bincentres_, rdf_, ls="-.", label="Ref.")

    plt.legend()

    plt.savefig(fig_path)

    return


class RdfValidator(AbstractValidator):


    def run(self):
        """"""
        params = self.task_params
        traj_fpath = params["trajectory"]["path"]
        indices = params["trajectory"].get("indices", ":")

        rdf_params = params["rdf_params"]
        nbins = rdf_params.get("nbins", 60)
        cutoff = rdf_params.get("cutoff", 6.)
        pairs = rdf_params["pairs"]
        assert len(pairs) == 2, "RDF needs two elements."

        frames = read(traj_fpath, indices)

        # NOTE: the atom order should be consistent in the entire trajectory
        first_indices, second_indices = [], []
        for i, sym in enumerate(frames[0].get_chemical_symbols()):
            if sym == pairs[0]:
                first_indices.append(i)
            if sym == pairs[1]:
                second_indices.append(i)
        assert len(first_indices) > 0, f"Cant found {pairs[0]}."
        assert len(second_indices) > 0, f"Cant found {pairs[1]}."
        if pairs[0] == pairs[1]:
            avg_density = len(first_indices)/frames[0].get_volume()
        else:
            avg_density = (len(first_indices)+len(second_indices))/frames[0].get_volume()

        dat_fpath = self.directory/("-".join(pairs)+".dat")
        if not dat_fpath.exists():
            data = calc_rdf(
                dat_fpath, frames, first_indices, second_indices, avg_density, nbins, cutoff
            )
        else:
            data = np.loadtxt(dat_fpath)

        # - reference
        ref_data = np.loadtxt(params["reference"])

        plot_rdf(self.directory/"rdf.png", data, ref_data, title="-".join(pairs))

        return


if __name__ == "__main__":
    pass