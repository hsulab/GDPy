#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    print("Used default matplotlib style.")

import ase
from ase.io import read, write

"""This module is used to analyse the particle distribution in the system.
"""

def calc_gyration(atoms, particle: int="Cr", method="cluster", cutoff=30., nbins=60):
    """"""
    # - get distances
    #   different methods for cluster/surface systems
    # -- get the com of the structure
    com = atoms.get_center_of_mass()
    print(com)

    # -- get the particle positions
    selected_positions = []
    for a in atoms:
        if a.symbol == particle:
            selected_positions.append(a.position)
    selected_positions = np.array(selected_positions)
    nselected = selected_positions.shape[0]
    #print(nselected)
    #print(selected_positions)

    # -- calc distances
    if method == "cluster":
        distances = np.linalg.norm(selected_positions - com, axis=1)
    elif method == "surface":
        # distance to the com plane for both upper and bottom surface
        distances = np.fabs(selected_positions[:,2] - com[2])
    else:
        raise NotImplementedError(f"Unknown method {method}.")
    print("distances: ", distances)

    # - bins
    #cutoff, nbins = 30., 60
    binwidth = cutoff/nbins
    bincentres = np.linspace(binwidth/2., cutoff+binwidth/2., nbins+1)
    left_edges = np.copy(bincentres) - binwidth/2.

    bins = np.linspace(0., cutoff+binwidth, nbins+2)

    hist_, edges_ = np.histogram(distances, bins)
    #dis_hist.append(hist_)
    print(hist_)

    # TODO: only for atom
    particle_mass = ase.data.atomic_masses[ase.data.atomic_numbers[particle]]
    avg_density = nselected*particle_mass/atoms.get_volume()

    if method == "cluster":
        vshells = 4.*np.pi*left_edges**2*binwidth # NOTE: VMD likely uses this formula
        vshells[0] = 1. # avoid zero in division

    elif method == "surface":
        cell = atoms.get_cell(complete=True)
        plane_area = np.linalg.norm(np.cross(cell[0], cell[1]))
        #print(plane_area)
        vshells = 2.*binwidth*np.ones(left_edges.shape)*plane_area # bottom and upper
    else:
        raise NotImplementedError(f"Unknown method {method}.")
    rdf = hist_/vshells/avg_density
    print(rdf)

    return bincentres, rdf

def plot_rdf(bincentres, rdf, step=5):
    """"""
    print("rdf shape: ", rdf.shape)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,12))

    plt.suptitle("Mass Distribution Function")
    ax.set_xlabel("r [Å]")
    ax.set_ylabel("g(r)")

    avg_rdf = np.average(rdf, axis=0)
    std_rdf = np.sqrt(np.var(rdf, axis=0))
    print("avg rdf: ", avg_rdf)
    ax.plot(bincentres, avg_rdf, label="avg")
    ax.fill_between(bincentres, avg_rdf-std_rdf, avg_rdf+std_rdf, alpha=0.2, label="std")

    # - stepwise
    nframes = rdf.shape[0]
    for i in range(0, nframes, step):
        cur_rdf = rdf[i]
        ax.plot(bincentres, cur_rdf, label=f"rdf{i}")

    ax.legend()

    plt.savefig("mdf.png")

    return


if __name__ == "__main__":
    start, step = 4000, 500
    rdf_frames = read("./xxx/mc.xyz", f"{start}::{step}")

    nframes = len(rdf_frames)
    print("nframes for rdf: ", nframes)

    #print(nframes)
    rdf = []
    for atoms in rdf_frames:
        bincentres, cur_rdf = calc_gyration(atoms, method="surface")
        rdf.append(cur_rdf)
    rdf = np.array(rdf)

    plot_rdf(bincentres, rdf)
    ...