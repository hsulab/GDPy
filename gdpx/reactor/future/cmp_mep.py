#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import numpy as np

from ase.io import read, write
from ase.geometry import find_mic


def write_neb_data(opt_images):
    """"""
    nimages = len(opt_images)
    
    energies = np.array([a.get_potential_energy() for a in opt_images])
    energies = energies - energies[0]
    
    differences = np.zeros(len(opt_images))
    init_pos = opt_images[0].get_positions()
    for i in range(1,nimages):
        a = opt_images[i]
        vector = a.get_positions() - init_pos
        vmin, vlen = find_mic(vector, a.get_cell())
        #differences[i] = np.linalg.norm(vlen)
        differences[i] = np.linalg.norm(vmin)
    
    # save results to file
    neb_data = "./neb.dat"
    content = ""
    for i in range(nimages):
        content += "{:4d}  {:10.6f}  {:10.6f}\n".format(
            i, differences[i], energies[i]
        )
    with open(neb_data, "w") as fopen:
        fopen.write(content)

    return


#dft_frames = read(
#    "/mnt/scratch2/users/40247882/validations/structures/surfaces-2x2/hcp2fcc_brg_mep.xyz",
#    ":"
#)
#write_neb_data(dft_frames)


def plot_comparasion(dft_file, mlp_file, comment=None):
    import matplotlib as mpl
    mpl.use('Agg') #silent mode
    from matplotlib import pyplot as plt
    plt.style.use('presentation')

    from scipy.interpolate import make_interp_spline

    fig, axarr = plt.subplots(
        nrows=1, ncols=1, 
        gridspec_kw={'hspace': 0.3}, figsize=(16,12)
    )
    plt.suptitle("Reaction Path Comparasion "+comment)

    axarr = [axarr]
    #axarr = axarr.flatten()

    #plt.suptitle("GCMC Evolution")
    ax = axarr[0]

    ax.set_xlabel("Reaction Coordinate [AA]")
    ax.set_ylabel("Potential Energy [eV]")

    # DFT data
    dft_data = np.loadtxt(dft_file, dtype=float)
    ax.scatter(dft_data[:,1], dft_data[:,2], marker="x", label="DFT")
    model = make_interp_spline(dft_data[:,1], dft_data[:,2], k=2)
    xs = np.linspace(dft_data[0,1], dft_data[-1,1], 500)
    ys = model(xs)
    ax.plot(xs, ys)

    # MLP data
    mlp_data = np.loadtxt(mlp_file, dtype=float)
    ax.scatter(mlp_data[:,1], mlp_data[:,2], marker="x", label="MLP")
    model = make_interp_spline(mlp_data[:,1], mlp_data[:,2], k=2)
    xs = np.linspace(mlp_data[0,1], mlp_data[-1,1], 500)
    ys = model(xs)
    ax.plot(xs, ys)
    if mlp_data.shape[1] == 4:
        apex = 2
        ax.errorbar(
            mlp_data[:,1], mlp_data[:,2], yerr=apex*mlp_data[:,3],
            elinewidth=4, linewidth=0,
            label="2*deviation"
        )

    ax.legend()

    plt.savefig("mep.png")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dft"
    )
    parser.add_argument(
        "--mlp"
    )
    parser.add_argument(
        "-c", "--comment"
    )

    args = parser.parse_args()

    plot_comparasion(args.dft, args.mlp, args.comment)