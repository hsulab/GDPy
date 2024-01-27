#!/usr/bin/env python3
# -*- coding: utf-8 -*


import numpy as np
from scipy.spatial import distance_matrix


from ase.io import read, write


def switch_function(distances, r_cut, r_shift=0., nn=6, mm: int=None):
    """"""
    if mm is None:
        mm = nn*2

    scaled_distances = (distances - r_shift) / r_cut

    return (1 - scaled_distances**nn) / (1 - scaled_distances**mm)


def compute_coordination_number():
    """"""
    frames = read(
        "/scratch/gpfs/jx1279/copper+alumina/dptrain/r8/_explore/_mrxn/Cu13+s001p32/_sinter/3xCu13/_200ps/cand0_400K_200ps/traj.dump", 
        ":1"
    )
    nframes = len(frames)
    print(f"nframes: {nframes}")

    atoms = frames[0]

    positions = atoms.positions[-13:, :]
    print(positions)
    clusters = [
        atoms.positions[-13:, :],
        atoms.positions[-26:-13, :],
        atoms.positions[-39:-26, :],
    ]

    com_clusters = [np.mean(p, axis=0) for p in clusters]
    print(com_clusters)
    print(np.linalg.norm(com_clusters[2] - [0., 25.239, 0.] - com_clusters[1]))
    print(np.linalg.norm(com_clusters[2] - com_clusters[0]))

    def xxx(positions):
        dmat = distance_matrix(positions, positions)
        #print(dmat)

        sf = switch_function(dmat, r_cut=3.8, nn=8, mm=14)
        np.fill_diagonal(sf, 0.)
        coordination = np.sum(sf, axis=1)
        print(coordination)

        return coordination
    
    for positions in clusters:
        print(np.sum(xxx(positions)))

    return


if __name__ == "__main__":
    compute_coordination_number()
    ...