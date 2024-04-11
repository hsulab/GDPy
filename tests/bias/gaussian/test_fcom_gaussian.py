#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

from ase.geometry import find_mic

def compute_center_of_mass(
    cell, masses, positions, saved_positions, scaled: bool=False, pbc: bool = True
):
    """Compute center_of_mass in the fractional space.

    The positions should be properly processed with find_mic.

    Args:
        positions: The cartesian coordinates of a group of atoms.

    """

    shift = positions - saved_positions
    curr_vectors, curr_distances = find_mic(shift, cell, pbc=True)

    shifted_positions = positions + curr_vectors

    # dcom/dx = masses/np.sum(masses)
    com = masses @ shifted_positions / np.sum(masses)

    if scaled:
        com = cell.scaled_positions(com)
        for i in range(3):  # FIXME: seleced pbc?
            com[i] %= 1.0
            com[i] %= 1.0 # need twice see ase test

    return shifted_positions, com


def compute_com_energy_and_forces(cell, masses, com, saved_coms, sigma: float, omega: float):
    """"""
    # - compute energy
    x, x_t = com, saved_coms
    x1 = x - x_t
    x2 = x1**2/2./sigma**2 # uniform sigma?
    v = omega*np.exp(-np.sum(x2, axis=1))

    energy = v.sum(axis=0)

    # - compute forces
    # -- dE/dx
    dEdx = np.sum(-v[:, np.newaxis]*x1/sigma**2, axis=0)[np.newaxis, :]
    print(f"{dEdx =}")

    m = (masses/np.sum(masses))[:, np.newaxis]
    print(f"{m.shape =}")

    c_inv = np.linalg.inv(cell)
    forces = -np.transpose(c_inv @ (m @ dEdx).T)

    return energy, forces



if __name__ == "__main__":
    """"""
    from ase.io import read, write
    frames = read("./traj_500.xyz", ":200:50")

    scaled = True

    # - 
    groups = [[45, 46], [47]]

    saved_positions = []
    for g in groups:
        positions = frames[0].get_positions()
        saved_positions.append([positions[i] for i in g])
    # print(saved_positions)

    com_records = [[], []]
    for atoms in frames:
        masses = atoms.get_masses()
        positions = atoms.get_positions()
        for i, g in enumerate(groups):
            shifted_positions, com = compute_center_of_mass(
                atoms.cell,
                masses[g],
                positions[g],
                saved_positions=saved_positions[i],
                scaled=scaled,
                pbc=True,
            )
            # print(f"{i}: {fcom =}")
            com_records[i].append(com)
    # print(f"{fcom_records =}")

    # - 
    for i, g in enumerate(groups):
        com = np.array(com_records[i][0])
        print(f"{com =}")
        com_record = np.array(com_records[i])
        print(f"{com_record =}")
        energy, forces = compute_com_energy_and_forces(
            cell=atoms.get_cell(complete=True),
            masses=atoms.get_masses()[g],
            com=com, saved_coms=com_record,
            # sigma=1.2, omega=0.2
            sigma=0.1, omega=0.2
        )
        print(f"{energy =}")
        print(f"{forces =}")
    ...
