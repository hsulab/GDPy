#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

import jax
import jax.numpy as jnp

from ase.io import read, write
from ase.geometry import find_mic

jax.config.update("jax_enable_x64", True)


def check_mass():

    def center_of_mass(positions, masses):
        """"""

        return jnp.dot(masses, positions) / jnp.sum(masses)

    def scale_positions(positions, cell):
        """"""
    
        return jnp.linalg.solve(cell, jnp.transpose(positions)).T
    
    atoms = read("./traj_500.xyz", "0")
    print(atoms)
    
    # cell = np.array(atoms.get_cell(complete=True))
    # jac = jax.jacfwd(scale_positions, argnums=0)(
    #     atoms.positions[47], cell
    # )
    # print(f"{jac =}")
    positions = atoms.get_positions()
    print(f"{positions[47] =}")

    jac = jax.jacfwd(center_of_mass, argnums=0)(
        positions[47], atoms.get_masses()[47]
    )
    print(f"{jac =}")

    return

# check_mass()
# exit()


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

def compute_com_energy(
    masses, 
    positions, 
    cell,
    saved_coms, sigma: float, omega: float
):
    """"""
    com = jnp.dot(masses, positions) / jnp.sum(masses)

    com = jnp.linalg.solve(cell.T, jnp.transpose(com)).T

    x, x_t = com, saved_coms
    x2 = (x - x_t)**2/2./sigma**2 # uniform sigma?
    v = omega*jnp.exp(-jnp.sum(x2, axis=1))

    energy = v.sum(axis=0)

    return energy


if __name__ == "__main__":
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
        atoms = frames[0]
        com = np.array(com_records[i][0])
        print(f"{com =}")
        com_record = np.array(com_records[i])
        energy, gradients = jax.value_and_grad(compute_com_energy, argnums=1)(
            atoms.get_masses()[g],
            atoms.get_positions()[g],
            np.array(atoms.get_cell(complete=True)),
            saved_coms=com_record,
            # sigma=1.2, omega=0.2  # cart_com
            sigma=0.1, omega=0.2  # frac_com
        )
        forces = -gradients
        print(f"{energy =}")
        print(f"{forces =}")
    ...
