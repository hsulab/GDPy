#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import jax
import jax.numpy as jnp
import numpy as np

from gdpx.bias.harmonic.distance import (
    compute_distance, compute_distance_harmonic_energy_and_forces)


def compute_distance_harmonic_jax(positions, a0, a1, center, kspring):
    """"""
    distance = jnp.linalg.norm(positions[a0, :] - positions[a1, :])

    energy = 0.5 * kspring * (distance - center) ** 2

    return energy, distance


if __name__ == "__main__":
    from ase.io import read, write

    atoms = read("./packed-init-344.xyz")

    center, kspring = 2.8, 3.6

    # compute by numpy
    vec, dis = compute_distance(atoms.cell, atoms.positions[[37, 38], :], pbc=True)
    print(f"{dis =}")

    energy, forces = compute_distance_harmonic_energy_and_forces(
        vec, dis, center=center, kspring=kspring
    )
    print(f"{energy =}")
    print(f"{forces =}")

    # compute by jax
    (energy, distance), gradients = jax.value_and_grad(
        compute_distance_harmonic_jax, argnums=0, has_aux=True
    )(atoms.get_positions(), 37, 38, center=center, kspring=kspring)
    print(f"{distance =}")
    print(f"{energy =}")
    forces = -np.array(gradients)[[37, 38]]
    print(f"{forces =}")
