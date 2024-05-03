#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import jax
import jax.numpy as jnp
import numpy as np

from gdpx.bias.gaussian.distance import (compute_bias_forces,
                                         compute_colvar_and_gradient,
                                         compute_gaussian_and_gradient)


def compute_distance_gaussian_jax(positions, a0, a1, sigma, omega, s_t):
    """"""
    s = jnp.linalg.norm(positions[a0, :] - positions[a1, :])

    s1 = s - s_t
    s2 = s1**2 / 2.0 / sigma**2
    v = jnp.sum(omega * jnp.exp(-jnp.sum(s2, axis=1)))

    return v, s


if __name__ == "__main__":
    from ase.io import read, write

    atoms = read("./packed-init-344.xyz")

    sigma, omega = np.array([0.2]), 0.5
    history_records = np.array([[2.4], [3.9]])

    # compute by numpy
    s, dsdx = compute_colvar_and_gradient(
        atoms.cell, atoms.positions[[37, 38], :], pbc=True
    )
    print(f"{s =}")
    print(f"{dsdx =}")

    v, dvds = compute_gaussian_and_gradient(s, history_records, sigma, omega)
    print(f"{v =}")
    print(f"{dvds =}")
    forces = compute_bias_forces(dvds, dsdx)
    print(f"{forces =}")

    # compute by jax
    (energy, distance), gradients = jax.value_and_grad(
        compute_distance_gaussian_jax, argnums=0, has_aux=True
    )(atoms.get_positions(), 37, 38, sigma=sigma, omega=omega, s_t=history_records)
    print(f"{distance =}")
    print(f"{energy =}")
    forces = -np.array(gradients)[[37, 38]]
    print(f"{forces =}")
