#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import jax
import jax.numpy as jnp


@jax.jit
def compute_distance(positions, pair_indices):
    """"""
    pair_positions = jnp.take(positions, pair_indices, axis=0)
    dvecs = pair_positions[0] - pair_positions[1]
    distances = jnp.linalg.norm(dvecs, axis=1)

    #return distances[:, jnp.newaxis]
    return distances[0]

@jax.jit
def compute_distance_bias(positions, pair_indices, x_t, sigma, omega):
    """"""
    pair_positions = jnp.take(positions, pair_indices, axis=0)
    dvecs = pair_positions[0] - pair_positions[1]
    distances = jnp.linalg.norm(dvecs, axis=1)
    distances = distances[:, jnp.newaxis]

    x2 = (distances - x_t)**2/2./sigma**2
    v = omega*jnp.exp(-jnp.sum(x2, axis=1))

    return v.sum(axis=0)


if __name__ == "__main__":
    ...