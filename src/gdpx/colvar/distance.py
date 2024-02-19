#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

import jax
import jax.numpy as jnp


class DistanceColvar():

    def __init__(self, pairs, *args, **kwargs) -> None:
        """"""
        self.params = np.array(pairs).T

        return

    @staticmethod
    @jax.jit
    def cvfunc(positions, params):
        """"""
        pair_indices = params

        pair_positions = jnp.take(positions, pair_indices, axis=0)
        dvecs = pair_positions[0] - pair_positions[1]
        distances = jnp.linalg.norm(dvecs, axis=1)

        return distances[jnp.newaxis, :] # (1, num_dim)


if __name__ == "__main__":
    ...