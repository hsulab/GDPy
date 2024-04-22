#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

import jax
import jax.numpy as jnp


@jax.jit
def compute_bond_angles(positions, trimer_indices):
    """Compute angles.

    For very small/acute angles, results by arccos are inaccurate. arctan may be
    more effective.

    """
    trimer_positions = jnp.take(positions, trimer_indices.T, axis=0)
    # TODO: shifts
    dvecs1 = trimer_positions[1] - trimer_positions[0]
    dnorms1 = jnp.linalg.norm(dvecs1, axis=1)
    dvecs2 = trimer_positions[2] - trimer_positions[0]
    dnorms2 = jnp.linalg.norm(dvecs2, axis=1)

    angles = jnp.arccos(jnp.sum(dvecs1 * dvecs2, axis=1) / dnorms1 / dnorms2)

    return angles


compute_angle_jacobian = jax.jacrev(compute_bond_angles, argnums=0)


@jax.jit
def pseudo_inverse_of_jacobian(jac, eps=0.0001):
    """"""
    dim = jac.shape[0]
    jac_inv = jnp.transpose(jac) @ jnp.linalg.inv(
        jac @ jnp.transpose(jac) + eps * jnp.eye(dim)
    )

    return jac_inv


@jax.jit
def optimisation_step(jac, disp, eps=0.0001):
    """"""
    jac_inv = pseudo_inverse_of_jacobian(jac, eps)

    return jnp.reshape(jac_inv @ disp, (-1, 3))


if __name__ == "__main__":
    ...
