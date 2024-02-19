#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import itertools

import numpy as np

import jax
import jax.numpy as jnp

from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs

from .gptools import compute_body3_descriptor


def switch_function(r, r_cut, gamma=2):
    """"""
    pr = r/r_cut

    return 1. - (1.+gamma)*pr**gamma + gamma*pr**(1+gamma)

def radial_function(r, r_span, width, r_cut, gamma):
    """Compute radial function.

    Args:
        r:
        r_span: shape (1, num_span)
        f_cut: shape (num_r, 1)

    """
    constant = 1./(r_cut/r_span.shape[1])
    f_cut = switch_function(r, r_cut, gamma)

    return constant*1./r**2*f_cut*jnp.exp(-(r-r_span)**2/2/width**2)

def compute_feature(positions, pair_indices, r_span, width, r_cut, gamma):
    """"""
    pair_positions = jnp.take(positions, pair_indices, axis=0)
    dis_vecs = pair_positions[0] - pair_positions[1]
    distances = jnp.linalg.norm(dis_vecs, axis=1)[:, jnp.newaxis] # (num_b2, 1)

    radial_values = radial_function(distances, r_span, width, r_cut, gamma)

    feature = jnp.sum(radial_values, axis=0)

    return feature

def benchmark(frames):
    """"""
    # - 
    r_cut = 6.0
    r_delta = 0.5

    # -
    (
        body2_mapping, body2_features, body2_gradients, 
        body3_mapping, body3_features, body3_gradients
    ) = compute_body3_descriptor(frames, r_cut=r_cut)

    positions = jnp.array(frames[0].positions)
    pair_indices = np.array([x[1:] for x in body2_mapping]).T
    print(pair_indices)

    r_span = jnp.linspace(-1., r_cut+2.0, num=13)[jnp.newaxis, :]

    feature = compute_feature(positions, pair_indices, r_span, width=r_delta, r_cut=r_cut, gamma=2)
    print(feature)

    feature_gradient = jax.jacfwd(compute_feature, argnums=0)(
        positions, pair_indices, r_span, width=r_delta, r_cut=r_cut, gamma=2
    )
    print(feature_gradient.shape)
    feature_drdx = jnp.sum(feature_gradient, axis=0)
    print(feature_drdx)
    print(feature_drdx.shape)

    #test_b2 = jnp.array([[1.28]])
    #print(switch_function(test_b2, r_cut, 2))
    #print(jax.jacfwd(switch_function, argnums=0)(test_b2, r_cut, 2))
    #xxx = jax.jacfwd(radial_function, argnums=0)(test_b2, r_span, r_delta, r_cut, 2)
    #print(xxx)

    return


if __name__ == "__main__":
    ...