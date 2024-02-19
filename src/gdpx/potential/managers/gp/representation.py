#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import itertools

import numpy as np

from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs

from .gptools import compute_body3_descriptor
from .bench import benchmark


def switch_function(r, r_cut, gamma=2):
    """"""
    pr = r/r_cut

    return 1. - (1.+gamma)*pr**gamma + gamma*pr**(1+gamma)

def switch_function_gradient(r, r_cut, gamma=2):
    """"""
    pr = r/r_cut

    return (-(1.+gamma)*gamma*pr**(gamma-1) + gamma*(1+gamma)*pr**gamma)/r_cut

def switch_function_value_and_grad(r, r_cut, gamma=2):
    """"""
    pr = r/r_cut

    return (
        1. - (1.+gamma)*pr**gamma + gamma*pr**(1+gamma),
        (-(1.+gamma)*gamma*pr**(gamma-1) + gamma*(1+gamma)*pr**gamma)/r_cut
    )


def radial_function(r, r_span, width, r_cut, gamma):
    """Compute radial function.

    Args:
        r:
        r_span: shape (1, num_span)
        f_cut: shape (num_r, 1)

    """
    constant = 1./(r_cut/r_span.shape[1])
    f_cut = switch_function(r, r_cut, gamma)

    return constant*1./r**2*f_cut*np.exp(-(r-r_span)**2/2/width**2)


def radial_function_gradient(r, r_span, width, r_cut):
    """Compute radial function.

    Args:
        r:
        r_span: shape (1, num_span)
        f_cut: shape (num_r, 1)

    """
    constant = 1./(r_cut/r_span.shape[1])

    f_cut, fg_cut = switch_function_value_and_grad(r, r_cut) # r_ij

    v = np.exp(-(r-r_span)**2/2/width**2)

    x = (
        1./r**2*f_cut*(-(r-r_span)/width**2)*v +
        (-2./r**3*f_cut+1./r**2*fg_cut)*v
    )

    return constant*x

def angular_function():
    """"""

    return

def covariance_function(rho1, rho2):
    """"""

    return


def compute_feature():
    """"""
    radial = radial_function(body2_features, r_span, r_delta, r_cut, 2)
    print(radial.shape)
    print(np.sum(radial, axis=0))

    radial_grad = radial_function_gradient(body2_features, r_span, r_delta, r_cut)

    #test_b2 = np.array([[1.28]])
    #radial_grad = radial_function_gradient(test_b2, r_span, r_delta, r_cut)
    print(radial_grad.shape)

    natoms = 4
    radial_gcart = np.zeros((natoms, 3))
    for dimer_index, gcart, gradial in zip(body2_mapping, body2_gradients, radial_grad):
        i, j = dimer_index[1], dimer_index[2]
        radial_gcart[i, :] += np.sum(gradial[:, np.newaxis]*gcart[:, 0:3], axis=0)
        radial_gcart[j, :] += np.sum(gradial[:, np.newaxis]*gcart[:, 3:6], axis=0)
    print(radial_gcart)

    return


def train():
    """"""
    frames = read("./md/Cu4/cand1/traj.xyz", ":1")
    nframes = len(frames)

    print("!!! JAX !!!")
    benchmark(frames)

    print("!!! MANUAL !!!")

    # - 
    r_cut = 6.0
    r_delta = 0.5

    # -
    (
        body2_mapping, body2_features, body2_gradients, 
        body3_mapping, body3_features, body3_gradients
    ) = compute_body3_descriptor(frames, r_cut=r_cut)

    # - b2
    print("- BODY-2 -")
    #print(body2_features)
    print(body2_mapping)

    #r_span = np.linspace(0., r_cut, num=13, endpoint=True)[np.newaxis, :]
    r_span = np.linspace(-1., r_cut+2.0, num=13)[np.newaxis, :]
    num_dim = r_span.shape[1]

    # -
    radial = radial_function(body2_features, r_span, r_delta, r_cut, 2)

    features = np.zeros((nframes, num_dim))
    for dimer_index, v in zip(body2_mapping, radial):
        fi = dimer_index[0]
        features[fi, :] += v
    print(features)

    return


if __name__ == "__main__":
    ...