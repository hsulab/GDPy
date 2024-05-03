#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

import jax
import jax.numpy as jnp

from ase.io import read, write

from dscribe.descriptors import ValleOganov

#def compute_pcv(r, r_k, coef):
#    """"""
#    num, dim = r_k.shape
#    #k = np.arange(1, 1+num)
#    k = jnp.arange(0, num)
#
#    #r_norm2 = jnp.linalg.norm(r-r_k, axis=1)**2
#    #r_norm2 = jnp.linalg.norm(r-r_k, axis=1)
#    #r_norm2 = jnp.linalg.norm(r-r_k, axis=1)/dim
#    #r_norm2 = jnp.linalg.norm(r-r_k, axis=1)/jnp.sqrt(dim)
#    r_norm2 = jnp.sum(r*r_k, axis=1)/jnp.linalg.norm(r)/jnp.linalg.norm(r_k, axis=1)
#
#    r_exp = jnp.exp(-coef*r_norm2)
#
#    #print(f"r_norm2: {r_norm2}")
#    #print(f"r_exp: {r_exp}")
#
#    s = (jnp.sum(k*r_exp)) / jnp.sum(r_exp)
#    z = -1./coef*jnp.log(jnp.sum(r_exp))
#
#    ret = jnp.hstack([s, z])
#
#    return ret, ret

def compute_pcv(r, r_k, coef):
    """"""
    num, dim = r_k.shape
    #k = np.arange(1, 1+num)
    k = jnp.arange(0, num)

    #r_norm2 = np.linalg.norm(r-r_k, axis=1)**2
    #r_norm2 = np.linalg.norm(r-r_k, axis=1)
    #r_norm2 = np.linalg.norm(r-r_k, axis=1)/dim
    #r_norm2 = np.linalg.norm(r-r_k, axis=1)/np.sqrt(dim)
    #r_norm2 = np.sum((r-r_k)**2, axis=1)/dim**2
    r_norm2 = jnp.sum(r*r_k, axis=1)/jnp.linalg.norm(r)/jnp.linalg.norm(r_k, axis=1)

    r_exp = jnp.exp(-coef*r_norm2)

    #print(f"r_norm2: {r_norm2}")
    #print(f"r_exp: {r_exp}")

    s = (jnp.sum(k*r_exp)) / jnp.sum(r_exp)
    z = -1./coef*jnp.log(jnp.sum(r_exp))

    ret = jnp.hstack([s, z])

    return ret, ret


class FingerprintColvar():

    def __init__(self, data, *args, **kwargs) -> None:
        """"""

        # - 
        self.params = {}

        # - landmarks
        frames_ = read(data, ":")
        frames = [a[240:] for a in frames_]

        _, self.features = self.calculate_features(frames)

        self.coefficient = 20.

        return

    @property
    def dim(self):
        """"""

        return 2

    def calculate_features(self, frames):
        """"""
        vo2 = ValleOganov(
            species = ["Cu"],
            function = "distance",
            sigma = 10**(-5),
            n = 161,
            #n = 17,
            r_cut = 8.,
        )

        vo3 = ValleOganov(
            species = ["Cu"],
            function = "angle",
            sigma = 10**(-5),
            n = 181,
            r_cut = 4.,
        )

        #vo2_features = vo2.create(frames)
        vo2_gradients, vo2_features = vo2.derivatives(frames)
        #print(vo2_features.shape)
        #print(vo2_gradients.shape)

        return vo2_gradients, vo2_features
    
    def cvfunc(self, atoms, params):
        """"""
        gradients, features = self.calculate_features([atoms[240:]])
        features = features[np.newaxis, :]
        gradients = gradients[np.newaxis, :, :, :]
        #print(f"feature shape: {features.shape}")
        #print(f"gradient shape: {gradients.shape}")
        #print(features)
        #print(self.features)

        curr_features = features
        mark_features = self.features

        #curr_features = jnp.array(features, dtype=jnp.float64)
        #mark_features = jnp.array(self.features, dtype=jnp.float64)

        #cv, _ = compute_pcv(curr_features, mark_features, self.coefficient)
        #print(cv)

        cvg, cv = jax.jacfwd(compute_pcv, argnums=0, has_aux=True)(
            curr_features, mark_features, self.coefficient
        )
        cv = cv[jnp.newaxis, :]
        #print(cv)
        #print(cvg)

        return cv


if __name__ == "__main__":
    ...