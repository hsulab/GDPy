#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

from ase.io import read, write

from dscribe.descriptors import ValleOganov


class RmsdColvar():

    def __init__(self, data) -> None:
        """"""

        self.params = {}

        # - landmarks
        frames_ = read(data, ":")
        frames = [a[240:] for a in frames_]

        _, self.features = self.calculate_features(frames)

        return
    
    @property
    def dim(self):
        """"""

        return 1

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
        features = features[np.newaxis, :] # (1, feature_dimension)
        gradients = gradients[np.newaxis, :, :, :]

        #cv = 

        return



if __name__ == "__main__":
    ...