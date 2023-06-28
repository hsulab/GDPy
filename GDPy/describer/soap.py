#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib
from typing import Optional, List, Mapping

import numpy as np
from sklearn.decomposition import PCA

from ase import Atoms
from ase.io import read, write

from ..core.register import registers
from .describer import AbstractDescriber

#try:
#    from dscribe.descriptors import SOAP
#except Exception as e:
#    print(e)

# NOTE: If there is no dscribe, this class will not be registered.
from dscribe.descriptors import SOAP


@registers.describer.register("soap")
class SoapDescriber(AbstractDescriber):

    cache_features = "features.npy"

    def __init__(self, params, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)

        self.descriptor = copy.deepcopy(params)

        return

    def run(self, dataset, *args, **kwargs):
        """"""
        ...
        self._debug(f"n_jobs: {self.njobs}")

        # - for single system
        frames = dataset[0]._images
        if not (self.directory/self.cache_features).exists():
            features = self._compute_descripter(frames=frames)
            np.save(self.directory/self.cache_features, features)
        else:
            features = np.load(self.directory/self.cache_features)

        return features
        
    def _compute_descripter(self, frames: List[Atoms]) -> np.array:
        """Calculate vector-based descriptors.

        Each structure is represented by a vector.

        """
        self._print("start calculating features...")
        desc_params = copy.deepcopy(self.descriptor)

        soap = SOAP(**desc_params)
        ndim = soap.get_number_of_features()
        self._print(f"soap descriptor dimension: {ndim}")
        features = soap.create(frames, n_jobs=self.njobs)
        self._print("finished calculating features...")

        # - save calculated features 
        features = features.reshape(-1,ndim)

        return features


if __name__ == "__main__":
    ...