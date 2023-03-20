#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from typing import NoReturn, Union, List

import numpy as np

from ase import Atoms
from ase.io import read, write

from dscribe.descriptors import SOAP

from GDPy.selector.selector import AbstractSelector
from GDPy.selector.cur import cur_selection, fps_selection


"""Selector using descriptors.
"""


class DescriptorBasedSelector(AbstractSelector):

    """Selector using descriptors.
    """

    name = "dscribe"

    default_parameters = dict(
        random_seed = None,
        descriptor = None,
        sparsify = dict(
            trajwise = False,
            # -- cur
            method = "cur",
            zeta = "-1",
            strategy = "descent"
            # -- fps
            # method = "fps",
            #min_distance = 0.1,
            #metric = "euclidean",
            #metric_params = {}
        ),
        number = [4, 0.2]
    )

    verbose = False #: output verbosity

    def __init__(self, directory="./", *args, **kwargs):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        # - check params
        criteria_method = self.sparsify["method"]
        assert criteria_method in ["cur", "fps"], f"Unknown selection method {criteria_method}."

        assert self._inp_fmt in ["stru", "traj"], f"{self._inp_fmt} is not allowed for {self.name}."

        return

    def _compute_descripter(self, frames: List[Atoms]):
        """Calculate vector-based descriptors.

        Each structure is represented by a vector.

        """
        features_path = self.directory / "features.npy"

        self.pfunc("start calculating features...")
        desc_params = copy.deepcopy(self.descriptor)
        desc_name = desc_params.pop("name", None)

        features = None
        if desc_name == "soap":
            soap = SOAP(**desc_params)
            ndim = soap.get_number_of_features()
            self.pfunc(f"soap descriptor dimension: {ndim}")
            features = soap.create(frames, n_jobs=self.njobs)
        else:
            raise RuntimeError(f"Unknown descriptor {desc_name}.")
        self.pfunc("finished calculating features...")

        # - save calculated features 
        features = features.reshape(-1,ndim)
        if self.verbose:
            np.save(features_path, features)
            self.pfunc(f"number of soap instances {len(features)}")

        return features

    def _select_indices(self, frames: List[Atoms], *args, **kwargs):
        """Return selected indices."""
        nframes = len(frames)
        num_fixed = self._parse_selection_number(nframes)

        # NOTE: currently, only cur and fps are supported
        # TODO: clustering ...
        if num_fixed > 0:
            features = self._compute_descripter(frames)
            if nframes == 1:
                scores, selected_indices = [np.NaN], [0]
            else:
                scores, selected_indices = self._sparsify(features, num_fixed)
            self._plot_results(features, selected_indices)
        else:
            scores, selected_indices = [], []
        
        # - add score to atoms
        #   only save scores from last property
        for score, i in zip(scores, selected_indices):
            frames[i].info["score"] = score

        return selected_indices
    
    def _sparsify(self, features, num_fixed: int):
        """"""
        # TODO: sparsify each traj separately?
        criteria_params = copy.deepcopy(self.sparsify)
        method = criteria_params.pop("method", "cur")
        is_trajwise = criteria_params.pop("trajwise", False)
        if not is_trajwise:
            if method == "cur":
                # -- cur decomposition
                scores, selected_indices = cur_selection(
                    features, num_fixed, **criteria_params, rng = self.rng
                )
            elif method == "fps":
                scores, selected_indices = fps_selection(
                    features, num_fixed, **criteria_params, rng=self.rng
                )
            else:
                ...
        else:
            raise NotImplementedError("Can't sparsify each trajectory separately.")

        return scores, selected_indices
    
    def _plot_results(self, features, selected_indices, *args, **kwargs):
        """"""
        # - plot selection
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        reducer = PCA(n_components=2)
        reducer.fit(features)
        proj = reducer.transform(features)
        selected_proj = reducer.transform(
            np.array([features[i] for i in selected_indices])
        )

        plt.scatter(proj[:,0], proj[:,1], label="ALL")
        plt.scatter(selected_proj[:,0], selected_proj[:,1], label="SEL")
        plt.legend()
        plt.axis("off")
        plt.savefig(self.info_fpath.parent/(self.info_fpath.stem+".png"))

        return


if __name__ == "__main__":
    pass
