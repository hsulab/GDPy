#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from typing import NoReturn, Union, List

import numpy as np

from ase import Atoms
from ase.io import read, write

from dscribe.descriptors import SOAP

from GDPy.core.register import registers
from GDPy.selector.selector import AbstractSelector, group_markers
from GDPy.selector.cur import cur_selection, fps_selection


"""Selector using descriptors.
"""


@registers.selector.register
class DescriptorSelector(AbstractSelector):

    """Selector using descriptors.
    """

    name = "dscribe"

    default_parameters = dict(
        mode = "stru",
        descriptor = None,
        sparsify = dict(
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
        number = [4, 0.2],
        verbose = False
    )

    def __init__(self, directory="./", *args, **kwargs):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        #print(f"random_seed: {self.random_seed}")
        #print(f"random_seed: {self.rng}")
        #print(f"random_seed: {self.rng.bit_generator.state}")

        # - check params
        criteria_method = self.sparsify["method"]
        assert criteria_method in ["cur", "fps"], f"Unknown selection method {criteria_method}."

        assert self.mode in ["stru", "traj"], f"Unknown selection mode {self.mode}."

        return

    def _compute_descripter(self, frames: List[Atoms]) -> np.array:
        """Calculate vector-based descriptors.

        Each structure is represented by a vector.

        """
        features_path = self.directory / "features.npy"

        self._print("start calculating features...")
        desc_params = copy.deepcopy(self.descriptor)
        desc_name = desc_params.pop("name", None)

        features = None
        if desc_name == "soap":
            soap = SOAP(**desc_params)
            ndim = soap.get_number_of_features()
            self._print(f"soap descriptor dimension: {ndim}")
            features = soap.create(frames, n_jobs=self.njobs)
        else:
            raise RuntimeError(f"Unknown descriptor {desc_name}.")
        self._print("finished calculating features...")

        # - save calculated features 
        features = features.reshape(-1,ndim)
        if self.verbose:
            np.save(features_path, features)
            self._print(f"number of soap instances {len(features)}")

        return features

    def _mark_structures(self, data, *args, **kwargs) -> None:
        """Mark structures.

        The selected_indices is the local indices for input markers.

        """
        if self.mode == "stru":
            markers = data.markers
            frames = data.get_marked_structures() # reference of atoms

            # -
            features, selected_indices = self._select_structures(frames)
            if selected_indices:
                self._plot_results(features, selected_indices)
            
            # - update markers
            selected_markers = [markers[i] for i in selected_indices]
            data.markers = selected_markers 
        elif self.mode == "traj":
            # TODO: plot figure...
            for traj in data:
                curr_markers = traj.markers
                #print("curr_markers: ", curr_markers)
                curr_frames = traj.get_marked_structures()
                curr_feactures, curr_indices = self._select_structures(curr_frames)
                #print(f"curr_indices: {curr_indices}")
                new_markers = [curr_markers[i] for i in curr_indices]
                #print("new_markers: ", new_markers)
                traj.markers = new_markers
            #print("markers: ", data.get_unpacked_markers())
        else:
            ...
        
        return
    
    def _select_structures(self, frames: List[Atoms]):
        """"""
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
        else:
            scores, selected_indices = [], []
        
        # - add score to atoms
        #   only save scores from last property
        for score, i in zip(scores, selected_indices):
            frames[i].info["score"] = score
            
        return features, selected_indices
    
    def _sparsify(self, features, num_fixed: int):
        """"""
        # TODO: sparsify each traj separately?
        criteria_params = copy.deepcopy(self.sparsify)
        method = criteria_params.pop("method", "cur")
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

        return scores, selected_indices
    
    def _plot_results(self, features, selected_indices, *args, **kwargs):
        """"""
        # - plot selection
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        try:
            plt.style.use("presentation")
        except Exception as e:
            ...

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
    ...