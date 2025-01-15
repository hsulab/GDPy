#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools

import numpy as np
import numpy.typing
from ase import Atoms
from dscribe.descriptors import SOAP

from .cur import cur_selection, fps_selection
from .selector import AbstractSelector

"""Selector using descriptors.
"""


class DescriptorSelector(AbstractSelector):
    """Selector using descriptors."""

    name = "dscribe"

    default_parameters = dict(
        mode="stru",
        descriptor=None,
        sparsify=dict(
            # -- cur
            method="cur",
            zeta="-1",
            strategy="descent",
            # -- fps
            # method = "fps",
            # min_distance = 0.1,
            # metric = "euclidean",
            # metric_params = {}
        ),
        number=[4, 0.2],
        verbose=False,
    )

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        # - check params
        criteria_method = self.sparsify["method"]
        assert criteria_method in [
            "cur",
            "fps",
        ], f"Unknown selection method {criteria_method}."

        assert self.mode in ["stru", "traj"], f"Unknown selection mode {self.mode}."

        return

    def _compute_descripter(self, frames: list[Atoms]) -> numpy.typing.NDArray:
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
        features = features.reshape(-1, ndim)
        if self.verbose:
            np.save(features_path, features)
            self._print(f"number of soap instances {len(features)}")

        return features

    def _mark_structures(self, data, *args, **kwargs) -> None:
        """Mark structures.

        The selected_indices is the local indices for input markers.

        """
        # - group markers
        if self.axis is None:
            marker_groups = dict(all=data.markers)
        else:
            marker_groups = {}
            for k, v in itertools.groupby(data.markers, key=lambda x: x[self.axis]):
                if k in marker_groups:
                    marker_groups[k].extend(list(v))
                else:
                    marker_groups[k] = list(v)
        self._debug(f"marker_groups: {marker_groups}")

        selected_markers = []
        features, sind_grps, oind_grps = None, {}, {}
        for grp_name, markers in marker_groups.items():
            frames = data.get_marked_structures(markers)  # reference of atoms

            # -
            curr_features, curr_selected_indices = self._select_structures(frames)
            # - update markers
            curr_selected_markers = [markers[i] for i in curr_selected_indices]
            selected_markers.extend(curr_selected_markers)

            # - prepare for plot
            if curr_selected_indices:
                if features is None:
                    curr_nframes = 0
                    features = curr_features
                else:
                    curr_nframes = features.shape[0]
                    features = np.vstack((features, curr_features))
                # - selected ones
                sind_grps[grp_name] = [x + curr_nframes for x in curr_selected_indices]
                # - other ones
                oind_grps[grp_name] = [x + curr_nframes for x in range(len(markers))]

        if any([len(v) for k, v in sind_grps.items()]):
            self._plot_results(features, sind_grps, oind_grps)

        data.markers = np.array(selected_markers)

        return

    def _select_structures(self, frames: list[Atoms]):
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
                scores = scores[selected_indices]
        else:
            features, scores, selected_indices = None, [], []

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
                features, num_fixed, **criteria_params, rng=self.rng
            )
        elif method == "fps":
            scores, selected_indices = fps_selection(
                features, num_fixed, **criteria_params, rng=self.rng
            )
        else:
            ...

        return scores, selected_indices

    def _plot_results(self, features, groups: dict, others: dict, *args, **kwargs):
        """"""
        # - plot selection
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        try:
            plt.style.use("presentation")
        except Exception:
            ...

        if features.shape[0] > 1:
            reducer = PCA(n_components=2)
            reducer.fit(features)
            proj = reducer.transform(features)

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            for grp_name, inds in groups.items():
                sc = ax.scatter(
                    proj[others[grp_name], 0],
                    proj[others[grp_name], 1],
                    marker="o",
                    alpha=0.25,
                    label=f"grp-{grp_name} {len(others[grp_name])} -> {len(inds)}",
                )
                # --
                selected_proj = reducer.transform(np.array([features[i] for i in inds]))
                ax.scatter(
                    selected_proj[:, 0],
                    selected_proj[:, 1],
                    marker="*",
                    alpha=0.5,
                    color="r",
                )
            ax.legend(fontsize=12)
            ax.axis("off")
            fig.savefig(self.info_fpath.parent / (self.info_fpath.stem + ".png"))
            plt.close()
        else:
            ...  # Cannot plot PCA with only one structure...

        return


if __name__ == "__main__":
    ...
