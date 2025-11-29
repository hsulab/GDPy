import copy
import pathlib

import numpy as np
import numpy.typing
from ase import Atoms
from dscribe.descriptors import SOAP

from gdpx.data.array import AtomsNDArray
from gdpx.group import evaluate_group_expression

from .clustering import group_structures
from .selector import BaseSelector
from .sparsification import cur_selection, fps_selection


def cleave_structures_by_group(structures: list[Atoms], grp_expr: str) -> tuple[list[Atoms], list[int]]:
    """"""
    new_structures, mapping_indices = [], []
    for i, atoms in enumerate(structures):
        group_indices = evaluate_group_expression(atoms, grp_expr)
        if group_indices:
            cleaved = atoms[group_indices]
            new_structures.append(cleaved)
            mapping_indices.append(i)
        else:
            ...

    return new_structures, mapping_indices


def plot_configuration_map(png_fpath: pathlib.Path, group_features: list) -> None:
    """"""
    import matplotlib.pyplot as plt

    try:
        plt.style.use("presentation")  # type: ignore
    except Exception:
        ...

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    for grp_name, a_x, a_y, s_x, s_y in group_features:
        num_candidates = len(a_x)
        num_selected = len(s_x)
        # Plot all candidates
        _ = ax.scatter(
            a_x,
            a_y,
            marker="o",
            s=100,
            alpha=0.4,
            label=f"grp-{grp_name} {num_candidates} -> {num_selected}",
        )
        # Plot selected ones
        ax.scatter(
            s_x,
            s_y,
            marker="*",
            s=50,
            alpha=0.8,
            color="k",
            facecolor="none",
        )

    ax.legend(fontsize="small")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.savefig(png_fpath, bbox_inches="tight")
    plt.close()

    return


class DescriptorSelector(BaseSelector):
    """Selector using descriptors."""

    name = "dscribe"

    default_parameters = dict(
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
        use_cache=False,
        cluster_group=None,
    )

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        # Verify params
        criteria_method = self.sparsify["method"]
        assert criteria_method in [
            "cur",
            "fps",
        ], f"Unknown selection method {criteria_method}."

        return

    def _compute_descripter(self, frames: list[Atoms]) -> numpy.typing.NDArray:
        """Calculate vector-based descriptors.

        Each structure is represented by a vector.

        """
        features_path = self.directory / "features.npy"

        if not features_path.exists():
            self._print(f"start calculating features with {self.njobs} processes...")
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

            assert isinstance(features, np.ndarray)
            features = features.reshape((-1, ndim))

            # Save calculated features only when we need features for further analysis
            # It is not worthy to do with the number of structures is smaller than 100_000 as
            # it takes a lot of disk space even for a small number.
            if self.use_cache:
                np.save(features_path, features)
        else:
            self._print("load cache features...")
            features = np.load(features_path)

        self._print(f"The shape of features is {features.shape}.")

        return features

    def _mark_structures(self, data: AtomsNDArray) -> None:
        """Mark structures.

        The selected_indices is the local indices for input markers.

        """
        # Group markers
        marker_groups = group_structures(data, group_by=self.group_by)
        self._debug(f"marker_groups: {marker_groups}")

        selected_markers = []
        features, sind_grps, oind_grps = None, {}, {}
        for grp_name, markers in marker_groups.items():
            frames = data.get_marked_structures(markers)

            curr_features, curr_selected_indices = self._select_structures(frames)
            curr_selected_markers = [markers[i] for i in curr_selected_indices]
            selected_markers.extend(curr_selected_markers)

            # Prepare for plotting
            if curr_selected_indices and curr_features is not None:
                if features is None:
                    curr_nframes = 0
                    features = curr_features
                else:
                    curr_nframes = features.shape[0]
                    features = np.vstack((features, curr_features))
                # Selected ones
                sind_grps[grp_name] = [x + curr_nframes for x in curr_selected_indices]
                # Other ones
                oind_grps[grp_name] = [x + curr_nframes for x in range(len(markers))]

        if any([len(v) for _, v in sind_grps.items()]):
            self._plot_results(features, sind_grps, oind_grps)

        data.markers = np.array(selected_markers)

        return

    def _select_structures(self, frames: list[Atoms]) -> tuple[numpy.typing.NDArray | None, list[int]]:
        """"""
        nframes = len(frames)
        num_fixed = self._parse_selection_number(nframes)

        if num_fixed > 0:
            if self.cluster_group is None:
                features = self._compute_descripter(frames)
                if nframes == 1:
                    scores, selected_indices = [np.NaN], [0]
                else:
                    scores, selected_indices = self._sparsify(features, num_fixed)
                    scores = scores[selected_indices]
            else:
                cleaved_frames, mapping_indices = cleave_structures_by_group(frames, self.cluster_group)
                if cleaved_frames:
                    features = self._compute_descripter(cleaved_frames)
                    if len(cleaved_frames) == 1:
                        scores, selected_indices = [np.NaN], mapping_indices
                    else:
                        scores, selected_indices = self._sparsify(features, num_fixed)
                        scores = scores[selected_indices]
                    # Map back to original indices
                    selected_indices = [mapping_indices[i] for i in selected_indices]
                else:
                    features, scores, selected_indices = None, [], []
                num_atoms_in_cluster = len(cleaved_frames[0]) if cleaved_frames else 0
                self._print(
                    f"  number of atoms in the cleaved cluster of structure 0 changed from {len(frames[0])} to {num_atoms_in_cluster}."
                )
        else:
            features, scores, selected_indices = None, [], []

        # Add score to atoms and only save scores from last property
        for score, i in zip(scores, selected_indices):
            frames[i].info["score"] = score

        return features, selected_indices

    def _sparsify(self, features, num_fixed: int):
        """"""
        criteria_params = copy.deepcopy(self.sparsify)
        method = criteria_params.pop("method", "cur")
        if method == "cur":
            scores, selected_indices = cur_selection(features, num_fixed, **criteria_params, rng=self.rng)
        elif method == "fps":
            scores, selected_indices = fps_selection(features, num_fixed, **criteria_params, rng=self.rng)
        else:
            raise Exception(f"Unknown sparsification {method}.")

        return scores, selected_indices

    def _plot_results(self, features, groups: dict, others: dict):
        """Perform PCA and show the configuration map."""
        from sklearn.decomposition import PCA

        if features.shape[0] > 1:
            reducer = PCA(n_components=2)
            reducer.fit(features)
            proj = reducer.transform(features)

            group_features = []
            for grp_name, inds in groups.items():
                selected_proj = reducer.transform(np.array([features[i] for i in inds]))
                group_features.append(
                    [
                        grp_name,
                        proj[others[grp_name], 0],
                        proj[others[grp_name], 1],
                        selected_proj[:, 0],
                        selected_proj[:, 1],
                    ]
                )

            plot_configuration_map(self.info_fpath.parent / (self.info_fpath.stem + ".png"), group_features)
        else:
            ...  # Cannot plot PCA with only one structure...

        return
