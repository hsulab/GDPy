#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools


import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use("presentation")
except Exception as e:
    ...

from .selector import AbstractSelector
from ..utils.geometry import wrap_traj


class BasinSelector(AbstractSelector):

    name: str = "basin"

    default_parameters: dict = dict(dispintv=0.4)  # angstrom per atom

    def __init__(self, directory="./", axis=None, *args, **kwargs) -> None:
        """"""
        super().__init__(directory, axis, *args, **kwargs)

        return

    def _mark_structures(self, data, *args, **kwargs) -> None:
        """"""
        super()._mark_structures(data, *args, **kwargs)

        print(f"data: {data}")

        if self.axis is None:
            axis = 0
        else:
            axis = self.axis

        if data.ndim >= 1:
            # assume a single trajectory
            structures = data.get_marked_structures()
            nstructures = len(structures)

            disps = self._compute_rmsd_matrix(structures)

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            c = ax.imshow(disps, cmap="hot", interpolation="nearest")
            fig.colorbar(c)
            plt.savefig(self.directory / "rmsd.png")

            # --
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.set_title("Root Mean Squared Displacement")

            ax.scatter(range(nstructures), disps[0], c="b")
            ax.set_xlabel("number")
            ax.set_ylabel("RMSE [Å]")
            ax.tick_params(axis="y", labelcolor="b")

            energies = [a.get_potential_energy() for a in structures]
            ax2 = ax.twinx()
            ax2.scatter(range(nstructures), energies, c="r")
            ax2.set_ylabel("Potential Energy [eV]")
            ax2.tick_params(axis="y", labelcolor="r")

            plt.savefig(self.directory / "disp.png")

            # -- select
            curr_index = 0
            selected_indices = [curr_index]
            for i in range(nstructures):
                if i == curr_index:
                    for j in range(i, nstructures):
                        if np.all(disps[selected_indices, j] > self.dispintv):
                            curr_index = j
                            selected_indices.append(curr_index)
                        else:
                            ...
                else:
                    continue
            print(f"selected_indices: {selected_indices}")

            curr_markers = data.markers
            selected_markers = [curr_markers[i] for i in selected_indices]
            data.markers = selected_markers
        else:
            raise RuntimeError(
                f"Basin does not support array dimension with {data.ndim}"
            )

        return

    def _compute_rmsd_matrix(self, structures):
        """"""
        # --
        structures = wrap_traj(structures)  # TODO: copy atoms?
        nstructures, natoms = len(structures), len(structures[0])

        # --
        pairs = itertools.product(
            *[range(nstructures), range(nstructures)]
        )  # [[0, 0], [0, 1]]
        pairs = np.array(list(pairs)).T

        posmat = np.array([a.positions.flatten() for a in structures])
        disps = np.sqrt(
            np.sum(
                (np.take(posmat, pairs[0], axis=0) - np.take(posmat, pairs[1], axis=0))
                ** 2,
                axis=1,
            )
            / natoms
        )
        # print(disps.shape)
        disps = np.reshape(disps, (nstructures, nstructures))
        # print(disps.shape)

        return disps


if __name__ == "__main__":
    ...
