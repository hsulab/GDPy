import pathlib
from typing import Union

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList

from gdpx.data.array import AtomsNDArray
from gdpx.group import evaluate_group_expression

from .validator import BaseValidator


def compute_connectivities(traj: list[Atoms], group_a: str, group_b: list[str], cutoffs: list[float]):
    """"""
    # Define adsorbates and adsorption sites
    first_atoms = traj[0]
    group_indices_a = evaluate_group_expression(first_atoms, group_a)
    site_groups = [evaluate_group_expression(first_atoms, g) for g in group_b]

    num_site_groups = len(site_groups)

    a_index_map = {a: i for i, a in enumerate(group_indices_a)}
    site_index_maps = [{atom: idx for idx, atom in enumerate(group)} for group in site_groups]

    # Check whether OH stays on Cu-Zn or Zn-Zn sites
    num_atoms = len(traj[0])
    nlist = NeighborList([max(cutoffs) / 2] * num_atoms, skin=0.3, self_interaction=False, bothways=True)

    # Run over the trajectory and compute connectivities
    all_connectivities = []
    for istep, atoms in enumerate(traj):
        connectivities = np.zeros(
            tuple([len(group_indices_a)] + [len(g) for g in site_groups]),
            dtype=int,
        )

        nlist.update(atoms)
        # if istep % 1000 == 0:
        #     self._print(f"---> {istep}")
        for i in group_indices_a:
            indices, offsets = nlist.get_neighbors(i)
            for j, o in zip(indices, offsets):
                for g_idx, index_map in enumerate(site_index_maps):
                    local_idx = index_map.get(j, None)
                    if local_idx is None:
                        continue
                    distance = np.linalg.norm(atoms.positions[i] - (atoms.positions[j] + np.dot(o, atoms.get_cell())))
                    if distance <= cutoffs[g_idx]:
                        idx = [a_index_map[i]] + [slice(None)] * num_site_groups
                        idx[g_idx + 1] = local_idx
                        connectivities[tuple(idx)] += 1
        # Check connectivities, 2 means both Cu/Zn and Zn/Zn sites are connected,
        # 1 means only one of them is connected, 0 means none of them is connected
        # Then we only keep 2 as 1, and others (1 or 0) as 0, to check whether the OH group is connected to both sites or not.
        connectivities[connectivities < num_site_groups] = 0
        connectivities[connectivities == num_site_groups] = 1
        # Print if OH binds on what bridge site
        # for i in group_indices_a:
        #     if np.sum(connectivities[group_indices_a.index(i), :, :]) > 0:
        #         print(f"OH group {i} binds on bridge site:")
        #         # get nonzero indices and map them to real atom indices
        #         nonzero_indices = np.argwhere(connectivities[group_indices_a.index(i), :, :] > 0)
        #         for local_indices in nonzero_indices:
        #             site_atoms = [site_groups[g_idx][local_idx] for g_idx, local_idx in enumerate(local_indices)]
        #             print(f"  {site_atoms}")
        all_connectivities.append(connectivities)
    all_connectivities = np.array(all_connectivities)

    return all_connectivities


class CorrelationValidator(BaseValidator):
    """Estimate the bond correlation."""

    def __init__(
        self,
        group_a: str,
        group_b: list[str],
        cutoff: Union[float, list[float]],
        directory: Union[str, pathlib.Path] = "./",
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        self.group_a = group_a
        self.group_b = group_b

        if isinstance(cutoff, (int, float)):
            self.cutoffs = [float(cutoff)] * len(group_b)
        else:
            self.cutoffs = cutoff

        return

    def run(self, dataset: dict, worker=None, *args, **kwargs) -> bool:
        """"""
        super().run()

        is_finished = True

        # Find some optional parameters
        labels = kwargs.get("labels", None)

        # Process reference and prediction data
        self._print("process reference ->")
        reference = dataset.get("reference")
        if reference is not None:
            self._irun(reference, "ref-", labels)

        self._print("process prediction ->")
        prediction = dataset.get("prediction")
        if prediction is not None:
            self._irun(prediction, "pre-", labels)

        return is_finished

    def _process_data(self, data) -> list[list[Atoms]]:
        """"""
        data = AtomsNDArray(data)
        self._debug(f"data: {data}")

        if data.ndim == 1:
            data = [data.tolist()]
        elif data.ndim == 2:  # assume it is from minimisations...
            data = data.tolist()
        else:
            raise RuntimeError(f"Invalid shape {data.shape}.")

        return data

    def _irun(self, data, prefix, labels):
        """"""
        mdtrajs = self._process_data(data)

        all_connectivities = []
        for i, traj in enumerate(mdtrajs):
            self._print(f"---> traj {i:>02d}")
            traj = [atoms for atoms in traj if atoms is not None]
            self._print(f"    num_frames: {len(traj)}")
            connectivities = compute_connectivities(traj, self.group_a, self.group_b, self.cutoffs)
            all_connectivities.append(connectivities)
        all_connectivities = np.array(all_connectivities)

        np.save(self.directory / "conn.npy", all_connectivities)
        self._print(f"connectivities {all_connectivities.shape}")

        return
