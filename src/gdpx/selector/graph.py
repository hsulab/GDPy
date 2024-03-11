#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List, NoReturn

import numpy as np
import networkx as nx

from joblib import delayed, Parallel

from ase import Atoms

from ..builder.group import create_an_intersect_group
from ..graph.creator import StruGraphCreator
from ..graph.comparison import paragroup_unique_chem_envs
from ..utils.command import CustomTimer

from .selector import AbstractSelector


def single_create_structure_graph(
    atoms: Atoms, graph_params: dict, group_commands: List[str] = None
) -> List[nx.Graph]:
    """Create structure graph and get selected chemical environments.

    Find atoms with selected chemical symbols or in the defined region.

    Args:
        graph_params: Parameters for the graph representation.
        spec_params: Selected criteria.
        atoms: Input structure.

    Returns:
        A list of graphs that represent the chemical environments of selected atoms.

    """
    stru_creator = StruGraphCreator(**graph_params)

    # - find atoms whose environments need to extract
    ads_indices = create_an_intersect_group(atoms, group_commands)

    _ = stru_creator.generate_graph(atoms, ads_indices_=ads_indices)

    chem_envs = stru_creator.extract_chem_envs(atoms)

    return chem_envs


class GraphSelector(AbstractSelector):

    name = "graph"

    default_parameters = dict(
        group_commands=None,
        pbc_grid=[2, 2, 0],
        graph_radius=2,
        neigh_params=dict(covalent_ratio=1.1, skin=0.0),
    )

    def __init__(self, directory="./", *args, **kwargs):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        # -- check params
        if self.group_commands is None:
            raise RuntimeError(f"{self.name} selector need group_commands.")

        return

    def _select_indices(self, frames: List[Atoms], *args, **kwargs) -> List[int]:
        """"""
        nframes = len(frames)

        selected_indices = []

        graph_params = copy.deepcopy(
            dict(
                adsorbate_elements=[],
                pbc_grid=self.pbc_grid,
                graph_radius=self.graph_radius,
                neigh_params=self.neigh_params,
            )
        )
        group_commands = copy.deepcopy(self.group_commands)

        # - create graphs
        with CustomTimer(name="create-graphs", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_create_structure_graph)(a, graph_params, group_commands)
                for a in frames
            )

        # - check if the ret is empty
        ret_envs = []
        for x in ret:
            ret_envs.extend(x)

        if ret_envs:
            self._print("Typical Chemical Environment " + str(ret_envs[0]))
            ret_env_groups = ret
            with CustomTimer(name="check-uniqueness", func=self._print):
                unique_envs, unique_groups = paragroup_unique_chem_envs(
                    ret_env_groups,
                    list(enumerate(frames)),
                    directory=self.directory,
                    n_jobs=1,
                )
            selected_indices = [x[0][0] for x in unique_groups]
        else:
            self._print("Cant find valid atoms for comparison...")
            selected_indices = list(range(nframes))

        self._write_results(frames, selected_indices, unique_groups=unique_groups)

        return selected_indices

    def _write_results(self, frames, selected_indices, *args, **kwargs) -> NoReturn:
        """Write selection results into file that can be used for restart."""
        data = []
        for s in selected_indices:
            atoms = frames[s]
            # - gather info
            confid = atoms.info.get("confid", -1)
            natoms = len(atoms)
            ae = atoms.get_potential_energy() / natoms
            maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            data.append([s, confid, natoms, ae, maxforce])

        # NOTE: output index and confid maybe inconsistent since the frames are
        #       sorted in advance...
        if data:
            np.savetxt(
                self.info_fpath,
                data,
                fmt="%8d  %8d  %8d  %12.4f  %12.4f",
                # fmt="{:>8d}  {:>8d}  {:>8d}  {:>12.4f}  {:>12.4f}",
                header="{:>6s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}".format(
                    *"index confid natoms AtomicEnergy MaxForce".split()
                ),
            )
        else:
            np.savetxt(
                self.info_fpath,
                [[np.NaN]] * 5,
                header="{:>6s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}".format(
                    *"index confid natoms AtomicEnergy MaxForce".split()
                ),
            )

        # -- unique info
        unique_groups = kwargs.get("unique_groups", None)
        if unique_groups is not None:
            unique_data = []
            for i, x in enumerate(unique_groups):
                data = ["ug" + str(i)]
                data.extend([a[0] for a in x])
                unique_data.append(data)
            content = "# unique, indices\n"
            content += "# ncandidates {}\n".format(len(selected_indices))
            for d in unique_data:
                content += ("{:<8s}  " + "{:<8d}  " * (len(d) - 1) + "\n").format(*d)

            unique_info_path = self.info_fpath.parent / (
                self.info_fpath.stem + "-extra.txt"
            )
            with open(unique_info_path, "w") as fopen:
                fopen.write(content)

        return


if __name__ == "__main__":
    ...
