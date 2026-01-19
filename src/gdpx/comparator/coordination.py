from typing import Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from joblib import Parallel, delayed

from gdpx.group import evaluate_group_expression
from gdpx.utils.profiler import CustomTimer

from .comparator import BaseComparator


def switch_function(distances, r_cut, r_shift=0.0, nn: int = 6, mm: Optional[int] = None):
    """"""
    nn = 6 if nn is None else nn
    mm = nn * 2 if mm is None else mm

    scaled_distances = (distances - r_shift) / r_cut

    return (1 - scaled_distances**nn) / (1 - scaled_distances**mm)


def compute_coordition_number_by_neighbour(
    nl, cell, positions, indices_a, indices_b, r_cut, nn: Optional[int] = None, mm: Optional[int] = None
):
    """"""
    # For the two same groups, the distance matrix may not be square if self-inetraction is considered
    # since one atom can interact with several itselves under periodic boundary condition.
    dmat = []
    for i in indices_a:
        distances = []
        nei_indices, nei_offsets = nl.get_neighbors(i)
        for j, offset in zip(nei_indices, nei_offsets):
            if j in indices_b:
                distance = np.linalg.norm(positions[i, :] - positions[j, :] + np.dot(offset, cell))
                if 1e-8 < distance <= r_cut:
                    distances.append(distance)
            else:
                ...
        dmat.append(distances)

    assert nn is not None

    coordination = []
    for distances in dmat:
        curr_coordination = np.sum(switch_function(np.array(distances), r_cut=r_cut, nn=nn, mm=mm))
        coordination.append(curr_coordination)
    coordination = np.array(coordination)

    return coordination


class CoordinationComparator(BaseComparator):
    def __init__(self, pairs=None, acn_rmse: float = 0.02, acn_max: float = 0.20, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        # - check pairs that should be a List of 2-element Lists
        pairs = pairs if pairs is not None else []
        npairs = len(pairs)
        assert npairs > 0, "Need at least 1 pair."
        pairs_ = []
        for i, pair in enumerate(pairs):
            ngroups = len(pair)
            assert ngroups >= 2, f"Need at least two groups to define the pair {i}."
            pairs_.append(pair + [None] * (5 - ngroups))
        self.pairs = pairs_

        self.acn_rmse = acn_rmse
        self.acn_max = acn_max

        return

    @staticmethod
    def _process_single_structure(atoms, pair_info):
        """"""
        natoms = len(atoms)
        max_r_cut = np.max([pair[2] for pair in pair_info])
        nl = NeighborList(
            cutoffs=[max_r_cut / 2.0] * natoms,
            skin=0.2,
            sorted=False,
            self_interaction=True,
            bothways=True,
        )
        nl.update(atoms)
        coordination = []
        for group_a, group_b, r_cut, nn, mm in pair_info:
            indices_a = evaluate_group_expression(atoms, group_a)
            indices_b = evaluate_group_expression(atoms, group_b)
            curr_coordination = compute_coordition_number_by_neighbour(
                nl, atoms.cell, atoms.positions, indices_a, indices_b, r_cut, nn, mm
            )
            coordination.extend(sorted(curr_coordination.tolist()))

        return coordination

    def prepare_data(self, frames: list[Atoms]):
        """"""
        with CustomTimer(name="coordination", func=self._debug):
            coordinations = Parallel(n_jobs=self.njobs)(
                delayed(self._process_single_structure)(atoms, self.pairs) for atoms in frames
            )

        coordinations = np.array(coordinations)

        return coordinations

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        """"""
        coordinations = self.prepare_data([a1, a2])
        fp1, fp2 = coordinations
        is_similar = self.compare_fingerprints(fp1, fp2)

        return is_similar

    def compare_fingerprints(self, fp1, fp2) -> bool:
        """"""
        fp1, fp2 = np.array(fp1), np.array(fp2)

        assert fp1.shape == fp2.shape, f"{fp1.shape} vs. {fp2.shape} ..."
        nfeatures = fp1.shape[0]

        is_similar = False

        error = fp1 - fp2
        rmse = np.sqrt(np.sum(error**2) / nfeatures)
        if rmse <= self.acn_rmse and np.max(np.fabs(error)) <= self.acn_max:
            is_similar = True

        return is_similar
