#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import multiprocessing

from typing import Optional, List, Tuple

import numpy as np

import joblib

from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import find_mic

from . import registers
from .builder import StructureModifier
from .utils import (
    check_overlap_neighbour,
    convert_composition_to_list,
    rotate_a_molecule,
)


RANDOM_INTEGER_HIGH: int = 1e16


def insert_species(
    substrate: Atoms,
    composition_list,
    region,
    intermol_dismin,
    covalent_ratio,
    custom_dmin_dict,
    random_state,
) -> Optional[Atoms]:
    """"""
    rng = np.random.Generator(np.random.PCG64(random_state))
    # rng = random_state

    atoms = substrate
    atoms.set_tags(0)

    excluded_pairs = list(itertools.permutations(range(len(atoms)), 2))

    species_list = itertools.chain(
        *[[k for i in range(v)] for k, v in composition_list]
    )
    species_list = sorted(species_list, key=lambda a: a.get_chemical_formula())
    num_species = len(species_list)

    random_positions = region.get_random_positions(size=num_species, rng=rng)
    if num_species > 1:
        pair_positions = np.array(list(itertools.combinations(random_positions, 2)))
        raw_vectors = pair_positions[:, 0, :] - pair_positions[:, 1, :]
        mic_vecs, mic_dis = find_mic(v=raw_vectors, cell=atoms.cell)
        if np.min(mic_dis) >= intermol_dismin:
            is_molecule_valid = True
        else:
            is_molecule_valid = False
    else:
        is_molecule_valid = True

    if is_molecule_valid:
        intra_bonds = []
        for a, p in zip(species_list, random_positions):
            # count number of atoms
            prev_num_atoms = len(atoms)
            curr_num_atoms = prev_num_atoms + len(a)
            intra_bonds.extend(
                list(itertools.permutations(range(prev_num_atoms, curr_num_atoms), 2))
            )
            # rotate and translate
            a = rotate_a_molecule(a, use_com=True, rng=rng)
            # a.translate(p - np.mean(a.positions, axis=0))
            a.translate(p - a.get_center_of_mass())
            a.set_tags(int(np.max(atoms.get_tags()) + 1))
            atoms += a

        excluded_pairs.extend(intra_bonds)
        if not check_overlap_neighbour(
            atoms,
            covalent_ratio=covalent_ratio,
            custom_dmin_dict=custom_dmin_dict,
            excluded_pairs=excluded_pairs,
        ):
            atoms = None
        else:
            ...
    else:
        atoms = None

    return atoms


class InsertModifier(StructureModifier):

    name = "insert"

    def __init__(
        self,
        region,
        composition: dict,
        custom_dmin: list = [],
        covalent_ratio=[0.8, 2.0],
        molecular_distol=[-np.inf, np.inf],
        max_times_size: int = 5,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        # - system definition
        self.region = copy.deepcopy(region)  # TODO: str, dict, or a Region obeject?
        shape = region.pop("method", None)
        self._region = registers.create("region", shape, convert_name=True, **region)

        self.composition = composition
        self._composition_list = convert_composition_to_list(composition, self._region)

        for k, v in self._composition_list:
            if len(k) > 1:
                self._insert_molecule = True
                break
        else:
            self._insert_molecule = False

        self.intermol_dismin = molecular_distol[0]
        self.intermol_dismax = molecular_distol[1]

        custom_dmin_dict = {}
        for i, j, d in custom_dmin:
            s_i, s_j = atomic_numbers[i], atomic_numbers[j]
            custom_dmin_dict[(s_i, s_j)] = d
            custom_dmin_dict[(s_j, s_i)] = d
        self.custom_dmin_dict = custom_dmin_dict

        # - bond distance check
        self.covalent_ratio = covalent_ratio
        self.MAX_TIMES_SIZE = max_times_size

        return

    def run(
        self, substrates: List[Atoms] = None, size: int = 1, *args, **kwargs
    ) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        # If no substrates, create a empty box with the region as its cell TODO:
        _substrates = self.substrates
        if _substrates is None:
            _substrates = [Atoms()]

        num_substrates = len(_substrates)

        self._print(f"{self.njobs =}")

        # PERF: For easy random tasks, use stratified parallel run.
        #       try small_times_size first and increase it if not
        #       enough structures are generated.
        num_attempts = 0
        combined_frames = [[] for _ in range(num_substrates)]
        num_frames = 0
        for i in range(self.MAX_TIMES_SIZE):
            curr_max_attempts = self.njobs * 2 ** int(
                np.log(num_substrates * size - num_frames)
            )
            num_attempts += curr_max_attempts*num_substrates
            ret = self._irun(_substrates, max_attempts=curr_max_attempts)

            for batch_frames in ret:
                for i, atoms in enumerate(batch_frames):
                    if len(combined_frames[i]) < size:
                        if atoms is not None:
                            combined_frames[i].append(atoms)
                    else:
                        ...
                combined_num_frames = [len(cf) for cf in combined_frames]
                if np.all([cnf == size for cnf in combined_num_frames]):
                    break

            combined_num_frames = [len(cf) for cf in combined_frames]
            num_frames = np.sum(combined_num_frames)
            self._print(
                f"Need {size}*{num_substrates} structures and {num_frames} is created in {curr_max_attempts} attempts."
            )
            if np.all([cnf == size for cnf in combined_num_frames]):
                break

        if num_attempts >= RANDOM_INTEGER_HIGH:
            self._print("The random structures may have duplicates.")

        frames = list(itertools.chain(*combined_frames))
        num_frames = len(frames)
        self._print(f"{num_frames =}")

        if num_frames != size * num_substrates:
            raise RuntimeError(
                f"Need {size}*{num_substrates} structures but only {num_frames} is created."
            )

        return frames

    def _irun(self, substrates, max_attempts: int) -> List[List[Atoms]]:
        """"""
        # prepare inputs for parallel
        batches = []
        for _ in range(max_attempts):
            prepared_substrates = [copy.deepcopy(a) for a in substrates]
            # print(f"{prepared_substrates =}")

            num_prepared_substrates = len(prepared_substrates)
            prepared_random_states = self.rng.integers(
                low=0, high=RANDOM_INTEGER_HIGH, size=num_prepared_substrates
            )
            batches.append([prepared_substrates, prepared_random_states])

        # if multiprocessing.get_start_method() != "spawn":
        if True:
            backend = "loky"
            ret = joblib.Parallel(n_jobs=self.njobs, backend=backend)(
                joblib.delayed(insert_species_batch)(
                    substrates=curr_substrates,
                    composition_list=self._composition_list,
                    region=self._region,
                    intermol_dismin=self.intermol_dismin,
                    covalent_ratio=self.covalent_ratio,
                    custom_dmin_dict=self.custom_dmin_dict,
                    random_states=curr_random_states,
                )
                for curr_substrates, curr_random_states in batches
            )
        else:
            raise RuntimeError("multiprocessing.")

        return ret


def insert_species_batch(
    substrates: Atoms,
    composition_list,
    region,
    intermol_dismin,
    covalent_ratio,
    custom_dmin_dict,
    random_states,
):
    """"""
    frames = []
    for atoms, random_state in zip(substrates, random_states):
        # print(atoms)
        new_atoms = insert_species(
            substrate=atoms,
            composition_list=composition_list,
            region=region,
            intermol_dismin=intermol_dismin,
            covalent_ratio=covalent_ratio,
            custom_dmin_dict=custom_dmin_dict,
            random_state=random_state,
        )
        frames.append(new_atoms)

    return frames


if __name__ == "__main__":
    ...
