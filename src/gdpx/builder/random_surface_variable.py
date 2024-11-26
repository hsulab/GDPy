#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
from typing import List, Optional

import joblib
import numpy as np
import scipy as sp

from ase import Atoms
from ase.data import atomic_numbers

from ..geometry.composition import CompositionSpace
from ..geometry.insert import (batch_insert_fragments_at_once,
                               insert_fragments_at_once)
from ..geometry.spatial import get_bond_distance_dict
from ..nodes.region import RegionVariable
from .builder import StructureModifier

RANDOM_INTEGER_HIGH: int = 1_000_000_000_000


def stratified_random_structures(
    substrates,
    composition_space,
    region,
    molecular_distances,
    covalent_ratio,
    bond_distance_dict,
    max_attempts: int,
    n_jobs,
    rng,
) -> List[List[Atoms]]:
    """"""
    # prepare inputs for parallel
    batches = []
    for _ in range(max_attempts):
        prepared_substrates = [copy.deepcopy(a) for a in substrates]
        num_prepared_substrates = len(prepared_substrates)

        prepared_fragments = composition_space.get_fragments_from_one_composition(rng)

        prepared_random_states = rng.integers(
            low=0, high=RANDOM_INTEGER_HIGH, size=num_prepared_substrates
        )
        batches.append(
            [prepared_substrates, prepared_fragments, prepared_random_states]
        )

    backend = "loky"
    ret = joblib.Parallel(n_jobs=n_jobs, backend=backend)(
        joblib.delayed(batch_insert_fragments_at_once)(
            substrates=curr_substrates,
            fragments=curr_fragments,
            region=region,
            molecular_distances=molecular_distances,
            covalent_ratio=covalent_ratio,
            bond_distance_dict=bond_distance_dict,
            random_states=curr_random_states,
        )
        for curr_substrates, curr_fragments, curr_random_states in batches
    )

    return ret  # type: ignore


class RandomSurfaceVariableModifier(StructureModifier):

    name: str = "random_surface_variable"

    MAX_TIMES_SIZE: int = 10

    def __init__(
        self,
        composition,
        region,
        covalent_ratio=[0.8, 2.0],
        molecular_distances=[None, None],
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        # Save init params
        self._init_params = dict(
            composition=composition,
            covalent_ratio=covalent_ratio,
            molecular_distances=molecular_distances,
            **kwargs,
        )

        # Check composition
        self._compspec = CompositionSpace(composition)

        self._print(f"{self._compspec._compositions =}")

        # Check region
        self.region = RegionVariable(**region).value

        # Spatial tolerance
        self.covalent_ratio = covalent_ratio

        if molecular_distances[0] is None:
            molecular_distances[0] = -np.inf
        if molecular_distances[1] is None:
            molecular_distances[1] = np.inf
        self.molecular_distances = molecular_distances

        # To compatible with GA engine
        self._substrate = None
        if self.substrates is not None:
            self._substrate = self.substrates[0]

        return

    def run(
        self, substrates: Optional[List[Atoms]] = None, size: int = 1, *args, **kwargs
    ) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        num_substrates = len(self.substrates)

        # Infer chemical species may occur in structures
        chemical_symbols = self._compspec.get_chemical_symbols()
        for substrate in self.substrates:
            chemical_symbols.extend(substrate.get_chemical_symbols())
        chemical_symbols = set(chemical_symbols)
        chemical_numbers = [atomic_numbers[s] for s in chemical_symbols]

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
            num_attempts += curr_max_attempts * num_substrates
            # ret = self._irun(self.substrates, size=curr_max_attempts)
            ret = stratified_random_structures(
                self.substrates,
                composition_space=self._compspec,
                region=self.region,
                molecular_distances=self.molecular_distances,
                covalent_ratio=self.covalent_ratio,
                bond_distance_dict=get_bond_distance_dict(chemical_numbers),
                max_attempts=curr_max_attempts,
                n_jobs=self.njobs,
                rng=self.rng,
            )

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
    
    def as_dict(self) -> dict:
        """"""
        params = copy.deepcopy(self._init_params)

        # Make sure we have correct class name
        params["method"] = self.name

        return params


if __name__ == "__main__":
    ...
