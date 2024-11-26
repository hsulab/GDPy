#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import time
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
    substrate: Atoms,
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
    prepared_substrates = [copy.deepcopy(substrate) for _ in range(max_attempts)]
    prepared_fragments = [
        composition_space.get_fragments_from_one_composition(rng)
        for _ in range(max_attempts)
    ]
    prepared_random_states = rng.integers(
        low=0, high=RANDOM_INTEGER_HIGH, size=max_attempts
    )

    print(f"{n_jobs=}")
    backend = "loky"
    # backend = "threading"
    ret = joblib.Parallel(n_jobs=n_jobs, backend=backend)(
        joblib.delayed(insert_fragments_at_once)(
            substrate=substrate,
            fragments=fragments,
            region=region,
            molecular_distances=molecular_distances,
            covalent_ratio=covalent_ratio,
            bond_distance_dict=bond_distance_dict,
            random_state=random_state,
        )
        for substrate, fragments, random_state in zip(
            prepared_substrates, prepared_fragments, prepared_random_states
        )
    )

    structures = [a for a in ret if a is not None]

    return structures  # type: ignore


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

        # Generate structures
        # PERF: For easy random tasks, use stratified parallel run.
        #       try small_times_size first and increase it if not
        #       enough structures are generated.
        frames = []
        for substrate in self.substrates:
            curr_frames = []
            for i in range(self.MAX_TIMES_SIZE):
                num_curr_frames = len(curr_frames)
                if num_curr_frames == size:
                    break
                max_attempts = self.njobs * 2 ** int(np.log(size - num_curr_frames))
                st = time.time()
                batch_frames = stratified_random_structures(
                    substrate,
                    composition_space=self._compspec,
                    region=self.region,
                    molecular_distances=self.molecular_distances,
                    covalent_ratio=self.covalent_ratio,
                    bond_distance_dict=get_bond_distance_dict(chemical_numbers),
                    max_attempts=max_attempts,
                    n_jobs=self.njobs,
                    rng=self.rng,
                )
                et = time.time()
                self._print(
                    f"stride-{i:>04d} generates {len(batch_frames)} structures with {max_attempts} attempts in {et-st:.2f} seconds."
                )
                for atoms in batch_frames:
                    curr_frames.append(atoms)
                    if len(curr_frames) == size:
                        break
            frames.extend(curr_frames)

        return frames
    
    def as_dict(self) -> dict:
        """"""
        params = copy.deepcopy(self._init_params)

        # Make sure we have correct class name
        params["method"] = self.name

        return params


if __name__ == "__main__":
    ...
