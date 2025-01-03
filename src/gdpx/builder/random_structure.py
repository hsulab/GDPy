#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import time
from typing import List, Optional

import joblib
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from gdpx.utils.atoms_tags import reassign_tags_by_species

from ..geometry.composition import CompositionSpace
from ..geometry.insert import insert_fragments_by_step
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
    outer_max_attempts: int,
    inner_max_attempts: int,
    n_jobs,
    rng,
) -> List[List[Atoms]]:
    """"""
    # prepare inputs for parallel
    prepared_substrates = [copy.deepcopy(substrate) for _ in range(outer_max_attempts)]
    prepared_fragments = [
        composition_space.get_fragments_from_one_composition(rng)
        for _ in range(outer_max_attempts)
    ]
    prepared_random_states = rng.integers(
        low=0, high=RANDOM_INTEGER_HIGH, size=outer_max_attempts
    )

    backend = "loky"
    ret = joblib.Parallel(n_jobs=n_jobs, backend=backend)(
        # joblib.delayed(insert_fragments_at_once)(
        joblib.delayed(insert_fragments_by_step)(
            substrate=substrate,
            fragments=fragments,
            region=region,
            molecular_distances=molecular_distances,
            covalent_ratio=covalent_ratio,
            bond_distance_dict=bond_distance_dict,
            random_state=random_state,
            max_attempts=inner_max_attempts
        )
        for substrate, fragments, random_state in zip(
            prepared_substrates, prepared_fragments, prepared_random_states
        )
    )

    structures = [a for a in ret if a is not None]

    return structures  # type: ignore


class RandomStructureImprovedModifier(StructureModifier):

    name: str = "random_structure_improved"

    def __init__(
        self,
        composition,
        region,
        box=None,
        pbc: bool=True,
        use_tags: bool=True,
        covalent_ratio=[0.8, 2.0],
        molecular_distances=[None, None],
        max_times_size: int=10,
        sort_by_tags: bool=True,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        # Save init params
        self._init_params = dict(
            composition=composition,
            region=region,
            box=box,
            pbc=pbc,
            covalent_ratio=covalent_ratio,
            molecular_distances=molecular_distances,
            max_times_size=max_times_size,
            **kwargs,
        )

        # Check composition
        self._compspec = CompositionSpace(composition)

        # Check box
        try:
            if box is not None:
                box = np.array(box)
                if box.size == 3:
                    self.box = np.diag(box)
                else:  # assume it is (3,3)
                    self.box = np.reshape(box, (3,3))
            else:
                self.box = None
        except:
            raise RuntimeError(f"box must be a (3,) or (3,3) array but `{box}` is given.")

        self.pbc = pbc

        # Check region
        self.region = RegionVariable(**region).value

        # Spatial tolerance
        self.covalent_ratio = covalent_ratio

        if molecular_distances[0] is None:
            molecular_distances[0] = -np.inf
        if molecular_distances[1] is None:
            molecular_distances[1] = np.inf
        self.molecular_distances = molecular_distances

        # Attempts
        self.max_times_size = max_times_size

        # To compatible with GA engine
        self.use_tags = use_tags
        if not self.use_tags:
            raise Exception("`random_structure_improved` must have use_tags to be True.")

        self._substrate = None
        if self.substrates is not None:
            self._substrate = self.substrates[0]
        else:
            self._substrate = Atoms("", cell=self.box, pbc=self.pbc)

        # Whether we should have a consistent tags
        self.sort_by_tags = sort_by_tags

        return

    def _infer_chemical_numbers_in_composition_space(self):
        """Infer what chemical numbers may occur based on the composition space and the substrates.

        This is normally used to determine the covalent bond distances.
        """
        chemical_symbols = self._compspec.get_chemical_symbols()
        for substrate in self.substrates:
            chemical_symbols.extend(substrate.get_chemical_symbols())
        chemical_symbols = set(chemical_symbols)
        chemical_numbers = [atomic_numbers[s] for s in chemical_symbols]

        return chemical_numbers

    def get_bond_distance_dict(self, ratio: float=1.0) -> dict:
        """"""
        chemical_numbers = self._infer_chemical_numbers_in_composition_space()
        bond_distance_dict = get_bond_distance_dict(chemical_numbers, ratio=ratio)

        return bond_distance_dict

    def run(
        self, substrates: Optional[List[Atoms]] = None, size: int = 1, *args, **kwargs
    ) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        if self.substrates is not None:
            ...
        else:
            if self.box is None:
                raise RuntimeError(f"box must be set when substrates are not given.")
            self.substrates = [Atoms("", cell=self.box, pbc=self.pbc)]

        # Infer chemical species may occur in structures
        bond_distance_dict = self.get_bond_distance_dict()

        # Generate structures
        # PERF: For easy random tasks, use stratified parallel run.
        #       try small_times_size first and increase it if not
        #       enough structures are generated.
        frames = []
        for isub, substrate in enumerate(self.substrates):
            self._print(f"generating structures based on substrate-{isub:>04d}.")
            curr_frames = []
            for i in range(self.max_times_size):
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
                    bond_distance_dict=bond_distance_dict,
                    outer_max_attempts=max_attempts,
                    inner_max_attempts=100,
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
                        self._print(f"stride-{i:>04d} has already obtained {size} structures.")
                        break
            num_curr_frames = len(curr_frames)
            if num_curr_frames != size:
                raise RuntimeError(f"Need {size} but only {num_curr_frames} are generated.")
            frames.extend(curr_frames)

        # Sort atoms in each structure by tags?
        if self.sort_by_tags:
            new_frames = []
            for atoms in frames:
                new_atoms = reassign_tags_by_species(atoms)
                new_frames.append(new_atoms)
            frames = new_frames

        return frames

    def as_dict(self) -> dict:
        """"""
        params = copy.deepcopy(self._init_params)

        # Make sure we have correct class name
        params["method"] = self.name

        return params


if __name__ == "__main__":
    ...
