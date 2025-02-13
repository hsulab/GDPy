#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
from typing import Optional

import ase
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import (  # get system composition (both substrate and top)
    CellBounds,
    closest_distances_generator,
    get_all_atom_types,
)

from gdpx.geometry.composition import CompositionSpace
from gdpx.nodes.region import RegionVariable

from .builder import StructureModifier
from .species import build_species
from .utils import compute_molecule_number_from_density


class RandomBulkBuilder(StructureModifier):

    name: str = "random_bulk"

    #: Number of attempts to create a random candidate.
    MAX_ATTEMPTS_PER_CANDIDATE: int = 1000

    #: Atom numbers of composition to insert.
    composition_atom_numbers: Optional[list[int]] = None

    #: Composition to insert.
    composition_blocks: Optional[dict[str, int]] = None

    def __init__(
        self,
        composition: dict[str, int],
        region: dict = {},
        cell=None,
        cell_volume: Optional[float] = None,
        cell_bounds: Optional[dict] = None,
        cell_splits: Optional[dict] = None,
        pbc: bool = True,
        use_tags: bool = True,
        covalent_ratio=[0.8, 2.0],
        molecular_distances=[None, None],
        test_too_far: bool = True,
        test_dist_to_slab: bool = True,
        max_times_size: int = 10,
        *args,
        **kwargs,
    ):
        """Generate random structures with an ASE built-in method.

        Args:
            max_times_size: Number of attempts to create a number of candidates.
                If 10 structures are to create, run will try 5*10=50 times.
            use_tags: Whether use tags to distinguish molecules.

        """
        super().__init__(*args, **kwargs)

        # Save init params
        _init_params = dict(
            composition=composition,
            region=region,
            cell=cell,
            covalent_ratio=covalent_ratio,
            molecular_distances=molecular_distances,
            max_times_size=max_times_size,
            test_too_far=test_too_far,
            test_dist_to_slab=test_dist_to_slab,
            cell_volume=cell_volume,
            cell_bounds=cell_bounds,
            cell_splits=cell_splits,
            pbc=pbc,
            **kwargs,
        )
        self._init_params = copy.deepcopy(_init_params)

        # Overwrite substrates if it is a file path
        if self._input_substrates is not None:
            self._init_params["substrates"] = self._input_substrates

        # Substrates are not allowed in random bulk.
        self._substrate = None
        if self.substrates is not None:
            if len(self.substrates) != 0:
                raise Exception("The random_bulk does not support substrates.")

        # Set random seed for generators due to compatibility
        if isinstance(self.random_seed, int):
            np.random.seed(self.random_seed)
        elif isinstance(self.random_seed, dict):
            np.random.set_state(self.random_seed)
        else:
            raise Exception(f"Invalid random seed `{self.random_seed}`.")

        # The number of attempts to generate structures
        self.max_times_size = max_times_size

        # Create a region
        self.region = RegionVariable(**region)

        # Check composition
        self._compspec = CompositionSpace(composition)

        self.covalent_ratio = covalent_ratio
        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        self.blmin = self._build_tolerance(
            self._compspec.get_chemical_numbers(), self.covalent_min
        )

        # Some box-related settings
        self.cell = cell

        self.cell_volume = cell_volume
        self.cell_bounds = cell_bounds

        self.cell_splits = cell_splits
        self._converted_cell_splits = None

        self.number_of_variable_cell_vectors = (
            0  # number_of_variable_cell_vectors
        )

        self.test_too_far = test_too_far
        self.test_dist_to_slab = test_dist_to_slab

        self.pbc = pbc
        if not self.pbc:
            raise Exception(
                "The random_bulk does not support non-periodic boundary conditions (pbc=False)."
            )

        # The built-in cut_and_splice will reinit tags from 0 if use_tags is false,
        # here, use_tags is set true no matter what type of system is explored to
        # retain tags information.
        self.use_tags = use_tags
        if not self.use_tags:
            raise Exception("`random_builder` must have use_tags to be True.")

        return

    def run(
        self,
        substrates: Optional[list[Atoms]] = None,
        size: int = 1,
        *args,
        **kwargs,
    ) -> list[Atoms]:
        """Modify input structures.

        Args:
            substrates: Building blocks.
            numbers: Number of each block.

        Returns:
            A list of structures.

        """
        super().run(substrates=substrates, *args, **kwargs)

        if self.substrates is not None:
            raise Exception(
                f"The random_bulk does not support substrates `{self.substrates}`."
            )
        else:
            ...

        # Instantiate the ase generator
        composition = self._compspec._compositions[0]
        generator = self._create_generator(composition)

        # Generate structures
        frames, num_frames, num_attempts = [], 0, 0
        for i in range(size * self.max_times_size):
            num_frames = len(frames)
            if num_frames < size:
                atoms = generator.get_new_candidate(
                    maxiter=self.MAX_ATTEMPTS_PER_CANDIDATE
                )
                if atoms is not None:
                    frames.append(atoms)
                    num_frames += 1
            else:
                num_attempts = i
                break
        else:
            num_attempts = size * self.max_times_size
            self._print(
                f"Failed to create {size} structures after {num_attempts} attempts, only {num_frames} are created."
            )

        # Make tags start with 1 if no substrate is used
        if self._substrate is not None:
            num_atoms_in_substrate = len(self._substrate)
            if num_atoms_in_substrate == 0:
                for atoms in frames:
                    prev_tags = atoms.get_tags()
                    atoms.set_tags(prev_tags + 1)

        return frames

    def _create_generator(
        self, composition: list[tuple[str, int]]
    ) -> StartGenerator:
        """"""
        composition_chemical_numbers = [
            atomic_numbers[s]
            for s in itertools.chain(*[[s] * n for s, n in composition])
        ]

        # check number_of_variable_cell_vectors
        if self.cell is None:
            self.cell = []
        number_of_variable_cell_vectors = 3 - len(self.cell)
        box_to_place_in = None
        if number_of_variable_cell_vectors > 0:
            box_to_place_in = [[0.0, 0.0, 0.0], np.zeros((3, 3))]
            if len(self.cell) > 0:
                box_to_place_in[1][
                    number_of_variable_cell_vectors:
                ] = self.cell
        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors
        self.box_to_place_in = box_to_place_in

        # check volume
        if self.cell_volume is None:
            radii = [
                covalent_radii[x] * self.covalent_max
                for x in composition_chemical_numbers
            ]
            self.cell_volume = np.sum([4 / 3.0 * np.pi * r**3 for r in radii])

        # cell bounds
        if isinstance(self.cell_bounds, dict):
            cell_bounds = {}
            angles = ["a", "b", "c"]
            for k in angles:
                cell_bounds[k] = self.cell_bounds.get(k, [15, 165])
            lengths = ["phi", "chi", "psi"]
            for k in lengths:
                cell_bounds[k] = self.cell_bounds.get(k, [2, 60])
            self.cell_bounds = CellBounds(cell_bounds)
        else:
            assert isinstance(self.cell_bounds, CellBounds)

        # cell splits
        if self.cell_splits is not None:
            splits_ = {}
            for r, p in zip(
                self.cell_splits["repeats"], self.cell_splits["probs"]
            ):
                splits_[tuple(r)] = p
            self._converted_cell_splits = splits_

        generator = StartGenerator(
            Atoms("", pbc=True),
            blocks=composition,
            blmin=self.blmin,
            number_of_variable_cell_vectors=self.number_of_variable_cell_vectors,
            box_to_place_in=self.box_to_place_in,
            box_volume=self.cell_volume,
            splits=self._converted_cell_splits,
            cellbounds=self.cell_bounds,
            test_dist_to_slab=self.test_dist_to_slab,
            test_too_far=self.test_too_far,
            rng=np.random,
        )  # structure generator

        return generator

    def _build_tolerance(
        self, unique_atom_types: list[int], ratio: float = 1.0
    ):
        """"""
        blmin = closest_distances_generator(
            atom_numbers=unique_atom_types,
            # be careful with test too far
            ratio_of_covalent_radii=ratio,
        )

        return blmin

    def _print_blmin(self, blmin):
        """"""
        elements = []
        for k in blmin.keys():
            elements.extend(k)
        elements = set(elements)
        # elements = [ase.data.chemical_symbols[e] for e in set(elements)]
        nelements = len(elements)

        index_map = {}
        for i, e in enumerate(elements):
            index_map[e] = i
        distance_map = np.zeros((nelements, nelements))
        for (i, j), dis in blmin.items():
            distance_map[index_map[i], index_map[j]] = dis

        symbols = [ase.data.chemical_symbols[e] for e in elements]

        content = "Bond Distance Minimum\n"
        content += "  covalent ratio: {}\n".format(self.covalent_min)
        content += (
            "  " + " " * 4 + ("{:>6}  " * nelements).format(*symbols) + "\n"
        )
        for i, s in enumerate(symbols):
            content += "  " + ("{:<4}" + "{:>8.4f}" * nelements + "\n").format(
                s, *list(distance_map[i])
            )
        content += "  too_far: {}, dist_to_slab: {}\n".format(
            self.test_too_far, self.test_dist_to_slab
        )
        content += "  note: default too far tolerance is 2 times\n"

        return content

    def __repr__(self):
        """"""
        content = ""
        content += f"----- {self.__class__.__name__} Parameters -----\n"
        content += f"random_seed: {self.random_seed}\n"

        content += str(self.region)
        if self.blmin is not None:
            content += self._print_blmin(self.blmin)
        else:
            content += "Bond Distance Minimum\n"
            content += "  covalent ratio: {}\n".format(self.covalent_min)

        return content

    def as_dict(self) -> dict:
        """"""
        params = copy.deepcopy(self._init_params)
        params["method"] = self.name

        return params


class RandomClusterBuilder(StructureModifier):

    name: str = "random_cluster"

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        raise Exception("Use `random_structure_improved` instead.")


class RandomSurfaceBuilder(StructureModifier):

    name: str = "random_surface"

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        raise Exception("Use `random_structure_improved` instead.")


if __name__ == "__main__":
    ...
