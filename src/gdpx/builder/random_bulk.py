#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
from typing import Optional, Union

import ase
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import (  # get system composition (both substrate and top)
    CellBounds,
    closest_distances_generator,
)

from gdpx.geometry.composition import CompositionSpace
from gdpx.utils.atoms_tags import (
    sort_structures_by_natoms_per_type,
    sort_structures_by_tags,
)

from .builder import StructureModifier


def get_random_cell_params(box_params: dict):
    """"""
    # Check number_of_variable_cell_vectors
    box_cell = np.array(box_params.get("cell", []))
    if box_cell.size not in [0, 3, 6, 9]:
        raise Exception("The cell must be an array with 0, 3, 6 or 9 entries.")
    box_cell = box_cell.reshape(-1, 3)
    box_cell_dim = box_cell.shape[0]

    number_of_variable_cell_vectors = 3 - box_cell_dim

    if number_of_variable_cell_vectors > 0:
        # Get box_to_place_in
        box_to_place_in = [[0.0, 0.0, 0.0], np.zeros((3, 3))]
        if box_cell_dim > 0:
            box_to_place_in[1][number_of_variable_cell_vectors:] = box_cell
        box_to_place_in = box_to_place_in

        # Get cell_bounds
        box_bounds = box_params.get("bounds", {})
        if isinstance(box_bounds, dict):
            cell_bounds = {}
            angles = ["a", "b", "c"]
            for k in angles:
                cell_bounds[k] = box_bounds.get(k, [15, 165])
            lengths = ["phi", "chi", "psi"]
            for k in lengths:
                cell_bounds[k] = box_bounds.get(k, [2, 60])
            cell_bounds = CellBounds(cell_bounds)
        else:
            cell_bounds = box_bounds
        assert isinstance(
            cell_bounds, CellBounds
        ), f"{cell_bounds} is not a CellBounds."

        # Get cell_splits
        box_splits = box_params.get("splits", None)
        if box_splits is not None:
            splits_ = {}
            for r, p in zip(box_splits["repeats"], box_splits["probs"]):
                splits_[tuple(r)] = p
            cell_splits = splits_
        else:
            cell_splits = None

        # Get cell_volume
        cell_volume = box_params.get("volume", None)
    else:
        box_to_place_in = None
        cell_bounds = None
        cell_splits = None
        cell_volume = None

    return (
        number_of_variable_cell_vectors,
        box_to_place_in,
        cell_bounds,
        cell_splits,
        cell_volume,
    )


def get_a_bulk_generator(
    composition: tuple[tuple[str, int]],
    min_bond_distance_dict,
    number_of_variable_cell_vectors: int,
    box_to_place_in,
    cell_bounds: Optional[CellBounds] = None,
    cell_splits: Optional[dict] = None,
    cell_volume: Optional[float] = None,
    atomic_radius_ratio: float = 1.0,
    test_too_far: bool = True,
    rng=np.random,
) -> StartGenerator:
    """"""
    composition_chemical_numbers = [
        atomic_numbers[s]
        for s in itertools.chain(*[[s] * n for s, n in composition])
    ]

    if number_of_variable_cell_vectors == 0:
        # Get the substrate and get random structures in a fixed box
        # similar to the ranomd_structure_improved
        substrate = box_to_place_in[1]
    else:
        # Get the substrate
        substrate = Atoms("", pbc=True)

        # Get the cell volume
        radii = np.array(
            [covalent_radii[x] for x in composition_chemical_numbers]
        )
        regular_volume = np.sum([4 / 3.0 * np.pi * r**3 for r in radii])
        if cell_volume is None:
            cell_volume = regular_volume * (atomic_radius_ratio**3)
        else:
            ...  # Give a warning if the volume is too small?

    generator = StartGenerator(
        substrate,
        blocks=composition,
        blmin=min_bond_distance_dict,
        number_of_variable_cell_vectors=number_of_variable_cell_vectors,
        box_to_place_in=box_to_place_in,
        box_volume=cell_volume,
        splits=cell_splits,
        cellbounds=cell_bounds,
        test_dist_to_slab=False,
        test_too_far=test_too_far,
        rng=rng,
    )

    return generator


class RandomBulkBuilder(StructureModifier):

    name: str = "random_bulk"

    def __init__(
        self,
        composition: dict[str, int],
        region: Optional[dict] = None,
        box: Optional[Union[list, dict]] = None,
        pbc: bool = True,
        use_tags: bool = True,
        covalent_ratio=[0.8, 2.0],
        molecular_distances=[None, None],
        test_too_far: bool = True,
        test_dist_to_slab: bool = True,
        max_times_size: int = 10,
        sort_by_tags: bool = True,
        sort_by_natoms_per_type: bool = True,
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
            box=box,
            pbc=pbc,
            covalent_ratio=covalent_ratio,
            molecular_distances=molecular_distances,
            max_times_size=max_times_size,
            test_too_far=test_too_far,
            test_dist_to_slab=test_dist_to_slab,
            sort_by_tags=sort_by_tags,
            sort_by_natoms_per_type=sort_by_natoms_per_type,
            **kwargs,
        )
        self._init_params = copy.deepcopy(_init_params)

        # Set random seed for generators due to compatibility
        if isinstance(self.random_seed, int):
            np.random.seed(self.random_seed)
        elif isinstance(self.random_seed, dict):
            np.random.set_state(self.random_seed)
        else:
            raise Exception(f"Invalid random seed `{self.random_seed}`.")

        # Overwrite substrates if it is a file path
        if self._input_substrates is not None:
            self._init_params["substrates"] = self._input_substrates

        # To compatible with GA engine,
        # and substrates are not allowed in random bulk.
        self._substrate = None
        if self.substrates is not None:
            if len(self.substrates) != 0:
                raise Exception("The random_bulk does not support substrates.")

        # The built-in cut_and_splice will reinit tags from 0 if use_tags is false,
        # here, use_tags is set true no matter what type of system is explored to
        # retain tags information.
        self.use_tags = use_tags
        if not self.use_tags:
            raise Exception("`random_builder` must have use_tags to be True.")

        # Check composition
        self._compspec = CompositionSpace(composition)

        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        self.blmin = self._build_tolerance(
            self._compspec.get_chemical_numbers(), self.covalent_min
        )

        self.test_too_far = test_too_far
        self.test_dist_to_slab = test_dist_to_slab

        # Some box-related settings
        self.box = box

        # Genetic algorithm bulk crossover needs
        # number_of_variable_cell_vectors and cell_bounds
        box_params = self.box
        if isinstance(box_params, list):
            box_params = dict(cell=box_params)
        elif isinstance(box_params, dict):
            ...
        else:
            raise Exception(
                f"Invalid box `{box_params}` with type `{type(box_params)}`."
            )

        (
            self.number_of_variable_cell_vectors,
            self.box_to_place_in,
            self.cell_bounds,
            self.cell_splits,
            self.cell_volume,
        ) = get_random_cell_params(box_params)

        self.pbc = pbc
        if not self.pbc:
            raise Exception(
                "The random_bulk does not support non-periodic boundary conditions (pbc=False)."
            )

        # Create region
        self.region = region
        if self.region is not None:
            raise Exception("The random_bulk does not accept region.")

        # The number of attempts to generate structures
        self.max_times_size = max_times_size

        #: Number of attempts to create a random candidate.
        self.max_attempts_per_candidate: int = 100

        # Whether we should have a consistent tags
        self.sort_by_tags = sort_by_tags

        # Whether we should have structures order by natoms per elements
        self.sort_by_natoms_per_type = sort_by_natoms_per_type

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
        possible_compositions = self._compspec._compositions

        bulk_generators = []
        for composition in possible_compositions:
            generator = get_a_bulk_generator(
                composition,
                min_bond_distance_dict=self.blmin,
                number_of_variable_cell_vectors=self.number_of_variable_cell_vectors,
                box_to_place_in=self.box_to_place_in,
                cell_bounds=self.cell_bounds,
                cell_splits=self.cell_splits,
                cell_volume=self.cell_volume,
                atomic_radius_ratio=self.covalent_max,
                test_too_far=self.test_too_far,
                rng=np.random,
            )
            bulk_generators.append(generator)
        num_generators = len(bulk_generators)

        max_attempts = size * self.max_times_size
        selected_generator_indices = self.rng.choice(
            num_generators, size=max_attempts, replace=True
        )

        # Generate structures
        frames, num_frames, num_attempts = [], 0, 0
        for i in range(max_attempts):
            num_frames = len(frames)
            if num_frames < size:
                atoms = bulk_generators[
                    selected_generator_indices[i]
                ].get_new_candidate(maxiter=self.max_attempts_per_candidate)
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

        # The ase startgenerator assigns tags from 0
        # while tag=0 is reserved for substrate,
        # so we need to reassign tags to avoid conflicts.
        assert self._substrate is None
        for atoms in frames:
            prev_tags = atoms.get_tags()
            assert prev_tags.min() == 0
            atoms.set_tags(prev_tags + 1)

        # Sort atoms in each structure by tags
        if self.sort_by_natoms_per_type:
            chemical_types = self._compspec.get_chemical_symbols()
            frames = sort_structures_by_natoms_per_type(frames, chemical_types)

        if self.sort_by_tags:
            frames = sort_structures_by_tags(frames)

        return frames

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
        content += "  note: too far tolerance is\n"
        content += "        2 times covalent bond distance\n"

        return content

    def __repr__(self):
        """"""
        content = ""
        content += f"----- {self.__class__.__name__} Parameters -----\n"
        content += f"random_seed: {self.random_seed}\n"

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
