#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib
import warnings

from typing import List, Mapping
from pathlib import Path

import numpy as np

import ase
from ase import units, Atoms
from ase.io import read, write
from ase.data import covalent_radii

from ase.ga.utilities import get_all_atom_types, closest_distances_generator # get system composition (both substrate and top)
from ase.ga.utilities import CellBounds
from ase.ga.startgenerator import StartGenerator

from gdpx.core.register import registers
from gdpx.builder.builder import StructureBuilder, StructureModifier
from gdpx.builder.species import build_species

""" Generate structures randomly
"""

def compute_molecule_number_from_density(molecular_mass, volume, density) -> int:
    """Compute the number of molecules in the region with a given density.

    Args:
        moleculer_mass: unit in g/mol.
        volume: unit in Ang^3.
        density: unit in g/cm^3.
    
    Returns:
        Number of molecules in the region.

    """
    number = (density/molecular_mass) * volume * units._Nav * 1e-24

    return int(number)


class RandomBuilder(StructureModifier):

    #: Number of attempts to create a random candidate.
    MAX_ATTEMPTS_PER_CANDIDATE: int = 1000

    #: Number of attempts to create a number of candidates.
    #       if 10 structures are to create, run will try 5*10=50 times.
    MAX_TIMES_SIZE: int = 5

    #: Whether use tags to distinguish molecules.
    use_tags: bool = False

    #: Atom numbers of composition to insert.
    composition_atom_numbers: List[int] = None

    #: Composition to insert.
    composition_blocks: Mapping[str,int] = None

    def __init__(
        self, composition: Mapping[str,int], substrates = None,
        region: dict={}, cell=None, covalent_ratio=[0.8, 2.0], 
        max_times_size: int=5, *args, **kwargs
    ):
        super().__init__(substrates=substrates, *args, **kwargs)

        # TODO: substrates should also be a Builder Object
        # TODO: if substrates is a ChemiclFormula?
        if isinstance(substrates, str) or isinstance(substrates, pathlib.Path):
            substrates = str(pathlib.Path(substrates).absolute())
        else:
            ...

        self._state_params = dict(
            composition = composition,
            substrates = substrates,
            region = region,
            cell = cell,
            covalent_ratio = covalent_ratio,
            max_times_size = max_times_size,
            test_too_far = kwargs.get("test_too_far", True), # test_too_far
            test_dist_to_slab = kwargs.get("test_dist_to_slab", True), # test_dist_to_slab
            cell_volume = kwargs.get("cell_volume", None),
            cell_bounds = kwargs.get("cell_bounds", None),
            cell_splits = kwargs.get("cell_splits", None),
            random_seed = self.random_seed
        )

        # - set random seed for generators due to compatibility
        if isinstance(self.random_seed, int):
            np.random.seed(self.random_seed)
        elif isinstance(self.random_seed, dict):
            np.random.set_state(self.random_seed)
        else:
            ...
        
        # - 
        self.MAX_TIMES_SIZE = max_times_size

        # - create region
        region = copy.deepcopy(region)
        shape = region.pop("method", "auto")
        self.region = registers.create("region", shape, convert_name=True, **region)

        # - parse composition
        self.composition = composition
        self._parse_composition()

        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        # - add substrate
        self._substrate = None
        if self.substrates is not None:
            self._substrate = self.substrates[0]
        if self._substrate is not None:
            unique_atom_types = get_all_atom_types(self._substrate, self.composition_atom_numbers)
            self.blmin = self._build_tolerance(unique_atom_types)
        else:
            unique_atom_types = set(self.composition_atom_numbers)
            self.blmin = self._build_tolerance(unique_atom_types)

        # - check cell
        self.cell = cell

        # - read from kwargs
        self.test_too_far = kwargs.get("test_too_far", True) # test_too_far
        self.test_dist_to_slab = kwargs.get("test_dist_to_slab", True) # test_dist_to_slab

        self.cell_volume = kwargs.get("cell_volume", None)
        self.cell_bounds = kwargs.get("cell_bounds", None)
        self.cell_splits = kwargs.get("cell_splits", None)
        self.number_of_variable_cell_vectors = 0 # number_of_variable_cell_vectors

        return
    
    def _load_substrates(self, inp_sub) -> List[Atoms]:
        """"""
        substrates = super()._load_substrates(inp_sub)
        if substrates is not None:
            assert len(substrates) == 1, "RandomBuilder only supports one substrate."

        return substrates

    def run(self, substrates: List[Atoms]=None, size: int=1, soft_error: bool=False, *args, **kwargs) -> List[Atoms]:
        """Modify input structures.

        Args:
            substrates: Building blocks.
            numbers: Number of each block.
        
        Returns:
            A list of structures.
        
        """
        super().run(substrates=substrates, *args, **kwargs)

        # - create generator
        generator = self._create_generator(self.substrates)

        # - run over
        frames = []
        for i in range(size*self.MAX_TIMES_SIZE):
            nframes = len(frames)
            if nframes < size:
                atoms = generator.get_new_candidate(maxiter=self.MAX_ATTEMPTS_PER_CANDIDATE)
                if atoms is not None:
                    frames.append(atoms)
                    nframes += 1
            else:
                break
        else:
            if size > 0:
                if soft_error:
                    warnings.warn(f"Failed to create {size} structures, only {nframes} are created.", UserWarning)
                else:
                    raise RuntimeError(f"Failed to create {size} structures, only {nframes} are created.")
            else:
                ...
        
        return frames
    
    def _update_settings(self, substarte: Atoms=None):
        """"""

        raise NotImplementedError()

    def _create_generator(self, substrates: List[Atoms]=None):
        """"""
        if substrates is not None:
            self._update_settings(substrates[0])
        else:
            self._update_settings()

        generator = StartGenerator(
            self._substrate, 
            self.composition_blocks, # blocks
            self.blmin,
            number_of_variable_cell_vectors=self.number_of_variable_cell_vectors,
            box_to_place_in=self.box_to_place_in,
            box_volume=self.cell_volume,
            splits=self.cell_splits,
            cellbounds=self.cell_bounds,
            test_dist_to_slab = self.test_dist_to_slab,
            test_too_far = self.test_too_far,
            rng = np.random
        ) # structure generator

        return generator

    def _parse_composition(self):
        # --- Define the composition of the atoms to optimize ---
        blocks = []
        for k, v in self.composition.items():
            k = build_species(k)
            if isinstance(v, int): # number
                v = v
            else: # string command
                data = v.split()
                if data[0] == "density":
                    v = compute_molecule_number_from_density(
                        np.sum(k.get_masses()), self.region.get_volume(), 
                        density = float(data[1])
                    )
                else:
                    raise RuntimeError(f"Unrecognised composition {k:v}.")
            blocks.append((k,v))
        for k, v in blocks:
            if len(k) > 1:
                self.use_tags = True
                break
        else:
            self.use_tags = False
        self.composition_blocks = blocks

        atom_numbers = [] # atomic number of inserted atoms
        for species, num in self.composition_blocks:
            numbers = []
            for s, n in ase.formula.Formula(species.get_chemical_formula()).count().items():
                numbers.extend([ase.data.atomic_numbers[s]]*n)
            atom_numbers.extend(numbers*num)
        self.composition_atom_numbers = atom_numbers
        
        return

    def _build_tolerance(self, unique_atom_types: List[int]):
        """"""
        blmin = closest_distances_generator(
            atom_numbers=unique_atom_types,
            # be careful with test too far
            ratio_of_covalent_radii = self.covalent_min
        )

        return blmin

    def _print_blmin(self, blmin):
        """"""
        elements = []
        for k in blmin.keys():
            elements.extend(k)
        elements = set(elements)
        #elements = [ase.data.chemical_symbols[e] for e in set(elements)]
        nelements = len(elements)

        index_map = {}
        for i, e in enumerate(elements):
            index_map[e] = i
        distance_map = np.zeros((nelements, nelements))
        for (i, j), dis in blmin.items():
            distance_map[index_map[i], index_map[j]] = dis

        symbols = [ase.data.chemical_symbols[e] for e in elements]

        content =  "Bond Distance Minimum\n"
        content += "  covalent ratio: {}\n".format(self.covalent_min)
        content += "  "+" "*4+("{:>6}  "*nelements).format(*symbols)+"\n"
        for i, s in enumerate(symbols):
            content += "  "+("{:<4}"+"{:>8.4f}"*nelements+"\n").format(s, *list(distance_map[i]))
        content += "  too_far: {}, dist_to_slab: {}\n".format(self.test_too_far, self.test_dist_to_slab)
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


class BulkBuilder(RandomBuilder):

    name: str = "random_bulk"

    def _update_settings(self, substarte: Atoms = None):
        """"""
        # - ignore substrate
        self._substrate = Atoms("", pbc=True)

        unique_atom_types = set(self.composition_atom_numbers)
        self.blmin = self._build_tolerance(unique_atom_types)

        # - check number_of_variable_cell_vectors
        if self.cell is None:
            self.cell = []
        number_of_variable_cell_vectors = 3 - len(self.cell)
        box_to_place_in = None
        if number_of_variable_cell_vectors > 0:
            box_to_place_in = [[0.,0.,0.], np.zeros((3,3))]
            if len(self.cell) > 0:
                box_to_place_in[1][number_of_variable_cell_vectors:] = self.cell
        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors
        self.box_to_place_in = box_to_place_in

        # --- check volume
        if self.cell_volume is None:
            radii = [covalent_radii[x]*self.covalent_max for x in self.composition_atom_numbers]
            self.cell_volume = np.sum([4/3.*np.pi*r**3 for r in radii])

        # --- cell bounds
        cell_bounds = {}
        angles, lengths = ["phi", "chi", "psi"], ["a", "b", "c"]
        for k in angles:
            cell_bounds[k] = self.cell_bounds.get(k, [15, 165])
        for k in lengths:
            cell_bounds[k] = self.cell_bounds.get(k, [2, 60])
        self.cell_bounds = CellBounds(cell_bounds)

        # --- splits
        if self.cell_splits is not None:
            splits_ = {}
            for r, p in zip(self.cell_splits["repeats"], self.cell_splits["probs"]):
                splits_[tuple(r)] = p
            self.cell_splits = splits_

        return
    
    def as_dict(self) -> dict:
        """"""
        params = copy.deepcopy(self._state_params)
        params["method"] = self.name

        return params


class ClusterBuilder(RandomBuilder):

    name: str = "random_cluster"

    def __init__(
        self, composition: Mapping[str, int], substrates=None, region: dict = {}, 
        cell=None, covalent_ratio=[0.8, 2.0], *args, **kwargs
    ):
        """"""
        super().__init__(
            composition, substrates, region, cell, covalent_ratio, *args, **kwargs
        )

        self.pbc = kwargs.get("pbc", False)

        return

    def _update_settings(self, substarte: Atoms = None):
        """"""
        # - ignore substrate
        if self.cell is None: # None or []
            self.cell = np.array([19.,0.,0.,0.,20.,0.,0.,0.,21.]).reshape(3,3)
        else:
            self.cell = np.reshape(self.cell, (-1,3))
        self._substrate = Atoms(cell = self.cell, pbc=self.pbc)

        if self.region.__class__.__name__ == "AutoRegion":
            from .region import LatticeRegion
            self.region = LatticeRegion(origin=np.zeros(3), cell=self.cell.flatten())

        unique_atom_types = set(self.composition_atom_numbers)
        self.blmin = self._build_tolerance(unique_atom_types)
            
        # - ignore lattice parameters
        self.cell_volume = None
        self.cell_splits = None
        self.cell_bounds = None

        self.box_to_place_in = [self.region._origin, self.region._cell]

        return

    def as_dict(self) -> dict:
        """"""
        params = copy.deepcopy(self._state_params)
        params["method"] = self.name
        params["pbc"] = self.pbc

        return params


class SurfaceBuilder(RandomBuilder):

    name: str = "random_surface"
    
    def __init__(
        self, region: dict, composition: Mapping[str,int], cell=[], 
        covalent_ratio=[1.0, 2.0], *args, **kwargs
    ):
        """"""
        super().__init__(
            region=region, composition=composition, cell=cell, 
            covalent_ratio=covalent_ratio, *args, **kwargs
        )

        # - check region
        assert self.region.__class__.__name__ == "LatticeRegion", f"{self.__class__.__name__} needs a LatticeRegion."

        return
    
    def _update_settings(self, substarte: Atoms = None):
        """"""        
        # - use init substrate
        if substarte is not None:
            self._substrate = substarte

        # define the closest distance two atoms of a given species can be to each other
        unique_atom_types = get_all_atom_types(self._substrate, self.composition_atom_numbers)
        self.blmin = self._build_tolerance(unique_atom_types)

        # - ignore lattice parameters
        self.cell_volume = None
        self.cell_splits = None
        self.cell_bounds = None

        self.box_to_place_in = [self.region._origin, self.region._cell]

        return

    def as_dict(self) -> dict:
        """"""
        params = copy.deepcopy(self._state_params)
        params["method"] = self.name

        return params


if __name__ == "__main__":
    ...
