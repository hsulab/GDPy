#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import warnings

from typing import List, Mapping
from pathlib import Path

import numpy as np

import ase
from ase import Atoms
from ase.data import covalent_radii

from ase.ga.utilities import get_all_atom_types, closest_distances_generator # get system composition (both substrate and top)
from ase.ga.utilities import CellBounds
from ase.ga.startgenerator import StartGenerator

from GDPy.core.register import registers
from GDPy.builder.builder import StructureBuilder 
from GDPy.builder.species import build_species

""" Generate structures randomly
"""


class RandomBuilder(StructureBuilder):

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
        self, composition: Mapping[str,int], substrate: Atoms=None,
        region: dict={}, cell=[], covalent_ratio=[1.0, 2.0], 
        directory="./", random_seed=None, *args, **kwargs
    ):
        super().__init__(directory, random_seed, *args, **kwargs)

        # - parse composition
        self.composition = composition
        self._parse_composition()

        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        # - add substrate
        self.substrate = substrate
        if self.substrate is not None:
            unique_atom_types = get_all_atom_types(self.substrate, self.composition_atom_numbers)
            self.blmin = self._build_tolerance(unique_atom_types)
        else:
            self.blmin = None

        # - create region
        region = copy.deepcopy(region)
        shape = region.pop("method", "auto")
        self.region = registers.create("region", shape, convert_name=True, **region)

        # - check cell
        self.cell = cell
        if not self.cell: # None or []
            self.cell = np.array(cell).reshape(-1,3)

        # - read from kwargs
        self.test_too_far = kwargs.get("test_too_far", True) # test_too_far

        self.test_dist_to_slab = kwargs.get("test_dist_to_slab", True) # test_dist_to_slab

        self.cell_volume = kwargs.get("cell_volume", None)
        self.cell_bounds = kwargs.get("cell_bounds", None)
        self.cell_splits = kwargs.get("cell_splits", None)
        self.number_of_variable_cell_vectors = 0 # number_of_variable_cell_vectors

        return

    def run(self, substrate: Atoms=None, size: int=1, soft_error: bool=False, *args, **kwargs) -> List[Atoms]:
        """Modify input structures.

        Args:
            substrates: Building blocks.
            numbers: Number of each block.
        
        Returns:
            A list of structures.
        
        """
        generator = self._create_generator(substrate)

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
            if soft_error:
                warnings.warn(f"Failed to create {size} structures, only {nframes} are created.", UserWarning)
            else:
                raise RuntimeError(f"Failed to create {size} structures, only {nframes} are created.")
        
        return frames
    
    def _update_settings(self, substarte: Atoms=None):
        """"""

        raise NotImplementedError()

    def _create_generator(self, substrate=None):
        """"""
        self._update_settings(substrate)

        # - create the starting population
        np.random.seed(self.random_seed)

        generator = StartGenerator(
            self.substrate, 
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
        blocks = [(k,v) for k,v in self.composition.items()] # for start generator
        for k, v in blocks:
            species = build_species(k)
            if len(species) > 1:
                self.use_tags = True
                break
        else:
            self.use_tags = False
        self.composition_blocks = blocks

        atom_numbers = [] # atomic number of inserted atoms
        for species, num in self.composition.items():
            numbers = []
            for s, n in ase.formula.Formula(species).count().items():
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

@registers.builder.register
class BulkBuilder(RandomBuilder):

    def _update_settings(self, substarte: Atoms = None):
        """"""
        # - ignore substrate
        self.substrate = Atoms("", pbc=True)

        unique_atom_types = set(self.composition_atom_numbers)
        self.blmin = self._build_tolerance(unique_atom_types)

        # - check number_of_variable_cell_vectors
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


@registers.builder.register
class ClusterBuilder(RandomBuilder):

    def _update_settings(self, substarte: Atoms = None):
        """"""
        # - ignore substrate
        if not self.cell: # None or []
            self.cell = np.array([19.,0.,0.,0.,20.,0.,0.,0.,21.]).reshape(3,3)
        self.substrate = Atoms(cell = self.cell, pbc=False)

        unique_atom_types = set(self.composition_atom_numbers)
        self.blmin = self._build_tolerance(unique_atom_types)
            
        # - ignore lattice parameters
        self.cell_volume = None
        self.cell_splits = None
        self.cell_bounds = None

        self.box_to_place_in = [self.region._origin, self.region._cell]

        return


@registers.builder.register
class SurfaceBuilder(RandomBuilder):
    
    def __init__(
        self, region: dict, composition: Mapping[str,int], cell=[], covalent_ratio=[1.0, 2.0], 
        directory="./", random_seed=None, *args, **kwargs
    ):
        """"""
        super().__init__(
            region=region, composition=composition, cell=cell, covalent_ratio=covalent_ratio,
            directory=directory, random_seed=random_seed, *args, **kwargs
        )

        # - check region
        assert self.region.__class__.__name__ == "LatticeRegion", f"{self.__class__.__name__} needs a LatticeRegion."

        return
    
    def _update_settings(self, substarte: Atoms = None):
        """"""        
        # - use init substrate
        if substarte is not None:
            self.substrate = substarte

        # define the closest distance two atoms of a given species can be to each other
        unique_atom_types = get_all_atom_types(self.substrate, self.composition_atom_numbers)
        self.blmin = self._build_tolerance(unique_atom_types)

        # - ignore lattice parameters
        self.cell_volume = None
        self.cell_splits = None
        self.cell_bounds = None

        self.box_to_place_in = [self.region._origin, self.region._cell]

        return


if __name__ == "__main__":
    ...