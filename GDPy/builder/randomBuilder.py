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
from ase.io import read, write

from ase.ga.utilities import get_all_atom_types, closest_distances_generator # get system composition (both substrate and top)
from ase.ga.utilities import CellBounds
from ase.ga.startgenerator import StartGenerator

from GDPy.core.register import registers
from GDPy.builder.builder import StructureBuilder 
from GDPy.builder.species import build_species

""" Generate structures randomly
"""


class RandomGenerator(StructureBuilder):

    supported_systems = ["bulk", "cluster", "surface"]

    MAX_FAILED = 10
    MAX_RANDOM_TRY = 100

    def __init__(self, params, directory=Path.cwd()):
        """"""
        super().__init__(directory)

        self.params = copy.deepcopy(params)
        self.generator = self._create_generator(self.params)

        return
    
    def _create_generator(self, params) -> None:
        """ create a random structure generator
        """
        # - parse composition
        # --- Define the composition of the atoms to optimize ---
        composition = params["composition"] # number of inserted atoms
        blocks = [(k,v) for k,v in composition.items()] # for start generator
        for k, v in blocks:
            species = build_species(k)
            if len(species) > 1:
                use_tags = True
                break
        else:
            use_tags = False
            #print("Perform atomic search...")

        atom_numbers = [] # atomic number of inserted atoms
        for species, num in composition.items():
            numbers = []
            for s, n in ase.formula.Formula(species).count().items():
                numbers.extend([ase.data.atomic_numbers[s]]*n)
            atom_numbers.extend(numbers*num)

        # unpack info
        system_type = params.get("type", "surface")
        assert system_type in self.supported_systems, f"{system_type} is not supported..."
        self.system_type = system_type

        cell = params.get("cell", []) # depands on system
        region = params.get("region", None) # 4x4 matrix, where place atoms
        splits = params.get("splits", None) # to repeat cell

        volume = params.get("volume", None)
        cell_bounds = None

        number_of_variable_cell_vectors =0
        if system_type == "bulk":
            slab = Atoms("", pbc=True)
            # - check number_of_variable_cell_vectors
            number_of_variable_cell_vectors = 3 - len(cell)
            box_to_place_in = None
            if number_of_variable_cell_vectors > 0:
                box_to_place_in = [[0.,0.,0.], np.zeros((3,3))]
                if len(cell) > 0:
                    box_to_place_in[1][number_of_variable_cell_vectors:] = cell
            # --- check volume
            if not volume:
                volume = 10.*len(atom_numbers) # AA^3
            # --- cell bounds
            cell_bounds = {}
            angles, lengths = ["phi", "chi", "psi"], ["a", "b", "c"]
            for k in angles:
                cell_bounds[k] = region.get(k, [15, 165])
            for k in lengths:
                cell_bounds[k] = region.get(k, [2, 60])
            cell_bounds = CellBounds(cell_bounds)
            # --- splits
            if splits:
                splits_ = {}
                for r, p in zip(splits["repeats"], splits["probs"]):
                    splits_[tuple(r)] = p
                splits = splits_

            # --- two parameters
            test_dist_to_slab = False
            test_too_far = True

        elif system_type == "cluster":
            if not cell:
                cell = np.ones(3)*20.
            else:
                cell = np.array(cell)
            slab = Atoms(cell = cell, pbc=True)
            
            # set box to explore
            # NOTE: shape (4,3), origin+3directions
            if region is None:
                region = np.zeros((4,3))
                region[1:,:] = 0.5*cell
            else:
                region = np.array(region)
            p0, v1, v2, v3 = region

            # parameters
            box_to_place_in = [p0, [v1, v2, v3]]
            test_dist_to_slab = False
            test_too_far = False

        elif system_type == "surface":
            # read substrate
            substrate_file = params["substrate"]
            self.params["substrate"] = str(Path(substrate_file).resolve())

            surfdis = params.get("surfdis", None)
            constraint = params.get("constraint", None)

            # create the surface
            slab = read(substrate_file) # NOTE: only one structure

            # define the volume in which the adsorbed cluster is optimized
            # the volume is defined by a corner position (p0)
            # and three spanning vectors (v1, v2, v3)
            pos = slab.get_positions()
            cell = slab.get_cell().complete()
            
            # create box for atoms to explore
            if region is None:
                assert surfdis is not None, "region and surfdis cant be undefined at the same time."
                p0 = np.array([0., 0., np.max(pos[:, 2]) + surfdis[0]]) # origin of the box
                v1, v2, v3 = cell.copy()
                v3[2] = surfdis[1]
                box_to_place_in = [p0, [v1, v2, v3]]
            else:
                region = np.array(region)
                if region.shape[0] == 3:
                    # auto add origin for [0, 0, 0]
                    p0 = [0., 0., 0.]
                    v1, v2, v3 = region
                elif region.shape[0] == 4:
                    p0, v1, v2, v3 = region
                box_to_place_in = [p0, [v1, v2, v3]]

            # two parameters
            test_dist_to_slab = True
            test_too_far = True

        # define the closest distance two atoms of a given species can be to each other
        unique_atom_types = get_all_atom_types(slab, atom_numbers)
        covalent_ratio = params.get("covalent_ratio", 0.8)
        blmin = closest_distances_generator(
            atom_numbers=unique_atom_types,
            ratio_of_covalent_radii = covalent_ratio # be careful with test too far
        )

        #print("colvent ratio is: ", covalent_ratio)
        #print("neighbour distance restriction")
        #self._print_blmin(blmin)

        # create the starting population
        #rng = np.random.default_rng(params.get("seed", 1112))
        np.random.seed(params.get("seed", 1112))
        rng = np.random # TODO: require rand function

        generator = StartGenerator(
            slab, 
            blocks, # blocks
            blmin,
            number_of_variable_cell_vectors=number_of_variable_cell_vectors,
            box_to_place_in=box_to_place_in,
            box_volume=volume,
            splits=splits,
            cellbounds=cell_bounds,
            test_dist_to_slab = test_dist_to_slab,
            test_too_far = test_too_far,
            rng = rng
        ) # structure generator

        # --- NOTE: we need some attributes to access
        self.slab = slab
        self.atom_numbers_to_optimise = atom_numbers

        self.use_tags = use_tags

        self.blmin = blmin
        self.cell_bounds = cell_bounds

        # - for output
        self.type = system_type
        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors

        self.box_to_place_in = box_to_place_in

        self.covalent_ratio = covalent_ratio
        self.test_dist_to_slab = test_dist_to_slab
        self.test_too_far = test_too_far

        return generator

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

        content =  "----- Bond Distance Minimum -----\n"
        content += "covalent ratio: {}\n".format(self.covalent_ratio)
        content += " "*4+("{:>6}  "*nelements).format(*symbols) + "\n"
        for i, s in enumerate(symbols):
            content += ("{:<4}"+"{:>8.4f}"*nelements+"\n").format(s, *list(distance_map[i]))
        content += "too_far: {}, dist_to_slab: {}\n".format(self.test_too_far, self.test_dist_to_slab)
        content += "note: default too far tolerance is 2 times\n"

        return content
    
    def run(self, ran_size) -> List[Atoms]:
        """"""
        nfailed = 0
        starting_population = []
        while len(starting_population) < ran_size:
            candidate = self.generator.get_new_candidate(maxiter=self.MAX_RANDOM_TRY)
            # TODO: add some geometric restriction here
            if candidate is None:
                # print(f"This creation failed after {maxiter} attempts...")
                nfailed += 1
            else:
                if self.system_type == "cluster":
                    region_centre = np.mean(self.generator.slab.get_cell().complete(), axis=0)
                    cop = np.mean(candidate.positions, axis=0)
                    candidate.positions += region_centre - cop
                starting_population.append(candidate)
            #print("now we have ", len(starting_population))
            if nfailed > int(np.ceil(ran_size*100)):
                warnings.warn(
                    f"Too many failed generations, {nfailed} nfailed, {len(starting_population)} ngenerated...", 
                    RuntimeWarning
                )
                break

        return starting_population
    
    def as_dict(self):
        """"""
        return copy.deepcopy(self.params)
    
    def __repr__(self):
        """"""
        content = ""
        content += "----- Generator Params -----\n"
        content += f"type :{self.type}\n"
        content += f"number_of_variable_cell_vectors: {self.number_of_variable_cell_vectors}\n"

        # output summary
        vec3_format = "{:>8.4f}  {:>8.4f}  {:>8.4f}\n"
        #content += "system cell\n"
        #content +=  "xxxxxx " + vec3_format.format(*list(cell[0]))
        #content += "xxxxxx " + vec3_format.format(*list(cell[1]))
        #content += "xxxxxx " + vec3_format.format(*list(cell[2]))
        box_to_place_in = self.box_to_place_in
        if not box_to_place_in:
            box_to_place_in = [[0.,0.,0.], np.zeros((3,3))]
        p0, [v1, v2, v3] = box_to_place_in
        content += "insertion region\n"
        content +=  "origin " + vec3_format.format(*list(p0))
        content += "xxxxxx " + vec3_format.format(*list(v1))
        content += "xxxxxx " + vec3_format.format(*list(v2))
        content += "xxxxxx " + vec3_format.format(*list(v3))

        content += self._print_blmin(self.blmin)

        return content


class RandomBuilder(StructureBuilder):

    #: Number of attempts to create a random candidate.
    MAX_ATTEMPTS_PER_CANDIDATE: int = 1000

    #: Number of attempts to create a number of candidates.
    #       if 10 structures are to create, run will try 5*10=50 times.
    MAX_TIMES_SIZE: int = 5

    use_tags = False
    composition_atom_numebrs: List[int] = None
    composition_blocks: Mapping[str,int] = None

    def __init__(
        self, composition: Mapping[str,int], region: dict={}, cell=[], covalent_ratio=[1.0, 2.0], 
        directory="./", random_seed=None, *args, **kwargs
    ):
        super().__init__(directory, random_seed, *args, **kwargs)

        # - create region
        region = copy.deepcopy(region)
        shape = region.pop("method", "auto")
        self.region = registers.create("region", shape, convert_name=True, **region)

        # - parse composition
        self.composition = composition
        self._parse_composition()

        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        # - check cell
        self.cell = cell
        if not self.cell: # None or []
            self.cell = np.array(cell).reshape(-1,3)

        # - other parameters
        self.blmin = None

        # - read from kwargs
        self.test_too_far = kwargs.get("test_too_far", True) # test_too_far

        self.test_dist_to_slab = kwargs.get("test_dist_to_slab", True) # test_dist_to_slab

        self.cell_volume = kwargs.get("cell_volume", None)
        self.cell_bounds = kwargs.get("cell_bounds", None)
        self.cell_splits = kwargs.get("cell_splits", None)
        self.number_of_variable_cell_vectors = 0 # number_of_variable_cell_vectors

        return

    def run(self, substrate: Atoms=None, size: int=1, *args, **kwargs) -> List[Atoms]:
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

        # --- NOTE: we need some attributes to access
        self.atom_numbers_to_optimise = self.composition_atom_numebrs

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
        self.composition_atom_numebrs = atom_numbers
        
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

        unique_atom_types = set(self.composition_atom_numebrs)
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
            radii = [covalent_radii[x]*self.covalent_max for x in self.composition_atom_numebrs]
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

        unique_atom_types = set(self.composition_atom_numebrs)
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
        self, region: dict, composition: Mapping[str,int], cell=None, covalent_ratio=[1.0, 2.0], 
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
        self.substrate = substarte

        # define the closest distance two atoms of a given species can be to each other
        unique_atom_types = get_all_atom_types(self.substrate, self.composition_atom_numebrs)
        self.blmin = self._build_tolerance(unique_atom_types)

        # - ignore lattice parameters
        self.cell_volume = None
        self.cell_splits = None
        self.cell_bounds = None

        self.box_to_place_in = [self.region._origin, self.region._cell]

        return


if __name__ == "__main__":
    """"""
    ...