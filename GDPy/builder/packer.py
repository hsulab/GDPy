#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
from typing import List

import numpy as np

import ase
from ase import Atoms
from ase.io import read, write
from ase.ga.utilities import closest_distances_generator, atoms_too_close

from GDPy.core.register import registers
from GDPy.builder.builder import StructureBuilder


"""Pack given molecules into given box.

This is a modifer as it requires substrates when initialising.

"""


@registers.builder.register
class PackerBuilder(StructureBuilder):

    #: Number of attempts to create a random candidate.
    MAX_ATTEMPTS_PER_CANDIDATE: int = 1000

    def __init__(
        self, box=np.eye(3)*20., covalent_ratio=[1.0, 2.0], 
        intermoleculer_distance=[4.0, 6.0],
        directory="./", random_seed=1112, 
        *args, **kwargs
    ):
        """"""
        super().__init__(directory, random_seed, *args, **kwargs)

        # TODO: replace this with Region
        self.box = box

        # distance restraints
        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        self.intermol_min = intermoleculer_distance[0]
        self.intermol_max = intermoleculer_distance[1]

        return
    
    def run(self, substrates: List[Atoms], numbers: List[int]=None, size: int=1, *args, **kwargs) -> List[Atoms]:
        """Modify input structures.

        Args:
            substrates: Building blocks.
            numbers: Number of each block.
        
        Returns:
            A list of structures.

        """
        # TODO: replace this with molecule builders
        self.subsrates = [copy.deepcopy(a) for a in substrates]

        n_substrates = len(self.subsrates)
        if numbers is not None:
            numbers_ = numbers
        else:
            numbers_ = [1]*n_substrates
        n_numbers = len(numbers_)
        assert n_numbers == n_substrates, "Incosistent number of substrates and their numbers."

        # TODO: check and overwrite tags
        for i, atoms in enumerate(self.subsrates):
            atoms.set_tags(i)
        
        # create neighbour list
        atomic_numbers = []
        for atoms in self.subsrates:
            atomic_numbers.extend(atoms.get_atomic_numbers())
        unique_atomic_numbers = list(set(atomic_numbers))
        blmin = closest_distances_generator(
            atom_numbers=unique_atomic_numbers,
            ratio_of_covalent_radii = self.covalent_min # be careful with test too far
        )
        print(blmin)

        # - run over...
        frames = []
        nframes = 0
        while nframes < size:
            atoms = self._irun(blmin)
            if atoms is not None:
                frames.append(atoms)
                nframes += 1
                print("nframes: ", nframes)

        return frames
    
    def _irun(self, blmin: dict, *args, **kwargs) -> Atoms:
        """"""
        box = self.box
        molecules = [copy.deepcopy(a) for a in self.subsrates]
        n_molecules = len(molecules)

        packed_structure = None
        for i in range(self.MAX_ATTEMPTS_PER_CANDIDATE):
            # -- get new molecule configurations
            curr_cops = np.dot(self.rng.random((n_molecules,3)), box)
            # -- check intermolecular distances
            pairs = itertools.combinations(curr_cops, 2)
            pair_distances = np.array(
                [*itertools.starmap(lambda p1, p2: np.linalg.norm(p1-p2), pairs)]
            )
            if np.any(pair_distances > self.intermol_max) or np.any(pair_distances < self.intermol_min):
                continue
            print(pair_distances)
            # -- translate molecules
            for cop, atoms in zip(curr_cops, molecules):
                self._translate(atoms, cop)
            # -- rotate molecules
            for j in range(self.MAX_ATTEMPTS_PER_CANDIDATE):
                packed_structure = Atoms()
                for atoms in molecules:
                    self._rotate(atoms)
                    packed_structure += atoms
                packed_structure.set_cell(box, scale_atoms=False, apply_constraint=True)
                # -- check atomic distances
                if not atoms_too_close(packed_structure, blmin, use_tags=True):
                    break
            else:
                packed_structure = None
            if packed_structure is not None:
                break
        else:
            #raise RuntimeError("Failed to pack molecules. Try increase max_attempts.")
            ...
        
        # - centre the atoms
        if packed_structure is not None:
            self._translate(packed_structure, np.sum(box/2., axis=0))

        return packed_structure
    
    def _translate(self, atoms: Atoms, dest) -> Atoms:
        """"""
        cop = np.average(atoms.get_positions(), axis=0)
        atoms.translate(dest - cop)

        return atoms
    
    def _rotate(self, atoms: Atoms) -> Atoms:
        """"""
        cop = np.average(atoms.get_positions(), axis=0)
        phi, theta, psi = 360 * self.rng.uniform(0,1,3)
        atoms.euler_rotate(
            phi=phi, theta=0.5 * theta, psi=psi,
            center=cop
        )

        return atoms


if __name__ == "__main__":
    atoms1 = read("/mnt/scratch2/users/40247882/porous/dptrain/r1/_explore/rxn/MeSCN.xyz")
    atoms2 = read("/mnt/scratch2/users/40247882/porous/dptrain/r1/_explore/rxn/MI.xyz")

    modifier = PackerBuilder()
    frames = modifier.run([atoms1, atoms2], size=10)
    print(frames)
    write("./sadas.xyz", frames)
    ...