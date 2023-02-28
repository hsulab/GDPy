#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List

import numpy as np

from ase import Atoms
from ase import data, units
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.ga.utilities import closest_distances_generator

from GDPy.builder.species import build_species
from GDPy.builder.group import create_a_group
from GDPy.mc.operators.move import MoveOperator


class SwapOperator(MoveOperator):

    MAX_ATTEMPTS = 1000

    def __init__(self, group: List[str], temperature: float = 300, pressure: float = 1, covalent_ratio=[0.8,2.0], max_disp: float = 2, use_rotation: bool = True):
        super().__init__(group, temperature, pressure, covalent_ratio, max_disp, use_rotation)

        assert len(self.group) == 2, "SwapOperator can only accept two groups."

        return
    
    def run(self, atoms: Atoms, rng=np.random) -> Atoms:
        """"""
        # - basic
        cur_atoms = atoms
        chemical_symbols = cur_atoms.get_chemical_symbols()
        cell = cur_atoms.get_cell(complete=True)

        # -- neighbour list
        nl = NeighborList(
            self.covalent_max*np.array(natural_cutoffs(cur_atoms)), 
            skin=0.0, self_interaction=False, bothways=True
        )

        # -- TODO: init blmin?
        type_list = list(set(chemical_symbols))
        unique_atomic_numbers = [data.atomic_numbers[a] for a in type_list]
        self.blmin = closest_distances_generator(
            atom_numbers=unique_atomic_numbers,
            ratio_of_covalent_radii = self.covalent_min # be careful with test too far
        )

        # - swap the species
        for i in range(self.MAX_ATTEMPTS):
            # -- swap
            cur_atoms = atoms.copy()

            # -- pick an atom
            #   either index of an atom or tag of an moiety
            group_indices = create_a_group(atoms, self.group[0])
            if len(group_indices) == 0:
                first_pick = None
            else:
                first_pick = rng.choice(group_indices, 1)
            # TODO: if use tag, return all indices in the tag
            group_indices = create_a_group(atoms, self.group[1])
            if len(group_indices) == 0:
                second_pick = None
            else:
                second_pick = rng.choice(group_indices, 1)
            self.pfunc(f"first: {first_pick} second: {second_pick}")
            # TODO: check ... this happens when same species are swapped.
            assert first_pick != second_pick, "Two moieties should be different."

            # -- find tag atoms
            first_species = cur_atoms[first_pick] # default copy
            second_species = cur_atoms[second_pick]
            self.pfunc(f"origin: {first_species.symbols} {first_species.positions}")
            self.pfunc(f"origin: {second_species.symbols} {second_species.positions}")
            first_species = self.rotate_species(first_species)
            second_species = self.rotate_species(second_species)

            # TODO: deal with pbc, especially for move step
            first_positions = copy.deepcopy(first_species.get_positions())
            second_positions = copy.deepcopy(second_species.get_positions())

            # -- swap
            cur_atoms.positions[first_pick] = second_positions
            cur_atoms.positions[second_pick] = first_positions

            first_species = cur_atoms[first_pick]
            second_species = cur_atoms[second_pick]
            self.pfunc(f"swapped: {first_species.symbols} {first_species.positions}")
            self.pfunc(f"swapped: {second_species.symbols} {second_species.positions}")

            # -- use neighbour list
            idx_pick = []
            idx_pick.extend(first_pick)
            idx_pick.extend(second_pick)
            if not self.check_overlap_neighbour(nl, cur_atoms, cell, idx_pick):
                self.pfunc(f"succeed to random after {i+1} attempts...")
                break
        else:
            cur_atoms = None

        return cur_atoms
    
    def rotate_species(self, species: Atoms):
        """"""
        species_ = species.copy() # TODO: make clean atoms?
        org_com = np.mean(species_.positions, axis=0)
        if self.use_rotation and len(species_) > 1:
            phi, theta, psi = 360 * self.rng.uniform(0,1,3)
            species_.euler_rotate(
                phi=phi, theta=0.5 * theta, psi=psi,
                center=org_com
            )

        return species_


if __name__ == "__main__":
    ...