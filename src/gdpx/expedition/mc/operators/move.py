#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from ase import Atoms
from ase import data, units
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.ga.utilities import closest_distances_generator

from gdpx.builder.group import create_a_group
from gdpx.builder.species import build_species

from .operator import AbstractOperator


class MoveOperator(AbstractOperator):

    name: str = "move"

    def __init__(
        self, particles: List[str]=None, region: dict={}, temperature: float=300., pressure: float=1.,
        covalent_ratio=[0.8,2.0], max_disp: float=2.0, use_rotation: bool=True,
        *args, **kwargs
    ):
        """"""
        super().__init__(
            region=region, temperature=temperature, pressure=pressure, 
            covalent_ratio=covalent_ratio, use_rotation=use_rotation,
            *args, **kwargs
        )

        self.particles = particles

        self.max_disp = max_disp

        return
    
    def run(self, atoms: Atoms, rng=np.random) -> Atoms:
        """"""
        super().run(atoms)
        self._extra_info = "-"

        # BUG: If there is no species in the system...
        species_indices = self._select_species(atoms, self.particles, rng=rng)

        # - basic
        cur_atoms = atoms.copy() # TODO: use clean atoms?
        chemical_symbols = cur_atoms.get_chemical_symbols()
        cell = cur_atoms.get_cell(complete=True)

        # -- TODO: init blmin?
        type_list = list(set(chemical_symbols))
        unique_atomic_numbers = [data.atomic_numbers[a] for a in type_list]
        self.blmin = closest_distances_generator(
            atom_numbers=unique_atomic_numbers,
            ratio_of_covalent_radii = self.covalent_min # be careful with test too far
        )

        # - neighbour list
        nl = NeighborList(
            self.covalent_max*np.array(natural_cutoffs(cur_atoms)), 
            skin=0.0, self_interaction=False, bothways=True
        )

        # - find tag atoms
        # record original position of species_indices
        species = cur_atoms[species_indices]
        self._extra_info = f"{species.get_chemical_formula()}_{species_indices}"

        #org_pos = new_atoms[species_indices].position.copy() # original position
        # TODO: deal with pbc, especially for move step
        org_com = np.mean(species.positions, axis=0)
        org_positions = species.positions.copy()

        # - move the atom
        for i in range(self.MAX_RANDOM_ATTEMPTS):
            rsq = 1.1
            while (rsq > 1.0):
                rvec = 2*rng.uniform(size=3) - 1.0
                rsq = np.linalg.norm(rvec)
            ran_pos = org_com + rvec*self.max_disp
            # -- make a copy and rotate
            species_ = self._rotate_species(species, rng=rng)
            curr_cop = np.average(species_.positions, axis=0)
            # -- translate
            new_vec = ran_pos - curr_cop
            species_.translate(new_vec)
            cur_atoms.positions[species_indices] = species_.positions.copy()
            # use neighbour list
            if not self.check_overlap_neighbour(nl, cur_atoms, cell, species_indices):
                self._print(f"succeed to random after {i+1} attempts...")
                self._print(f"original position: {org_com}")
                self._print(f"random position: {ran_pos}")
                self._print(f"actual position: {np.average(cur_atoms.positions[species_indices], axis=0)}")
                break
            cur_atoms.positions[species_indices] = org_positions
        else:
            cur_atoms = None

        return cur_atoms

    def check_overlap_neighbour(
        self, nl, new_atoms, cell, species_indices: List[int]
    ):
        """ use neighbour list to check newly added atom is neither too close or too
            far from other atoms
        """
        # - get symbols here since some operators may change the symbol
        chemical_symbols = new_atoms.get_chemical_symbols()

        overlapped = False
        nl.update(new_atoms)
        for idx_pick in species_indices:
            #self._print(f"- check index {idx_pick}")
            indices, offsets = nl.get_neighbors(idx_pick)
            if len(indices) > 0:
                #self._print(f"nneighs: {len(indices)}")
                # should close to other atoms
                for ni, offset in zip(indices, offsets):
                    dis = np.linalg.norm(new_atoms.positions[idx_pick] - (new_atoms.positions[ni] + np.dot(offset, cell)))
                    pairs = [chemical_symbols[ni], chemical_symbols[idx_pick]]
                    pairs = tuple([data.atomic_numbers[p] for p in pairs])
                    #print("distance: ", ni, dis, self.blmin[pairs])
                    if dis < self.blmin[pairs]:
                        overlapped = True
                        break
            else:
                # TODO: is no neighbours valid?
                self._print("no neighbours, being isolated...")
                overlapped = True
                break

        return overlapped
    
    def metropolis(self, prev_ene: float, curr_ene: float, rng=np.random) -> bool:
        """"""
        # - acceptance ratio
        kBT_eV = units.kB * self.temperature
        beta = 1./kBT_eV # 1/(kb*T), eV

        coef = 1.0
        ene_diff = curr_ene - prev_ene
        acc_ratio = np.min([1.0, coef * np.exp(-beta*(ene_diff))])

        #content = "\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n" %(
        #    self.acc_volume, len(self.tag_list[expart]), cubic_wavelength, coef
        #)
        content = "\nVolume %.4f Beta %.4f Coefficient %.4f\n" %(
            self.region.get_volume(), beta, coef
        )
        content += "Energy Difference %.4f [eV]\n" %ene_diff
        content += "Accept Ratio %.4f\n" %acc_ratio
        for x in content.split("\n"):
            self._print(x)

        rn_move = rng.uniform()
        self._print(f"{self.__class__.__name__} Probability %.4f" %rn_move)

        return rn_move < acc_ratio
    
    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["particles"] = self.particles
        params["max_disp"] = self.max_disp

        return params

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"max disp: {self.max_disp}\n"
        content += f"particles: \n"
        content += f"  {self.particles}\n"

        return content


if __name__ == "__main__":
    ...