#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from ase import Atoms
from ase import data, units
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.ga.utilities import closest_distances_generator

from GDPy.builder.group import create_a_group
from GDPy.builder.species import build_species


class MoveOperator():

    MAX_RANDOM_ATTEMPTS = 1000

    pfunc = print

    #: Name of current species to operate.
    _curr_species: str = None

    def __init__(
        self, group: str, temperature: float=300., pressure: float=1.,
        covalent_ratio=[0.8,2.0], max_disp: float=2.0, use_rotation: bool=True
    ):
        """"""
        self.group = group # atom index or tag

        # - thermostat
        self.temperature = temperature
        self.pressure = pressure

        # - close check
        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        self.max_disp = max_disp
        self.use_rotation = use_rotation

        return
    
    def run(self, atoms: Atoms, rng=np.random) -> Atoms:
        """"""
        # - pick an atom
        #   either index of an atom or tag of an moiety
        group_indices = create_a_group(atoms, self.group)
        if len(group_indices) == 0:
            idx_pick = None
        else:
            idx_pick = rng.choice(group_indices, 1)
        self.pfunc(f"move idx: {idx_pick}")
        # TODO: if use tag, return all indices in the tag

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
        # record original position of idx_pick
        species = cur_atoms[idx_pick]
        self._curr_species = species.get_chemical_formula()

        #org_pos = new_atoms[idx_pick].position.copy() # original position
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
            # -- make a copy
            species_ = species.copy()
            # -- Apply a random rotation to multi-atom blocks
            if self.use_rotation and len(idx_pick) > 1:
                phi, theta, psi = 360 * self.rng.uniform(0,1,3)
                species_.euler_rotate(
                    phi=phi, theta=0.5 * theta, psi=psi,
                    center=org_com
                )
            # -- Apply translation
            new_vec = ran_pos - org_com
            species_.translate(new_vec)
            cur_atoms.positions[idx_pick] = species_.positions.copy()
            # use neighbour list
            if not self.check_overlap_neighbour(nl, cur_atoms, cell, idx_pick):
                self.pfunc(f"succeed to random after {i+1} attempts...")
                self.pfunc(f"original position: {org_com}")
                self.pfunc(f"random position: {ran_pos}")
                break
            cur_atoms.positions[idx_pick] = org_positions
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
            self.pfunc(f"- check index {idx_pick}")
            indices, offsets = nl.get_neighbors(idx_pick)
            if len(indices) > 0:
                self.pfunc(f"nneighs: {len(indices)}")
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
                self.pfunc("no neighbours, being isolated...")
                overlapped = True
                # TODO: try rotate?
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
            0., beta, coef
        )
        content += "Energy Difference %.4f [eV]\n" %ene_diff
        content += "Accept Ratio %.4f\n" %acc_ratio
        self.pfunc(content)

        rn_move = rng.uniform()
        self.pfunc(f"{self.__class__.__name__} Probability %.4f" %rn_move)

        return rn_move < acc_ratio
    
    def as_dict(self):
        """"""

        return

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"max disp: {self.max_disp}\n"
        content += f"group: \n"
        content += f"  {self.group}\n"

        return content


if __name__ == "__main__":
    ...