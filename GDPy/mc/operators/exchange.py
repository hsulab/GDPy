#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
from typing import NoReturn, List

import numpy as np

from ase import Atoms
from ase import data, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.ga.utilities import closest_distances_generator

from GDPy.builder.group import create_a_group
from GDPy.builder.species import build_species
from GDPy.builder.region import CubicRegion


def get_tags_per_species(atoms: Atoms):
    """Get tags per species.

    Example:

        .. code-block:: python

            >>> atoms = Atoms("PtPtPtCOCO")
            >>> tags = [0, 0, 0, 1, 1, 2, 2]
            >>> atoms.set_tags(tags)
            >>> get_tags_per_species(atoms)
            >>> {'Pt3': [0], 'CO': [1, 2]}

    """

    tags = atoms.get_tags() # default is all zero

    tags_dict = {} # species -> tag list
    for key, group in itertools.groupby(enumerate(tags), key=lambda x:x[1]):
        cur_indices = [x[0] for x in group]
        #print(key, " :", cur_indices)
        cur_atoms = atoms[cur_indices]
        formula = cur_atoms.get_chemical_formula()
        #print(formula)
        if formula in tags_dict:
            tags_dict[formula].append(key)
        else:
            tags_dict[formula] = [key]

    return tags_dict


class ExchangeOperator():

    MAX_RANDOM_ATTEMPTS = 1000

    MIN_RANDOM_TAG = 10000
    MAX_RANDOM_TAG = 100000

    pfunc = print

    #: 
    _curr_operation = None # insert or remove

    _curr_tags_dict: dict = None

    _curr_volume: float = None

    def __init__(
        self, region: str, reservoir: dict, temperature: float=300., pressure: float=1.,
        covalent_ratio=[0.8,2.0], max_disp: float=2.0, use_rotation: bool=True,
        use_bias: bool=True
    ):
        """"""
        self.region = CubicRegion(region)

        self.species = reservoir["species"]
        self.mu = reservoir["mu"]

        # - thermostat
        self.temperature = temperature
        self.pressure = pressure

        # - close check
        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        self.max_disp = max_disp
        self.use_rotation = use_rotation
        self.use_bias = use_bias

        return
    
    def _insert(self, atoms_, tags_dict, species: str, rng):
        """"""
        atoms = atoms_.copy()

        # - prepare
        adpart = build_species(species) # particle to add
        
        # - add velocity in case the mixed MC/MD is performed
        MaxwellBoltzmannDistribution(adpart, temperature_K=self.temperature, rng=rng)

        # - add tag
        used_tags = set(atoms.get_tags().tolist())
        adpart_tag = 0
        while adpart_tag in used_tags:
            # NOTE: np.random only has randint
            adpart_tag = rng.integers(self.MIN_RANDOM_TAG, self.MAX_RANDOM_TAG)
        adpart_tag = int(adpart_tag)
        print("adpart tag: ", adpart_tag, type(adpart_tag))
        # NOTE: ase accepts int or list as tags
        adpart.set_tags(adpart_tag)

        # - add particle
        atoms.extend(adpart)

        # - find species indices
        #   NOTE: adpart is always added to the end
        species_indices = [i for i, t in enumerate(atoms.get_tags()) if t == adpart_tag]

        # - init blmin
        cell = atoms.get_cell(complete=True)
        chemical_symbols = atoms.get_chemical_symbols()

        type_list = list(set(chemical_symbols))
        unique_atomic_numbers = [data.atomic_numbers[a] for a in type_list]
        self.blmin = closest_distances_generator(
            atom_numbers=unique_atomic_numbers,
            ratio_of_covalent_radii = self.covalent_min # be careful with test too far
        )

        # - neighbour list
        nl = NeighborList(
            self.covalent_max*np.array(natural_cutoffs(atoms)), 
            skin=0.0, self_interaction=False, bothways=True
        )

        # - get a random position
        species = atoms[species_indices]
        org_com = np.mean(species.positions, axis=0)
        org_positions = species.positions.copy()

        for i in range(self.MAX_RANDOM_ATTEMPTS):
            ran_frac_pos = rng.uniform(0,1,3)
            # - make a copy
            species_ = species.copy()
            # -- TODO: check if it is in the region
            ran_pos = np.dot(ran_frac_pos, atoms.get_cell())
            if self.use_rotation and len(adpart) > 1:
                phi, theta, psi = 360 * rng.uniform(0,1,3)
                species_.euler_rotate(
                    phi=phi, theta=0.5 * theta, psi=psi,
                    center=org_com
                )
            new_vec = ran_pos - org_com
            species_.translate(new_vec)
            atoms.positions[species_indices] = copy.deepcopy(species_.positions)
            if not self.check_overlap_neighbour(nl, atoms, cell, species_indices):
                self.pfunc(f"succeed to insert after {i+1} attempts...")
                self.pfunc(f"original position: {org_com}")
                self.pfunc(f"random position: {ran_pos}")
                break
            atoms.positions[species_indices] = org_positions
        else:
            # -- remove adpart since the insertion fails
            del atoms[species_indices]
            atoms = None

        return atoms
    
    def _remove(self, atoms_, tags_dict: dict, species: str, rng):
        """"""
        atoms = atoms_.copy()

        # - pick a random atom/molecule
        tag_idx_pick = rng.choice(tags_dict[species])
        
        # - remove then
        tags = atoms.get_tags()
        species_indices = [i for i, t in enumerate(tags) if t==tag_idx_pick]
        del atoms[species_indices]

        self.pfunc(f"succeed to remove tag {tag_idx_pick}")

        return atoms
    
    def run(self, atoms: Atoms, rng=np.random) -> Atoms:
        """"""
        # - make copy
        tags_dict = get_tags_per_species(atoms)
        self._curr_tags_dict = tags_dict
        #print(tags_dict)
        # TODO: only consider species within the region

        # -- compute acceptable volume
        if self.use_bias:
            # --- determine on-the-fly
            acc_volume = self.region.get_empty_volume(atoms)
        else:
            # --- get normal region
            acc_volume = self.region.get_volume()
        self._curr_volume = acc_volume

        # - choose a species to exchange
        #valid_species = [k for k, v in tag_dict.items() if len(v) > 0]
        nparticles = len(tags_dict.get(self.species, []))

        # - choose insert or remove
        if nparticles > 0:
            rn_ex = rng.uniform()
            if rn_ex < 0.5:
                self.pfunc("...insert...")
                self._curr_operation = "insert"
                cur_atoms = self._insert(atoms, tags_dict, self.species, rng)
            else:
                self.pfunc("...remove...")
                self._curr_operation = "remove"
                cur_atoms = self._remove(atoms, tags_dict, self.species, rng)
        else:
            self.pfunc("...insert...")
            self._curr_operation = "insert"
            cur_atoms = self._insert(atoms, tags_dict, self.species, rng)

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

        # - cubic thermo de broglie 
        hplanck = units._hplanck # J/Hz = kg*m2*s-1
        #_mass = np.sum([data.atomic_masses[data.atomic_numbers[e]] for e in expart]) # g/mol
        _species = build_species(self.species)
        _species_mass = np.sum(_species.get_masses())
        #print("species mass: ", _mass)
        _mass = _species_mass * units._amu
        kbT_J = kBT_eV * units._e # J = kg*m2*s-2
        cubic_wavelength = (hplanck/np.sqrt(2*np.pi*_mass*kbT_J)*1e10)**3 # thermal de broglie wavelength

        # - compute coefficient
        # -- determine number of exchangeable particles
        #print("miaow: ", self._curr_tags_dict)
        nexatoms = len(self._curr_tags_dict[self.species])

        # -- compute composed coefficient
        if self._curr_operation == "insert":
            coef = self._curr_volume/(nexatoms+1)/cubic_wavelength
            ene_diff = curr_ene - self.mu - prev_ene
        elif self._curr_operation == "remove":
            coef = nexatoms*cubic_wavelength/self._curr_volume
            ene_diff = curr_ene + self.mu - prev_ene
        else:
            raise RuntimeError(f"Unknown exchange operation {self._curr_operation}.")
        print("coef: ", coef)

        acc_ratio = np.min([1.0, coef * np.exp(-beta*(ene_diff))])

        content = "\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n" %(
            self._curr_volume, nexatoms, cubic_wavelength, coef
        )
        #content = "\nVolume %.4f Beta %.4f Coefficient %.4f\n" %(
        #    0., beta, coef
        #)
        content += "Energy Difference %.4f [eV]\n" %ene_diff
        content += "Accept Ratio %.4f\n" %acc_ratio
        self.pfunc(content)

        rn_move = rng.uniform()
        self.pfunc(f"{self.__class__.__name__} Probability %.4f" %rn_move)

        # - reset stored temp data
        self._curr_operation = None
        self._curr_tags_dict = None
        self._curr_volume = None

        return rn_move < acc_ratio
    
    def as_dict(self):
        """"""

        return
    
    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        content += f"reservoir: "
        content += f"  species {self.species} with chemical potential {self.mu} [eV]\n"
        content += f"  within the region {self.region}\n"

        return content


if __name__ == "__main__":
    get_tags_per_species(None)
    ...