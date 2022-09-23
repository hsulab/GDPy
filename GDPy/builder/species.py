#!/usr/bin/env python3
# -*- coding: utf-8 -*

import time
from typing import List

import numpy as np

import ase
from ase import Atoms
from ase.collections import g2
from ase.build import molecule

from ase.neighborlist import natural_cutoffs, NeighborList

from GDPy.builder.builder import StructureGenerator

class FormulaBasedGenerator(StructureGenerator):

    def __init__(self, chemical_formula, directory="./", *args, **kwargs):
        """"""
        super().__init__(directory)

        self.atoms = self._parse_formula(chemical_formula)

        return
    
    def _parse_formula(self, formula):
        """"""
        # - build adsorbate
        atoms = None
        if formula in ase.data.chemical_symbols:
            atoms = Atoms(formula, positions=[[0.,0.,0.]])
        elif formula in g2.names:
            atoms = molecule(formula)
        else:
            atoms = None

        return atoms
    
    def run(self, *args, **kargs) -> List[Atoms]:
        """"""
        frames = None
        if self.atoms:
            frames = [self.atoms]

        return frames


def build_species(species):
    # - build adsorbate
    atoms = None
    if species in ase.data.chemical_symbols:
        atoms = Atoms(species, positions=[[0.,0.,0.]])
    elif species in g2.names:
        atoms = molecule(species)
        #print("molecule: ", atoms.positions)
        #exit()
    else:
        raise ValueError(f"Cant create species {species}")

    return atoms

# ===== ====== ===== ===== ====== ===== ===== ====== ===== =====
# Water Molecules Number Calculation @ 298.15K
# ===== ====== ===== ===== ====== ===== ===== ====== ===== =====
def calc_water_num(volume):
    """ Calculate number of water molecules at 298.15K with given volume
        (AA^3)
    """
    # water density
    water_molecule_weight = 18.0152 # g/mol
    water_density = 0.997074 # g/cm^3 298.15K
    n_avogadro = 6.02e23 # mol
    # volume = 16*13*10 # Å^3

    n_water_molecule_per_A = (water_density / water_molecule_weight) * n_avogadro * 1e-24

    return np.floor(n_water_molecule_per_A * volume)

# random place species into given region
def ideal_gas_insertin(pressure, volume, temperature):
    """ pV = nRT
    """
    #R = 8.314
    #nspecies = (pressure*volume) / (R*temperature) * 6.0221367e23
    kB = 1.38 * 1e-23 # J/K
    nmolecules = (pressure*volume) / (kB*temperature)

    return nmolecules

MAX_RANDOM_ATTEMPS = 100
def check_overlap_neighbour(
    nl, new_atoms, chemical_symbols, 
    species_indices
):
    """ use neighbour list to check newly added atom is neither too close or too
        far from other atoms
    """
    overlapped = False
    nl.update(new_atoms)
    for idx_pick in species_indices:
        print("- check index ", idx_pick)
        indices, offsets = nl.get_neighbors(idx_pick)
        if len(indices) > 0:
            print("nneighs: ", len(indices))
            # should close to other atoms
            for ni, offset in zip(indices, offsets):
                dis = np.linalg.norm(new_atoms.positions[idx_pick] - (new_atoms.positions[ni] + np.dot(offset, self.cell)))
                pairs = [chemical_symbols[ni], chemical_symbols[idx_pick]]
                pairs = tuple([data.atomic_numbers[p] for p in pairs])
                print("distance: ", ni, dis, self.blmin[pairs])
                if dis < self.blmin[pairs]:
                    overlapped = True
                    break
        else:
            # TODO: is no neighbours valid?
            print("no neighbours, being isolated...")
            overlapped = True
            # TODO: try rotate?
            break

    return overlapped

def random_position_neighbour(
    self, 
    new_atoms: Atoms, 
    species_indices,
    operation # move or insert
):
    """"""
    print("\n---- Generate Random Position -----\n")
    st = time.time()

    # - find tag atoms
    # record original position of idx_pick
    species = new_atoms[species_indices]

    #org_pos = new_atoms[idx_pick].position.copy() # original position
    # TODO: deal with pbc, especially for move step
    org_com = np.mean(species.positions.copy(), axis=0)
    org_positions = species.positions.copy()

    chemical_symbols = new_atoms.get_chemical_symbols()
    nl = NeighborList(
        self.covalent_max*np.array(natural_cutoffs(new_atoms)), 
        skin=0.0, self_interaction=False, bothways=True
    )
    for i in range(MAX_RANDOM_ATTEMPS): # maximum number of attempts
        print(f"random attempt {i}")
        if operation == "insert":
            # NOTE: z-axis should be perpendicular
            ran_frac_pos = self.rng.uniform(0,1,3)
            ran_pos = np.dot(ran_frac_pos, self.cell) 
            ran_pos[2] = self.cmin + ran_frac_pos[2] * (self.cmax-self.cmin)
        elif operation == "move":
            # get random motion vector
            rsq = 1.1
            while (rsq > 1.0):
                rvec = 2*self.rng.uniform(size=3) - 1.0
                rsq = np.linalg.norm(rvec)
            ran_pos = org_com + rvec*self.max_movedisp
        # update position of idx_pick
        #new_atoms.positions[species_indices] += new_vec
        species_ = species.copy()
        # --- Apply a random rotation to multi-atom blocks
        if self.use_rotation and len(species_indices) > 1:
            phi, theta, psi = 360 * self.rng.uniform(0,1,3)
            species_.euler_rotate(
                phi=phi, theta=0.5 * theta, psi=psi,
                center=org_com
            )
        # -- Apply translation
        new_vec = ran_pos - org_com
        species_.translate(new_vec)
        new_atoms.positions[species_indices] = species_.positions.copy()
        # use neighbour list
        if not self.check_overlap_neighbour(nl, new_atoms, chemical_symbols, species_indices):
            print(f"succeed to random after {i+1} attempts...")
            print("original position: ", org_com)
            print("random position: ", ran_pos)
            break
        new_atoms.positions[species_indices] = org_positions
    else:
        new_atoms = None

    et = time.time()
    print("used time: ", et - st)

    return new_atoms


def test_nvt():
    print(8.314*273.15/101325)
    temperature = 500 # K
    pressure = 1.0*101325 # atm
    #volume = 50*50*40 * 1e-30 # m^3
    volume = 100*100*50 * 1e-30 # m^3

    #temperature = 273.15 # K
    #pressure = 101325 # atm
    #volume = 0.0224
    print(ideal_gas_insertin(pressure, volume, temperature))

    return

if __name__ == "__main__":
    nwater = calc_water_num(5.6102*4.85857572*(18.871-8.0))
    print(nwater)
    exit()
    from ase.io import read, write
    atoms = read("/mnt/scratch2/users/40247882/pbe-oxides/eann-main/m09/gcmd/Pt111s3200/PtCOx-MD.xyz")

    cmin, cmax = 20, 30

    rng = np.random.default_rng(1112)
    
    for i in range(10):
        species = build_species("CO")
        ran_frac_pos = rng.uniform(0,1,3)
        ran_pos = np.dot(ran_frac_pos, atoms.cell) 
        ran_pos[2] = cmin + ran_frac_pos[2] * (cmax-cmin)

        print("pos: ", ran_pos)
        species.positions += ran_pos
        atoms.extend(species)

    for i in range(5):
        species = build_species("O2")
        ran_frac_pos = rng.uniform(0,1,3)
        ran_pos = np.dot(ran_frac_pos, atoms.cell) 
        ran_pos[2] = cmin + ran_frac_pos[2] * (cmax-cmin)

        print("pos: ", ran_pos)
        species.positions += ran_pos
        atoms.extend(species)
    
    write("/mnt/scratch2/users/40247882/pbe-oxides/eann-main/m09/gcmd/Pt111s3200/PtCOx-MD-new.xyz", atoms)

    pass
