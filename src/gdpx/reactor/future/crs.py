#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path

from ase import Atoms
from ase.io import read, write

import ase
from ase.ga.utilities import (closest_distances_generator, atoms_too_close,
                              atoms_too_close_two_sets)

from ase.constraints import FixAtoms

# molecular dynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin

from gdpx.computation.aseplumed import AsePlumed


class GridReactionSampling():

    """ steps
        1. slice surface grid
        2. random placement
        3. constrained MD (harmonic potential with large spring constant)
        4. free MD towards IS and FS
        5. collect data and train
    """

    grid_length = 5.0

    constrained_distance = 2.2

    # MD parameters
    temperature = 600 # Kelvin
    timestep = 2.0 # fs

    def __init__(self, reactant, surface):
        """"""
        self.reactant = reactant
        self.surface = surface

        # generate blmin
        # tag every molecules and state atomice types
        #unique_atom_types = []
        #for i, atoms in enumerate(reactants):
        #    atoms.set_tags(i)
        #    unique_atom_types.extend(atoms.get_atomic_numbers())
        #unique_atom_types = list(set(unique_atom_types))
        unique_atom_types = ["C", "O", "Cr", "Zn"]
        unique_atom_types = [6, 8, 24, 30]

        blmin = closest_distances_generator(
            atom_numbers=unique_atom_types,
            ratio_of_covalent_radii=0.8 # be careful with test too far
        )
        self.blmin = blmin

        return
    
    def __partition_surface_grid(self):
        # surface
        surface = self.surface
        positions = surface.positions

        # generate grid
        cell = self.surface.cell.complete()
        print("cell: ", cell)

        a, b, c, alpha, beta, gamma = self.surface.get_cell_lengths_and_angles()

        grid_length = self.grid_length
        na, nb = int(np.round(a/grid_length)), int(np.round(b/grid_length))
        la, lb = a/na, b/nb
        print("number of grids: ", na, nb)
        print("grid size", la, lb)

        # run over regions and place reactant
        candidates = []
        for i in range(na):
            for j in range(nb):
                print("region: ", i, j)
                x1, x2 = i*la, (i+1)*la
                y1, y2 = j*lb, (j+1)*lb
                # find zmax in the region
                indices = []
                for idx, pos in enumerate(positions):
                    if (x1 <= pos[0] < x2) and (y1 <= pos[1] < y2):
                        indices.append(idx)
                print("number in this region: ", len(indices))
                zmax = np.max(positions[indices][:, 2])
                origin = np.array([x1,y1,zmax])
                lengths = np.array([la, lb, 2.6]) # 2.6 is distance between reactant and surface
                print("origin: ", origin)

                # add reactant
                candidate = self.__place_random_configuration(self.constrained_distance, origin, lengths)
                candidates.append(candidate)
        write("cand.xyz", candidates)

        return candidates

    def __place_random_configuration(self, bond_distance, origin, lengths):
        """ randomly place reactants in a given region
        """
        # place positions
        atoms = self.reactant.copy()
        # adjust bond distance
        vec = atoms[0].position - atoms[1].position
        new_pos = atoms[0].position + vec / np.linalg.norm(vec) * bond_distance
        atoms[1].position = new_pos

        # Apply a random translation
        for i in range(100):
            candidate = self.surface.copy()
            ran_pos = np.random.random(3) 
            ran_pos[2] = 1.0
            pos = ran_pos * lengths + origin
            com = atoms.get_center_of_mass()
            atoms.translate(pos - com) # set to the origin
            # Apply a random rotation to multi-atom blocks
            phi, theta, psi = 360 * np.random.random(3)
            atoms.euler_rotate(
                phi=phi, theta=0.5 * theta, psi=psi,
                center=pos
            )
            # add reactant
            candidate.extend(atoms)
            # TODO: check anchor point (C is under O)
            # check distance
            if not atoms_too_close(candidate, self.blmin):
                # print(f"generate after {i+1} attempts...")
                break
        else:
            candidate = None
        
        return candidate
    
    def __molecular_dynamics(self, atoms):
        """ CMD
        """
        cons = FixAtoms(indices=[a.index for a in atoms if a.position[2] < 4.5])
        atoms.set_constraint(cons)

        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
        print("initial temperature: ", atoms.get_temperature())
        write("xxx.xyz", atoms)
        dyn = Langevin(
            atoms, 
            timestep = self.timestep*ase.units.fs, 
            temperature_K = self.temperature, 
            friction = 0.002, # TODO: what is the unit?
            fixcm = True
        )
        plumed_worker = AsePlumed(
            atoms, timestep=0,
            in_file = "plumed.dat",
            out_file = "plumed.out"
        )

        md_forces = atoms.get_forces()

        with open("md.xyz", "w") as fopen:
            fopen.write("")

        mdsteps = 500
        for i in range(mdsteps):
            dyn.step(md_forces)

            # bias forces
            energy = atoms.get_potential_energy()
            md_forces = atoms.get_forces()
            bias_forces = plumed_worker.external_forces(
                i, new_energy=energy, new_forces=md_forces
            )
            md_forces = bias_forces

            write("md.xyz", atoms, append=True)
        
        return
    
    def __call__(self, calc):

        # place MD
        if Path("./cand.xyz").exists():
            candidates = read("./cand.xyz", ":")
        else:
            candidates = self.__partition_surface_grid()
        print("number of candidates: ", candidates)

        # run constrained MD
        for cand in candidates[3:]:
            calc.reset()
            cand.calc = calc
            self.__molecular_dynamics(cand)

            # exit()

        return


if __name__ == "__main__":
    reactant = read("/mnt/scratch2/users/40247882/catsign/lasp-main/reactions/CO/allstr.arc", "-1", format="dmol-arc")
    surface = read("/mnt/scratch2/users/40247882/catsign/lasp-main/sample/ZnO-2Ov4Cr/IS/allstr.arc", "-1", format="dmol-arc")

    grs = GridReactionSampling(reactant, surface)

    from gdpx.computation.lasp import LaspNN
    pot_path = "/mnt/scratch2/users/40247882/catsign/lasp-main/ZnCrOCH.pot"
    pot = dict(
        H  = pot_path,
        C  = pot_path,
        O  = pot_path,
        Cr = pot_path,
        Zn = pot_path
    )
    calc = LaspNN(
        directory = "./LaspNN-Worker",
        command = "mpirun -n 4 lasp",
        pot=pot
    )

    grs(calc)