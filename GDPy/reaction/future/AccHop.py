#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse

import numpy as np

import matplotlib as mpl
mpl.use('Agg') #silent mode
from matplotlib import pyplot as plt
plt.style.use("presentation")

# ASE
import ase.units
from ase import Atoms
from ase.io import read, write
from ase.units import kB
from ase.ga.utilities import (closest_distances_generator, atoms_too_close,
                              atoms_too_close_two_sets)

# calculator
from ase.calculators.gaussian import Gaussian, GaussianOptimizer, GaussianIRC
from ase.calculators.emt import EMT

from xtb.ase.calculator import XTB

# optimisation
from ase.optimize.bfgslinesearch import BFGSLineSearch as QuasiNetwon

# molecular dynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin

from ase.optimize.minimahopping import PassedMinimum


class AccHopping():

    """
    This method performs accelerated (minima) hopping that automatically 
    finds local minima and transition states under certain thermal conditions. 
    """

    # output files
    traj_name = "miaow.xyz"

    # accelerated MD parameters
    vmax = 0.5 # V_max, eV
    bond_strain_limit = 0.5 # q, maximum bond change compared to the reference state
    P1 = 0.98 # control the curvature near the boundary

    timestep = 1.0 # in fs
    mdstep = 2500


    def __init__(
        self, 
        reactants, 
        temperature,
        mdstep,
        calc, # this is the reference calculator (usually DFT by GAUSSIAN or VASP)
        opt = None, 
        surrogate = None
    ):

        self.beta = 1 / (temperature*kB)

        self.mdstep = mdstep
        self.temperature = temperature

        print("TEMPERATURE: ", self.temperature)
        print("MDSTEP_CYCLE: ", self.mdstep)

        self.calc = calc
        self.opt = opt

        # create initial configuration
        nreactants = len(reactants)
        if nreactants == 1:
            # provide one converged complex as inputs
            self.atoms = reactants[0].copy()
            # TODO: check convergence
            # check number of molecular fragments
            all_tags = set(self.atoms.get_tags())
            print("number of fragments: ", len(all_tags))
        else:
            # generate a random initial structure
            self.atoms = self.__place_random_configuration(reactants)


        # event detection methods
        self._passedminimum = PassedMinimum()

        return

    def __call__(self, step):
        """ run this method
        """

        # check reference state R

        # set MD 
        atoms = self.atoms.copy()
        atoms.calc = self.calc

        self.cur_distance_matrix = self.__calc_bond_list(atoms)
        print("Reference Bond Matrix: ", self.cur_distance_matrix)

        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
        dyn = Langevin(
            atoms, 
            timestep = self.timestep*ase.units.fs, 
            temperature_K = self.temperature, 
            friction = 0.002, # TODO: what is the unit?
            fixcm = True
        )

        for i in range(step):
            # run hyperdynamics
            self.__molecular_dynamics(dyn, self.mdstep)

            # check if event happens
            self.__detect_event()
            if True:
                self.atoms = None # set current system to quenched state Q
                self.__calc_bond_list()
            else:
                self.atoms = None # restore state to dynamics state D

        return
    
    def __place_random_configuration(self, reactants):
        """ randomly place reactants in a given region
        """
        # tag every molecules and state atomice types
        unique_atom_types = []
        for i, atoms in enumerate(reactants):
            atoms.set_tags(i)
            unique_atom_types.extend(atoms.get_atomic_numbers())

        unique_atom_types = list(set(unique_atom_types))
        blmin = closest_distances_generator(
            atom_numbers=unique_atom_types,
            ratio_of_covalent_radii=0.8 # be careful with test too far
        )
        print(blmin)
        
        # place positions
        nreactants = len(reactants)
        candidate = Atoms("")
        cand_com = np.zeros(3) # set to origin
        for i in range(nreactants):
            atoms = reactants[i].copy()
            for dis in np.arange(2.0, 3.0, 0.05): # distance to the atom group
                print("dis: ", dis)
                # Apply a random translation
                pos = dis*np.linalg.norm(np.random.random(3)) + cand_com
                com = atoms.get_center_of_mass()
                atoms.translate(pos - com) # set to the origin
                # Apply a random rotation to multi-atom blocks
                phi, theta, psi = 360 * np.random.random(3)
                atoms.euler_rotate(
                    phi=phi, theta=0.5 * theta, psi=psi,
                    center=pos
                )
                # check distance
                if not atoms_too_close_two_sets(candidate, atoms, blmin):
                    candidate.extend(atoms)
                    break
            else:
                raise ValueError(f"Failed to insert the {i} reactant in 3.0 AA.")
            cand_com = candidate.get_center_of_mass()
        
        # optimise random formed complex
        self.calc.reset()
        #candidate.calc = self.calc
        #dyn = QuasiNetwon(candidate, trajectory=None)
        #dyn.run(fmax=0.05)

        opt = GaussianOptimizer(candidate, self.calc) # force loose 0.0025 (0.064 eV/AA) tight 0.000015
        opt.run(fmax="loose", steps=100, opt="")

        return candidate
    
    def __calc_bond_list(self, atoms):
        """ calculate bond list for comparision
            Can implement more efficient routine in the future...
        """
        # calc distance matrix
        distance_matrix = atoms.get_all_distances(mic=False, vector=False).copy()
        np.fill_diagonal(distance_matrix, -1.) # self-strain will be None, avoid true-divide in strain

        return distance_matrix
    
    def __molecular_dynamics(self, dyn, maxstep):
        """ perform accelrated dynamics (serial global hyperdynamics)
        """
        st = time.time() # start time

        atoms = dyn.atoms # not copy
        natoms = len(atoms)
        tags = atoms.get_tags() # used to discriminate reactants
        com = atoms.get_center_of_mass().copy()
        print("centre of mass at step 0: ", com)
        dis2com = np.linalg.norm(atoms.positions - com, axis=1)
        print("distances to com: ", dis2com)

        write(self.traj_name, atoms)

        energies = [atoms.get_potential_energy()]
        md_forces = atoms.get_forces().copy() # pristine forces + bias forces

        for i in range(maxstep):
            # update positions with MD
            print(f"\n\n===== MD step: {i} =====")
            dyn.step(md_forces)

            # print MD info
            print("potential energy: ", atoms.get_potential_energy())
            print("temperature: ", atoms.get_temperature())

            # write(self.traj_name, atoms, append=True)
            # continue # test brute-force MD

            # check bond distances
            new_bond_matrix = self.__calc_bond_list(atoms)
            bond_strain = (new_bond_matrix - self.cur_distance_matrix) / self.cur_distance_matrix # not abs

            md_forces = atoms.get_forces().copy()

            # boost_method = "JCP2020"
            boost_method = "JACS2014"
            vboost = 0.
            if boost_method == "JCP2020":
                # ----- JCP2020 serial global method ----- 
                # TODO: create graph and only accelerate atom pairs between molecules
                # also, not consider pairs with too long distance

                # extract from bond list
                max_index = np.unravel_index(np.argmax(np.fabs(bond_strain), axis=None), bond_strain.shape)

                ref_bond_disatnce = self.cur_distance_matrix[max_index]
                new_bond_distance = new_bond_matrix[max_index]
                max_bond_strain = bond_strain[max_index]

                content = "bond pairs: {}-{}\n".format(*max_index)
                content += " current maximum bond strain: {:.4f}\n".format(max_bond_strain)
                content += " distances (ref/new): {:.4f}  {:.4f}\n".format(
                    ref_bond_disatnce, new_bond_distance
                )
                print(content)

                if np.fabs(max_bond_strain) < self.bond_strain_limit:
                    # only apply BB on bond with max strain
                    vboost = self.vmax*(1-(max_bond_strain/self.bond_strain_limit)**2)
                    fboost = 2*self.vmax*max_bond_strain/self.bond_strain_limit**2
            
                    # calc md forces
                    bond_positions = atoms.positions[max_index, :]
                    bias_forces = (
                        fboost * (bond_positions[0, :] - bond_positions[1, :]) 
                        / ref_bond_disatnce / new_bond_distance
                    ) # on atom i
                    print("bias forces on atom i: ", bias_forces)

                    md_forces[max_index, :] = np.vstack([bias_forces, -bias_forces]) 

            elif boost_method == "JACS2014":
                # ----- TODO: JACS2014 adaptive bond boost -----
                # uses pre-defined r_eq not compared to reference bond state
                bmshape = bond_strain.shape[0] # bond matrix shape

                # check bond strain to extract valid bond to boost
                max_index = [0, 0]
                max_bond_strain = 1e-8

                valid_pairs = []
                for i in range(bmshape):
                    for j in range(i+1, bmshape):
                        # check bond strain is smaller than limit, and bond distance is close enough
                        # , and bonds should be formed by different reactants
                        if (
                            np.fabs(bond_strain[i, j]) < self.bond_strain_limit 
                            and new_bond_matrix[i, j] < 3.0 # BB not on too long bonds
                            # and (tags[i] != tags[j])
                        ):
                            valid_pairs.append((i,j))
                            # check max
                            if np.fabs(bond_strain[i, j]) > np.fabs(max_bond_strain):
                                max_bond_strain = bond_strain[i, j]
                                max_index = [i, j]
                nbonds = len(valid_pairs)
                if nbonds > 0:
                    print("number of bonds: ", nbonds)
                    print(valid_pairs)
                else:
                    print("no valid bond pairs, skip this step...")
                    continue

                max_index = tuple(max_index)
                ref_bond_disatnce = self.cur_distance_matrix[max_index]
                new_bond_distance = new_bond_matrix[max_index]

                content = "bond pairs: {}-{}\n".format(*max_index)
                content += " current maximum bond strain: {:.4f}\n".format(max_bond_strain)
                content += " distances (ref/new): {:.4f}  {:.4f}\n".format(
                    ref_bond_disatnce, new_bond_distance
                )
                print(content)

                # calculate boost energy
                vboost_separate = self.vmax / nbonds * (1 - (bond_strain / self.bond_strain_limit)**2)
                vboost = 0.
                for (i, j) in valid_pairs:
                    vboost += vboost_separate[i, j]

                fboost = 2*self.vmax/nbonds*bond_strain/self.bond_strain_limit**2

                bond_ratio_square = (max_bond_strain/self.bond_strain_limit)**2
                envelope = (1-bond_ratio_square)*((1-bond_ratio_square)/(1-self.P1**2*bond_ratio_square))

                # calculate bias forces
                for (i, j) in valid_pairs:
                    # use new distance
                    ref_dis = self.cur_distance_matrix[i, j]
                    new_dis = new_bond_matrix[i, j]
                    bond_positions = atoms.positions[(i,j), :]
                    bias_forces = (
                        envelope * fboost[i, j] * (bond_positions[0, :] - bond_positions[1, :])
                        / ref_dis / new_dis
                    )
                    md_forces[(i,j), :] = np.vstack([bias_forces, -bias_forces]) 

                # add extra for max_index
                uuu = max_bond_strain / self.bond_strain_limit
                xxx = 1 - uuu**2
                mmm = 1 - self.P1**2*uuu**2
                bond_positions = atoms.positions[max_index, :]
                envelope_grad = (
                    (-2*uuu * xxx / mmm) + (xxx * (-2*uuu*mmm - xxx*(-2*self.P1**2*uuu)) / mmm**2)
                ) * (
                    (bond_positions[0, :] - bond_positions[1, :]) / ref_bond_disatnce / new_bond_distance / self.bond_strain_limit
                )
                bias_forces = -envelope_grad * vboost
                md_forces[max_index, :] = np.vstack([bias_forces, -bias_forces])

            else:
                raise ValueError(f"Unknown boost method {boost_method}...")
            
            # TODO: add wall potential to avoid molecules moving away
            # harmonic V=1/2k(x-c)^2 F = -k(x-c)*dx/dr
            centre_of_spring = 3.6
            spring_constant = 2000.
            positions = atoms.positions # current positions
            for i in range(natoms):
                dvec = positions[i, :] - com
                dis = np.linalg.norm(dvec)
                if dis > centre_of_spring:
                    wall_force = -spring_constant*np.fabs(dis-centre_of_spring)*dvec/dis
                    md_forces[i, :] += wall_force

            # calculate boost factor
            boost_factor = np.exp(self.beta*vboost)
            print("boost_factor: {:.4f}".format(boost_factor))

            # save trajectory
            write(self.traj_name, atoms, append=True)

            # TODO: check event
            energies.append(atoms.get_potential_energy())
            passedmin = self._passedminimum(energies)
            if passedmin:
                print("Found Minimum!!!")
            else:
                print("NO MINIMUM...")
        
        et = time.time()

        print("Time for amd: {:.4f} s".format(et-st))

        return
    
    def __detect_event(self):
        """ Check a reactive event has passed using various methods
            1. Use minima hopping method JCP2004 similar to ASE one
            2. TODO: JCTC2016
            3. max atomic displacement usually for solid-state systems
        """

        return
    
    def __optimisation(self):
        """ perform optimisation on current structure
            quech the stucture to the nearby local minimum
        """

        return
    
    def __place_bias_on_minima(self):
        """ place bias on given minima
        """

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--temperature", default=673, type=float,
        help="simulation temperature"
    )
    parser.add_argument(
        "-n", "--mdstep", default=100, type=int,
        help="number of mdsteps"
    )

    args = parser.parse_args()

    # load reactants
    reactants = []
    atoms = read("/mnt/scratch2/users/40247882/amdrs/opt/H2CO/H2CO_opt.xyz")
    reactants.append(atoms.copy())
    atoms = read("/mnt/scratch2/users/40247882/amdrs/opt/H2CCHOH/H2CCHOH_opt.xyz")
    reactants.append(atoms.copy())

    # load optimised initial structure
    reactants = []
    atoms = read("/mnt/scratch2/users/40247882/amdrs/rs/test-structures/IS-1_opt.xyz")
    reactants.append(atoms.copy())

    # prepare calculator
    calc = Gaussian(
        directory="Gau-Worker",
        label="opt",
        chk = "MyJob.chk",
        mem = "1GB",
        nprocshared = 4,
        method = "b3lyp",
        # basis = "6-31G",
        basis = "6-31G"
    )
    # calc = EMT()
    # calc = XTB(method="GFN2-xTB")

    rs = AccHopping(reactants, args.temperature, args.mdstep, calc)
    rs(10)