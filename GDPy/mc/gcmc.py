#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from collections import namedtuple
from pathlib import Path
import shutil

import numpy as np

from ase import units
from ase import data
#from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase import Atom, Atoms
from ase.io import read, write

from ase.optimize import BFGS
from ase.constraints import FixAtoms


"""
# index element mass       ideal-nn rmin rmax prob-add
  0     Ag      196.966543 3        1.5  3.0  0.50
  1     O       15.9994    3        1.5  3.0  0.50

For each element, thermal de broglie wavelengths are needed.
"""

Reservior = namedtuple('Reservior', ['particle', 'temperature', 'pressure', 'mu']) # Kelvin, atm, eV
RunParam = namedtuple('RunParam', ['backend', 'calculator', 'optimiser', 'convergence', 'constraint']) # 

def estimate_chemical_potential(
    temperature: float, 
    pressure: float, # pressure, 1 bar
    total_energy: float,
    zpe: float,
    dU: float,
    dS: float, # entropy
    coef: float = 1.0
) -> float:
    """
    See experimental data
        https://janaf.nist.gov
    Examples
        O2 by ReaxFF
            molecular energy -5.588 atomic energy -0.109
        O2 by vdW-DF spin-polarised 
            molecular energy -9.196 atomic energy -1.491
            ZPE 0.09714 
            dU 8.683 kJ/mol (exp)
            entropy@298.15K 205.147 J/mol (exp)
        For two reservoirs, O and Pt
        Pt + O2 -> aPtO2
        mu_Pt = E_aPtO2 - G_O2
    Formula
        FreeEnergy = E_DFT + ZPE + U(T) + TS + pV
    """
    kJm2eV = units.kJ / units.mol # from kJ/mol to eV
    # 300K, PBE-ZPE, experimental data https://janaf.nist.gov
    temp_correction = zpe + (dU*kJm2eV) - temperature*(dS/1000*kJm2eV)
    pres_correction = units.kB*temperature*np.log(pressure/1.0) # eV
    chemical_potential = coef*(
        total_energy + temp_correction + pres_correction
    )

    return chemical_potential

class ReducedRegion():
    """
    spherical or cubic box
    """

    def __init__(
        self, 
        cell, # (3x3) lattice
        caxis: list, # min and max in z-axis
        mindis: float = 1.5 # minimum distance between atoms
    ):
        """"""
        # box
        self.cell = cell.copy()
        
        assert len(caxis) == 2
        self.cmin, self.cmax = caxis
        self.cell[2,2] = self.cmax 

        self.cvect = cell[2] / np.linalg.norm(cell[2])
        self.volume = np.dot(
            self.cvect*(self.cmax-self.cmin), np.cross(self.cell[0], self.cell[1])
        )

        self.mindis = mindis

        # random generator
        drng = np.random.default_rng()
        self.rng = drng

        return

    def random_position(self, positions: np.ndarray):
        """"""
        for i in range(1000): # maximum number of attempts
            ran_frac_pos = self.rng.uniform(0,1,3)
            ran_pos = np.dot(ran_frac_pos, self.cell) 
            # TODO: better
            ran_pos[2] = self.cmin + ran_frac_pos[2] * (self.cmax-self.cmin)
            if not self.check_overlap(self.mindis,ran_pos, positions):
                print('ran pos', ran_pos)
                break

        return ran_pos
    
    def calc_acc_volume(self, atoms) -> float:
        """calculate acceptable volume"""
        atoms_inside = [atom for atom in atoms if atom.position[2] > self.cmin]
        print("number of atoms inside the region", len(atoms_inside))
        radii = [data.covalent_radii[data.atomic_numbers[atom.symbol]] for atom in atoms_inside]
        atoms_volume = np.sum([4./3.*np.pi*r**3 for r in radii])

        acc_volume = self.volume - atoms_volume # A^3

        return acc_volume

    @staticmethod
    def check_overlap(mindis, ran_pos, positions):
        """"""
        # TODO: change this to the faste neigbour list construction
        # maybe use scipy
        overlapped = False
        for pos in positions:
            # TODO: use neighbour list?
            dis = np.linalg.norm(ran_pos-pos)
            if dis < mindis:
                overlapped = True
                break

        return overlapped


class GCMC():

    MCTRAJ = "./miaow.xyz"
    SUSPECT_DIR = "./suspects"

    def __init__(
        self, 
        type_list: list, 
        reservior: dict, 
        atoms: Atoms, 
        reduced_region: ReducedRegion,
        transition_array: np.array = np.array([0.0,0.0])
    ):
        """
        """
        # elements
        self.type_list = type_list
        self.type_map = {}
        for i, e in enumerate(self.type_list):
            self.type_map[e] = i

        # simulation system
        self.atoms = atoms # current atoms
        self.substrate_natoms = len(atoms)

        self.region = reduced_region

        # constraint
        cons_indices = [atom.index for atom in atoms if atom.position[2] < self.region.cmin]
        cons = FixAtoms(
            indices = cons_indices
        )
        atoms.set_constraint(cons)
        self.constraint = cons
        self.cons_indices = " ".join([str(i+1) for i in cons_indices])

        # probs
        self.trans_probs = transition_array # transition probabilities: motion, insertion, deletion
        self.accum_probs = np.cumsum(transition_array) / np.sum(transition_array)

        self.maxdisp = 2.0 # angstrom, for atom random move

        self.mc_atoms = atoms.copy() # atoms after MC move

        # set random generator
        self.set_rng()

        # TODO: reservoir
        res_format = reservior.get("format", "otf") # on-the-fly calculation
        if res_format == "direct":
            self.temperature, self.pressure = reservior["temperature"], -1000. # NOTE: pressure is None
            self.chem_pot = reservior["particle"]
            self.exparts = list(self.chem_pot.keys())
        else:
            # for single particle reservoir
            self.exparts = [reservior["particle"]] # exchangeable particle
            self.temperature, self.pressure = reservior["temperature"], reservior["pressure"]

            # - chemical potenttial
            self.chem_pot = {}
            self.chem_pot[self.exparts[0]] = estimate_chemical_potential(
                temperature=self.temperature, pressure=self.pressure,
                **reservior["energy_params"]
            )

        # statistical mechanics
        self.beta = {}
        self.cubic_wavelength = {}
        for expart in self.exparts:
            self.beta[expart], self.cubic_wavelength[expart] = self.compute_thermo_wavelength(
                expart, self.temperature
            )

        # reduced region 
        self.acc_volume = self.region.calc_acc_volume(self.atoms)

        # few iterative properties
        self.exatom_indices = {}
        for expart in self.exparts:
            self.exatom_indices[expart] = []

        return
    
    @staticmethod
    def compute_thermo_wavelength(expart: str, temperature: float):
        # - beta
        kBT_eV = units.kB * temperature
        beta = 1./kBT_eV # 1/(kb*T), eV

        # - cubic thermo de broglie 
        hplanck = units._hplanck # J/Hz = kg*m2*s-1
        _mass = data.atomic_masses[data.atomic_numbers[expart]] # g/mol
        _mass = _mass * units._amu
        kbT_J = kBT_eV * units._e # J = kg*m2*s-2
        cubic_wavelength = (hplanck/np.sqrt(2*np.pi*_mass*kbT_J)*1e10)**3 # thermal de broglie wavelength

        return beta, cubic_wavelength
    
    def set_rng(self):
        drng = np.random.default_rng()
        self.rng = drng

        return

    def __register_calculator(self, calc_params: dict):
        """"""
        self.convergence = calc_params.pop("convergence", None)
        self.repeat = calc_params.pop("repeat", 3) # try few times optimisation

        print("\n===== Calculator INFO =====\n")
        model = calc_params["model_params"]["model"]
        if model == "lasp":
            from GDPy.calculator.lasp import LaspNN
            self.calc = LaspNN(**self.calc_dict["kwargs"])
        elif model == "eann": # and inteface to lammps
            from GDPy.calculator.lammps import Lammps
            self.calc = Lammps(
                command = calc_params["command"],
                directory = calc_params["directory"],
                pair_style = calc_params["model_params"]
            )
        else:
            raise ValueError("Unknown potential to calculation...")
        
        backend = calc_params.pop("backend", None)
        if backend == "ase":
            from GDPy.calculator.ase_interface import AseDynamics
            self.worker = AseDynamics(self.calc, directory=self.calc.directory)
        elif backend == "lammps":
            from GDPy.calculator.lammps import LmpDynamics
            # use lammps optimisation
            self.worker = LmpDynamics(
                self.calc, directory=self.calc.directory
            )
        else:
            raise ValueError("Unknown interface to optimisation...")

        # TODO: check uncertainty control
        self.devi_tol = calc_params.pop("devi_tol", 0.015) # try few times optimisation
        calc_params["directory"] = Path(calc_params["directory"])

        # find uncertainty support
        model_files = calc_params["model_params"]["file"]
        if isinstance(model_files, str):
            num_models = 0
        else:
            num_models = len(model_files)
            calc_params["model_params"]["file"] = " ".join(model_files)
        self.suspects = None
        if num_models > 1:
            self.suspects = Path(self.SUSPECT_DIR)
            if self.suspects.exists():
                raise FileExistsError("{} already exists...".format(self.suspects))
            else:
                self.suspects.mkdir()

        return

    def run(
        self, nattempts,
        params: dict
    ):
        """"""
        # start info
        content = "===== Simulation Information @%s =====\n\n" % time.asctime( time.localtime(time.time()) )
        content += 'Temperature %.4f [K] Pressure %.4f [atm]\n' %(self.temperature, self.pressure)
        for expart in self.exparts:
            content += "--- %s ---\n" %expart
            content += 'Beta %.4f [eV-1]\n' %(self.beta[expart])
            content += 'Cubic Thermal de Broglie Wavelength %f\n' %self.cubic_wavelength[expart]
            content += 'Chemical Potential of is %.4f [eV]\n' %self.chem_pot[expart]
        print(content)
        
        # set calculator
        self.__register_calculator(params)

        # opt init structure
        self.step_index = "-init"
        self.energy_stored, self.atoms = self.optimise(self.atoms)
        print(self.atoms.cell)
        print('energy_stored ', self.energy_stored)

        # add optimised initial structure
        print('\n\nrenew trajectory file')
        write(self.MCTRAJ, self.atoms, append=False)

        # start monte carlo
        self.step_index = 0
        for idx in range(nattempts):
            self.step_index = idx
            print('\n\n===== MC Move %04d =====\n' %idx)
            # run standard MC move
            self.step()

            # TODO: save state
            write(self.MCTRAJ, self.atoms, append=True)

            # check uncertainty
        
        print("\n\nFINISHED PROPERLY @ %s." %time.asctime( time.localtime(time.time()) ))

        return

    def step(self):
        """ various actions
        [0]: move, [1]: exchange (insertion/deletion)
        """
        expart = self.rng.choice(self.exparts) # each element hase same prob to chooose
        print("selected particle: ", expart)
        rn_mcmove = self.rng.uniform()
        print("prob action", rn_mcmove)

        # step for selected type of particles
        nexatoms = len(self.exatom_indices[expart])
        if nexatoms > 0:
            if rn_mcmove < self.accum_probs[0]:
                # atomic motion
                print('current attempt is *motion*')
                self.attempt_move_atom(expart)
            elif rn_mcmove < self.accum_probs[1]:
                # exchange (insertion/deletion)
                rn_ex = self.rng.uniform()
                print("prob exchange", rn_ex)
                if rn_ex < 0.5:
                    print('current attempt is *insertion*')
                    self.attempt_insert_atom(expart)
                else:
                    print("current attempt is *deletion*")
                    self.attempt_delete_atom(expart)
            else:
                pass # never execute here
        else:
            print('current attempt is *insertion*')
            self.attempt_insert_atom(expart)

        return
    
    def pick_random_atom(self, expart):
        """"""
        nexpart = len(self.exatom_indices[expart])
        if nexpart == 0:
            idx_pick = None
        else:
            idx_pick = self.rng.choice(self.exatom_indices[expart])
        #print(idx_pick, type(idx_pick))

        return idx_pick
    
    def update_exlist(self):
        """update the list of exchangeable particles"""
        print("number of particles: ")
        for expart, indices in self.exatom_indices.items():
            print("{:<4s}  {:<8d}".format(expart, len(indices)))
            print(indices)

        return
    
    def attempt_move_atom(self, expart):
        """"""
        # pick an atom
        self.update_exlist()
        idx_pick = self.pick_random_atom(expart)
        if idx_pick is not None:
            print('select atom with index of %d' %idx_pick)
        else:
            print('no exchangeable atoms...')
            return 
        
        # try move
        cur_atoms = self.atoms.copy()
        pos = cur_atoms[idx_pick].position.copy()

        MAX_MOVE_ATTEMPTS = 100
        for idx in range(MAX_MOVE_ATTEMPTS):
            # get random motion vector
            rsq = 1.1
            while (rsq > 1.0):
                rvec = 2*self.rng.uniform(size=3) - 1.0
                rsq = np.linalg.norm(rvec)
            pos = pos + rvec*self.maxdisp

            # TODO: check if conflict with nerighbours
            overlapped = self.region.check_overlap(
                self.region.mindis, pos, cur_atoms.positions
            )
            if not overlapped:
                print("find suitable position for moving...")
                break
        else:
            print("Fail to move...")
            return

        cur_atoms[idx_pick].position = pos

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        beta = self.beta[expart]
        cubic_wavelength = self.cubic_wavelength[expart]

        coef = 1.0
        energy_change = energy_after - self.energy_stored
        acc_ratio = np.min([1.0, coef * np.exp(-beta*(energy_change))])

        content = '\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n' %(
            self.acc_volume, len(self.exatom_indices[expart]), cubic_wavelength, coef
        )
        content += 'Energy Change %.4f [eV]\n' %energy_change
        content += 'Accept Ratio %.4f\n' %acc_ratio
        print(content)

        rn_motion = self.rng.uniform()
        if rn_motion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
        else:
            print('fail to move')
            pass
        
        print('Translation Probability %.4f' %rn_motion)

        print('energy_stored is %12.4f' %self.energy_stored)
        
        return
    
    def attempt_insert_atom(self, expart):
        """atomic insertion"""
        self.update_exlist()
        # only one element for now
        cur_atoms = self.atoms.copy()
        ran_pos = self.region.random_position(cur_atoms.positions)
        extra_atom = Atom(expart, position=ran_pos)
        cur_atoms.extend(extra_atom)

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        nexatoms = len(self.exatom_indices[expart])
        beta = self.beta[expart]
        chem_pot = self.chem_pot[expart]
        cubic_wavelength = self.cubic_wavelength[expart]

        # try insert
        coef = self.acc_volume/(nexatoms+1)/cubic_wavelength
        energy_change = energy_after-self.energy_stored-chem_pot
        acc_ratio = np.min([1.0, coef * np.exp(-beta*(energy_change))])

        content = '\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n' %(
            self.acc_volume, nexatoms, cubic_wavelength, coef
        )
        content += 'Energy Change %.4f [eV]\n' %energy_change
        content += 'Accept Ratio %.4f\n' %acc_ratio
        print(content)

        rn_insertion = self.rng.uniform()
        if rn_insertion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
            # update exchangeable atoms
            self.exatom_indices[expart].append(len(self.atoms)-1)
        else:
            print('fail to insert...')
        print('Insertion Probability %.4f' %rn_insertion)

        print('energy_stored is %12.4f' %self.energy_stored)

        return

    def attempt_delete_atom(self, expart):
        """"""
        # pick an atom
        self.update_exlist()
        idx_pick = self.pick_random_atom(expart)
        if idx_pick is not None:
            print('select atom with index of %d' %idx_pick)
        else:
            print('no atom can be deleted...')
            return

        # try deletion
        cur_atoms = self.atoms.copy()
        del cur_atoms[idx_pick]

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        nexatoms = len(self.exatom_indices[expart])
        beta = self.beta[expart]
        cubic_wavelength = self.cubic_wavelength[expart]
        chem_pot = self.chem_pot[expart]

        coef = nexatoms*cubic_wavelength/self.acc_volume
        energy_change  = energy_after + chem_pot - self.energy_stored
        acc_ratio = np.min([1.0, coef*np.exp(-beta*(energy_change))])

        content = '\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n' %(
            self.acc_volume, nexatoms, cubic_wavelength, coef
        )
        content += 'Energy Change %.4f [eV]\n' %energy_change
        content += 'Accept Ratio %.4f\n' %acc_ratio
        print(content)

        rn_deletion = self.rng.uniform()
        if rn_deletion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
            # reformat exchangeable atoms
            # self.exatom_indices[expart].append(len(self.atoms)-1)
            self.exatoms_indices = {}
            for expart in self.exparts:
                self.exatom_indices[expart] = []
            chemical_symbols = self.atoms.get_chemical_symbols()
            for i in range(self.substrate_natoms, len(self.atoms)):
                self.exatom_indices[chemical_symbols[i]].append(i)
        else:
            pass

        print('Deletion Probability %.4f' %rn_deletion)

        print('energy_stored is %12.4f' %self.energy_stored)

        return


    def optimise(self, atoms):
        """"""
        self.worker.reset()
        self.worker.set_output_path(self.calc.directory)
        old_calc_dir = self.calc.directory

        repeat = 3
        fmax, steps = self.convergence
        
        for i in range(repeat):
            min_atoms, min_results = self.worker.minimise(
                atoms,
                fmax=fmax, steps=steps,
                constraint = self.cons_indices # for lammps
            )

            print("\n----- DYNAMICS MIN INFO -----\n")
            print(min_results) # TODO: extract force info from output

            # TODO: change this to optimisation
            forces = min_atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force < fmax:
                atoms = min_atoms
                print("minimisation converged...")
                if self.suspects is not None:
                    print("check uncertainty...")
                    self.calc.directory = self.suspects / ("step" + str(self.step_index))
                    # TODO: use potential manager
                    min_atoms.calc = self.calc
                    __dummy = min_atoms.get_forces()
                    devi_file = Path(self.calc.directory) / "model_devi.out"
                    devi_info = np.loadtxt(devi_file) # EANN
                    en_devi = float(devi_info[1])
                    print("total energy deviation: {:.4f}  tolerance: {:.4f}".format(en_devi, self.devi_tol*len(min_atoms)))
                    if en_devi > self.devi_tol*len(min_atoms) :
                        print("large deviation, and save trajectory...")
                        shutil.copytree(old_calc_dir, self.suspects / ("step" + str(self.step_index)+"-traj"))
                break
            else:
                atoms = min_atoms
        else:
            print("minimisation failed, use lateset energy...")

        self.calc.directory = old_calc_dir

        en = atoms.get_potential_energy()

        return en, atoms
    
    @staticmethod
    def __parse_deviation(devi_file):
        # TODO: check deviation
        # max_fdevi = np.loadtxt(devi_out)[1:,4] # DP
        max_fdevi = np.loadtxt(devi_file)[1:,5] # EANN

        return


if __name__ == '__main__':
    pass