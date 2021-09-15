#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
from sys import prefix

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
        print(len(atoms_inside))
        radii = [data.covalent_radii[data.atomic_numbers[atom.symbol]] for atom in atoms_inside]
        atoms_volume = np.sum([4./3.*np.pi*r**3 for r in radii])

        acc_volume = self.volume - atoms_volume # A^3

        return acc_volume

    @staticmethod
    def check_overlap(mindis, ran_pos, positions):
        """"""
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
        # simulation system
        self.atoms = atoms # current atoms
        self.nparts = len(atoms)

        self.region = reduced_region

        # constraint
        cons = FixAtoms(
            indices = [atom.index for atom in atoms if atom.position[2] < self.region.cmin]
        )
        atoms.set_constraint(cons)

        # probs
        self.trans_probs = transition_array # transition probabilities: motion, insertion, deletion
        self.accum_probs = np.cumsum(transition_array) / np.sum(transition_array)

        self.maxdisp = 2.0 # angstrom

        self.mc_atoms = atoms.copy() # atoms after MC move

        # set random generator
        self.set_rng()

        # TODO: reservoir
        self.nexatoms = 0
        self.expart = reservior["particle"] # exchangeable particle

        self.temperature, self.pressure = reservior["temperature"], reservior["pressure"]

        # - chemical potenttial
        self.chem_pot = estimate_chemical_potential(
            temperature=self.temperature, pressure=self.pressure,
            **reservior["energy_params"]
        )

        # - beta
        kBT_eV = units.kB * self.temperature
        self.beta = 1./kBT_eV # 1/(kb*T), eV

        # - cubic thermo de broglie 
        hplanck = units._hplanck # J/Hz = kg*m2*s-1
        _mass = data.atomic_masses[data.atomic_numbers[self.expart]] # g/mol
        _mass = _mass * units._amu
        kbT_J = kBT_eV * units._e # J = kg*m2*s-2
        self.cubic_wavelength = (hplanck/np.sqrt(2*np.pi*_mass*kbT_J)*1e10)**3 # thermal de broglie wavelength

        # reduced region 
        self.acc_volume = self.region.calc_acc_volume(self.atoms)

        # few iterative properties
        self.exatom_indices = []

        return
    
    @staticmethod
    def compute_thermo_wavelength():
        return
    
    def set_rng(self):
        drng = np.random.default_rng()
        self.rng = drng

        return

    def run(
        self, nattempts,
        params: dict
    ):
        """"""
        # start info
        content = "===== Simulation Information =====\n\n"
        content += 'Temperature %.4f [K] Pressure %.4f [atm]\n' %(self.temperature, self.pressure)
        content += 'Beta %.4f [eV-1]\n' %(self.beta)
        content += 'Cubic Thermal de Broglie Wavelength %f\n' %self.cubic_wavelength
        content += 'Chemical Potential of %s is %.4f [eV]\n' %(self.expart, self.chem_pot)
        print(content)
        
        # set calculator
        self.backend = params.pop("backend", None)
        self.convergence = params.pop("convergence", None)
        self.constraint = None
        if self.backend == "lammps":
            from GDPy.calculator.reax import LMPMin
            self.calc = LMPMin(**params) # change this to class with same methods as ASE-Dyn
            self.calc.reset() # remove info stored in calculator
        else:
            pass

        # opt init structure
        self.energy_stored, self.atoms = self.optimise(self.atoms)
        print(self.atoms.cell)
        print('energy_stored ', self.energy_stored)

        print('\n\nrenew trajectory file')
        with open(self.MCTRAJ, "w") as fopen:
            fopen.write('')

        # start monte carlo
        for idx in range(nattempts):
            print('\n\n===== MC Move %04d =====\n' %idx)
            # run standard MC move
            self.step()

            # TODO: save state
            write(self.MCTRAJ, self.atoms, append=True)

            # check uncertainty

        return

    def step(self):
        """ various actions
        [0]: move, [1]: exchange (insertion/deletion)
        """
        rn_mcmove = self.rng.uniform()
        print('prob action', rn_mcmove)
        # check if the action is valid, otherwise set the prob to zero
        if len(self.exatom_indices) > 0:
            if rn_mcmove < self.accum_probs[0]:
                # atomic motion
                print('current attempt is motion')
                self.attempt_move_atom()
            elif rn_mcmove < self.accum_probs[1]:
                # exchange (insertion/deletion)
                rn_ex = self.rng.uniform()
                print('prob exchange', rn_ex)
                if rn_ex < 0.5:
                    print('current attempt is insertion')
                    self.attempt_insert_atom()
                else:
                    print('current attempt is deletion')
                    self.attempt_delete_atom()
        else:
            # exchange (insertion/deletion)
            rn_ex = self.rng.uniform()
            print('prob exchange', rn_ex)
            if rn_ex < 0.5:
                print('current attempt is insertion')
                self.attempt_insert_atom()
            else:
                print('current attempt is deletion')

        return
    
    def pick_random_atom(self):
        """"""
        if self.nexatoms == 0:
            idx_pick = None
        else:
            idx_pick = self.rng.choice(self.exatom_indices)
        #print(idx_pick, type(idx_pick))

        return idx_pick
    
    def update_exlist(self):
        """update the list of exchangeable particles"""
        self.nexatoms = len(self.exatom_indices)
        print('number of particles: ', self.nexatoms, self.exatom_indices)

        return
    
    def attempt_move_atom(self):
        """"""
        # pick an atom
        self.update_exlist()
        idx_pick = self.pick_random_atom()
        if idx_pick is not None:
            print('select atom with index of %d' %idx_pick)
        else:
            print('no exchangeable atoms...')
            return 
        
        # try move
        cur_atoms = self.atoms.copy()
        pos = cur_atoms[idx_pick].position.copy()

        for idx in range(1):
            # get random motion vector
            rsq = 1.1
            while (rsq > 1.0):
                rvec = 2*self.rng.uniform(size=3) - 1.0
                rsq = np.linalg.norm(rvec)
            pos = pos + rvec*self.maxdisp
            # TODO: check if conflict with nerighbours
            # self.region.check_overlap(1.0, pos, )
        cur_atoms[idx_pick].position = pos

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        coef = 1.0
        energy_change = energy_after - self.energy_stored
        acc_ratio = coef * np.exp(-self.beta*(energy_change))

        content = '\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n' %(
            self.acc_volume, self.nexatoms, self.cubic_wavelength, coef
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
    
    def attempt_insert_atom(self):
        """atomic insertion"""
        self.update_exlist()
        # only one element for now
        cur_atoms = self.atoms.copy()
        ran_pos = self.region.random_position(cur_atoms.positions)
        extra_atom = Atom(self.expart, position=ran_pos)
        cur_atoms.extend(extra_atom)

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        # try insert
        coef = self.acc_volume/(self.nexatoms+1)/self.cubic_wavelength
        energy_change = energy_after-self.energy_stored-self.chem_pot
        acc_ratio = np.min([1.0, coef * np.exp(-self.beta*(energy_change))])

        content = '\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n' %(
            self.acc_volume, self.nexatoms, self.cubic_wavelength, coef
        )
        content += 'Energy Change %.4f [eV]\n' %energy_change
        content += 'Accept Ratio %.4f\n' %acc_ratio
        print(content)

        rn_insertion = self.rng.uniform()
        if rn_insertion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
            # update exchangeable atoms
            self.exatom_indices = list(range(self.nparts, len(self.atoms)))
        else:
            print('fail to insert...')
            pass
        print('Insertion Probability %.4f' %rn_insertion)

        print('energy_stored is %12.4f' %self.energy_stored)

        return

    def attempt_delete_atom(self):
        """"""
        # pick an atom
        self.update_exlist()
        idx_pick = self.pick_random_atom()
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

        coef = self.nexatoms*self.cubic_wavelength/self.acc_volume
        energy_change  = energy_after + self.chem_pot - self.energy_stored
        acc_ratio = np.min([1.0, coef*np.exp(-self.beta*(energy_change))])

        content = '\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n' %(
            self.acc_volume, self.nexatoms, self.cubic_wavelength, coef
        )
        content += 'Energy Change %.4f [eV]\n' %energy_change
        content += 'Accept Ratio %.4f\n' %acc_ratio
        print(content)

        rn_deletion = self.rng.uniform()
        if rn_deletion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
            # update exchangeable atoms
            self.exatom_indices = list(range(self.nparts, len(self.atoms)))
        else:
            pass

        print('Deletion Probability %.4f' %rn_deletion)

        print('energy_stored is %12.4f' %self.energy_stored)

        return
    
    def optimise(self, atoms):
        """"""
        self.calc.reset()
        atoms.calc = self.calc

        if self.constraint is not None:
            atoms.set_constraint(self.constraint)

        fmax, steps = self.convergence
        if self.backend == "ase":
            # TODO: use opt as arg
            dyn = BFGS(atoms)
            dyn.run(fmax=fmax, steps=steps)
        elif self.backend == "lammps":
            # run lammps
            atoms, min_stat = atoms.calc.minimise(atoms, steps=steps, fmax=fmax, zmin=self.region.cmin)
            print(min_stat)

        # TODO: change this to optimisation
        forces = atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 0.05:
            pass
        else:
            print('not converged')
        en = atoms.get_potential_energy()

        return en, atoms


if __name__ == '__main__':
    # set initial structure - bare metal surface
    atoms = read('/users/40247882/projects/oxides/gdp-main/mc-test/Pt_111_0.xyz')

    # set reservior
    pot = 'reax'
    if pot == 'dp':
        # vdW-DF spin-polarised energy
        molecule_energy, dissociation_energy = -9.19578234, (-9.19578234-2*(-1.49092275))
        # 300K, PBE-ZPE, experimental data https://janaf.nist.gov
        thermo_correction = 0.09714 + (8.683 * 0.01036427) - 298.15 * (205.147 * 0.0000103642723)
        # no pressure correction
        #chemical_potential = 0.5*(molecule_energy + thermo_correction - dissociation_energy)
        chemical_potential = 0.5*(molecule_energy + thermo_correction)
    elif pot == 'reax':
        #molecule_energy, dissociation_energy = -5.588397899826529, (-5.588397899826529-2*(-0.1086425742653097))
        molecule_energy, dissociation_energy = -5.588397899826529, -2*(-0.1086425742653097)
        # 300K, PBE-ZPE, experimental data https://janaf.nist.gov
        thermo_correction = 0.09714 + (8.683 * 0.01036427) - 298.15 * (205.147 * 0.0000103642723)
        # no pressure correction
        chemical_potential = 0.5*(molecule_energy + thermo_correction - dissociation_energy)
    else:
        pass

    res = Reservior(particle='O', temperature=300, pressure=1.0, mu=chemical_potential) # Kelvin, atm, eV

    # set region
    region = ReducedRegion(atoms.cell)

    # start mc
    type_map = {'O': 0, 'Pt': 1}
    transition_array = [0.5,0.5] # move and exchange
    gcmc = GCMC(type_map, res, atoms, region, 1000, transition_array)

    # set calculator
    backend = 'lammps'
    if backend == 'ase':
        from GDPy.calculator.dp import DP
        calc = DP(
            type_dict = type_map,
            model = '/users/40247882/projects/oxides/gdp-main/it-0005/ensemble/model-0/graph.pb'
        )
    elif backend == 'lammps':
        # reaxff uses real unit, force kcal/mol/A
        from GDPy.calculator.reax import ReaxLMP
        calc = ReaxLMP(
            directory = 'reax-worker',
            command='mpirun -n 1 lmp'
        )
    else:
        pass

    # TODO: check constraint from the region
    cons = FixAtoms(
        indices = [atom.index for atom in atoms if atom.position[2] < region.cmin]
    )

    run_params = RunParam(backend='reax', calculator=calc, optimiser='lammps', constraint=cons)

    gcmc.run(run_params)
