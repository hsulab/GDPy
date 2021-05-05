#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple

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

class ReducedRegion():
    """
    spherical or cubic box
    """

    def __init__(self, cell, mindis=1.0):
        """"""
        # box
        self.cell = cell.copy()
        self.cell[2,2] = 10.5 # TODO: add height

        self.cvect = cell[2] / np.linalg.norm(cell[2])
        self.cmin = 4.5 # 4.5
        self.cmax = 10.5

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
    
    def calc_acc_volume(self, atoms):
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

    def __init__(
        self, 
        type_map: dict, 
        gc_params: namedtuple, 
        atoms, 
        reduced_region: ReducedRegion,
        nMCmoves: int = 10, 
        transition_array: np.array = np.array([0.0,0.0])
    ):
        """
        """
        # simulation system
        self.expart = gc_params.particle # exchangeable particle
        self.atoms = atoms # current atoms
        self.nparts = len(atoms)
        self.nattempts = nMCmoves

        self.region = reduced_region

        # probs
        self.trans_probs = transition_array # transition probabilities: motion, insertion, deletion
        self.accum_probs = np.cumsum(transition_array) / np.sum(transition_array)

        self.maxdisp = 2.0 # angstrom

        self.mc_atoms = atoms.copy() # atoms after MC move

        # set random generator
        self.set_rng()

        # TODO: reservoir
        self.nexatoms = 0
        self.chem_pot = gc_params.mu

        # - beta
        kBT_eV = units.kB * gc_params.temperature
        self.beta = 1./kBT_eV # 1/(kb*T), eV

        # - cubic thermo de broglie 
        hplanck = units._hplanck # J/Hz = kg*m2*s-1
        _mass = data.atomic_masses[data.atomic_numbers[gc_params.particle]] # g/mol
        _mass = _mass * units._amu
        kbT_J = kBT_eV * units._e # J = kg*m2*s-2
        self.cubic_wavelength = (hplanck/np.sqrt(2*np.pi*_mass*kbT_J)*1e10)**3 # thermal de broglie wavelength

        # TODO: reduced region 
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

    def run(self, calc, backend=''):
        """"""
        # start info
        content = '===== Simulation Information =====\n\n'
        content += 'Temperature %.4f [K] Beta %.4f [eV]\n' %(1./self.beta/units.kB, self.beta)
        content += 'Cubic Thermal de Broglie Wavelength %f\n' %self.cubic_wavelength
        content += 'Chemical Potential of %s is %.4f [eV]\n' %(self.expart, self.chem_pot)
        print(content)
        
        # set calculator
        self.calc = calc
        self.calc.reset() # remove info stored in calculator

        #self.atoms.calc = self.calc
        #self.energy_stored = self.atoms.get_potential_energy()
        self.energy_stored = self.optimise(self.atoms)
        print(self.atoms.cell)
        print('energy_stored ', self.energy_stored)

        print('\n\nrenew trajectory file')
        with open('miaow.xyz', 'w') as fopen:
            fopen.write('')

        # start monte carlo
        for idx in range(self.nattempts):
            print('\n\n===== MC Move %04d =====\n' %idx)
            # run standard MC move
            self.step()

            # TODO: save state
            write('miaow.xyz', self.atoms, append=True)

        return

    def step(self):
        """ various actions
        [0]: move, [1]: exchange (insertion/deletion)
        """
        rn_mcmove = self.rng.uniform()
        print('prob action', rn_mcmove)
        # check if the action is valid, otherwise set the prob to zero
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
        energy_after = self.optimise(cur_atoms)

        coef = 1.0
        energy_change = energy_after - self.energy_stored
        acc_ratio = coef * np.exp(-self.beta*(energy_change))

        rn_motion = self.rng.uniform()
        if rn_motion < acc_ratio:
            self.atoms = cur_atoms
            self.energy_stored = energy_after
        else:
            print('fail to move')
            pass
        
        content = '\nCoefficient %.4f Energy Change %.4f [eV]\n' %(coef, energy_change)
        content += 'Translation Probability %.4f\n' %rn_motion
        content += 'Accept Ratio %.4f\n' %acc_ratio
        print(content)

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
        energy_after = self.optimise(cur_atoms)

        # try insert
        coef = self.acc_volume/(self.nexatoms+1)/self.cubic_wavelength
        energy_change = energy_after-self.energy_stored-self.chem_pot
        acc_ratio = np.min([1.0, coef * np.exp(-self.beta*(energy_change))])
        rn_insertion = self.rng.uniform()

        if rn_insertion < acc_ratio:
            self.atoms = cur_atoms
            self.energy_stored = energy_after
            # update exchangeable atoms
            self.exatom_indices = list(range(self.nparts, len(self.atoms)))
        else:
            print('fail to insert...')
            pass

        content = '\nCoefficient %.4f Energy Change %.4f [eV]\n' %(coef, energy_change)
        content += 'Insertion Probability %.4f\n' %rn_insertion
        content += 'Accept Ratio %.4f\n' %acc_ratio
        print(content)

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
        energy_after = self.optimise(cur_atoms)

        coef = self.nexatoms*self.cubic_wavelength/self.acc_volume
        energy_change = energy_after + self.chem_pot - self.energy_stored
        acc_ratio = np.min([1.0, coef*np.exp(-self.beta*(energy_change))])
        rn_deletion = self.rng.uniform()

        if rn_deletion < acc_ratio:
            self.atoms = cur_atoms
            self.energy_stored = energy_after
            # update exchangeable atoms
            self.exatom_indices = list(range(self.nparts, len(self.atoms)))
        else:
            pass

        content = '\nCoefficient %.4f Energy Change %.4f [eV]\n' %(coef, energy_change)
        content += 'Deletion Probability %.4f\n' %rn_deletion
        content += 'Accept Ratio %.4f\n' %acc_ratio
        print(content)

        print('energy_stored is %12.4f' %self.energy_stored)

        return
    
    def optimise(self, atoms):
        """"""
        self.calc.reset()
        atoms.calc = self.calc

        # TODO: check constraint from the region
        cons = FixAtoms(
            indices = [atom.index for atom in atoms if atom.position[2] < self.region.cmin]
        )
        atoms.set_constraint(cons)

        # TODO: use opt as arg
        dyn = BFGS(atoms)
        dyn.run(fmax=0.05, steps=200)
        # TODO: change this to optimisation
        forces = atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 0.05:
            pass
        else:
            print('not converged')
        en = atoms.get_potential_energy()

        return en

def calc_chem_pot():
    """ calculate the chemical potential
    """
    pass

if __name__ == '__main__':
    # set initial structure - bare metal surface
    atoms = read('/users/40247882/projects/oxides/gdp-main/mc-test/Pt_111_0.xyz')

    # set reservior
    pot = 'reax'
    if pot == 'dp':
        # vdW-DF energy
        molecule_energy, dissociation_energy = -9.19578234, (-9.19578234-2*(-1.49092275))
        # 300K, PBE-ZPE, experimental data https://janaf.nist.gov
        thermo_correction = 0.09714 + (8.683 * 0.01036427) - 298.15 * (205.147 * 0.0000103642723)
        # no pressure correction
        #chemical_potential = 0.5*(molecule_energy + thermo_correction - dissociation_energy)
        chemical_potential = 0.5*(molecule_energy + thermo_correction)
    elif pot == 'reax':
        molecule_energy, dissociation_energy = -5.588397899826529, (-5.588397899826529-2*(-0.1086425742653097))
        # 300K, PBE-ZPE, experimental data https://janaf.nist.gov
        thermo_correction = 0.09714 + (8.683 * 0.01036427) - 298.15 * (205.147 * 0.0000103642723)
        # no pressure correction
        chemical_potential = 0.5*(molecule_energy + thermo_correction - dissociation_energy)
    else:
        pass

    Reservior = namedtuple('Reservior', ['particle', 'temperature', 'pressure', 'mu']) # Kelvin, atm, eV
    res = Reservior(particle='O', temperature=300, pressure=1.0, mu=chemical_potential) # Kelvin, atm, eV

    # set region
    region = ReducedRegion(atoms.cell)

    # start mc
    type_map = {'O': 0, 'Pt': 1}
    transition_array = [0.5,0.5] # move and exchange
    gcmc = GCMC(type_map, res, atoms, region, 10000, transition_array)

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

    gcmc.run(calc)
