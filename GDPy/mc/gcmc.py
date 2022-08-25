#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
import shutil

import numpy as np

#from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase import Atom, Atoms
from ase.io import read, write

from ase.constraints import FixAtoms

from GDPy.builder.species import build_species
from GDPy.builder.region import Reservoir, ReducedRegion

from GDPy.utils.command import CustomTimer


class GCMC():

    MCTRAJ = "./traj.xyz"
    MAX_NUM_PER_SPECIES = 10000

    def __init__(
        self, 
        system: dict,
        restart: bool = False,
        probabilities: np.array = np.array([0.0,0.0]),
        random_seed = None,
        *args, **kwargs
    ):
        """
        """
        self.restart = restart

        # set random generator TODO: move to run?
        self.set_rng(random_seed)
        print("RANDOM SEED: ", random_seed)

        self._create_init_system(system)

        # transition probabilities: motion, insertion, deletion
        self.accum_probs = np.cumsum(probabilities) / np.sum(probabilities)

        return

    def set_rng(self, seed=None):
        # - assign random seeds
        if seed is None:
            self.rng = np.random.default_rng()
        elif isinstance(seed, int):
            self.rng = np.random.default_rng(seed)

        return
    
    def _create_init_system(self, params):
        """ system = substrate + region + reservoir
        """
        substrate = params.get("substrate", None)
        region_params = params.get("region", None)
        reservoir_params = params.get("reservoir", None)

        # - substrate atoms
        atoms = read(substrate)
        atoms.set_tags(0)

        # - region + reservoir
        reservoir = Reservoir(**reservoir_params)
        region = ReducedRegion(
            atoms=atoms, reservoir=reservoir, **region_params,
            rng = self.rng
        )

        self.region = region
        self.atoms = region.atoms

        self.exparts = self.region.reservoir.exparts
        self.chem_pot = self.region.reservoir.chem_pot
        self.beta = self.region.reservoir.beta
        self.cubic_wavelength = self.region.reservoir.cubic_wavelength

        self.acc_volume = self.region.calc_acc_volume(self.atoms)

        # NOTE: use tag_list to manipulate both atoms and molecules
        self.tag_list = {}
        for expart in self.region.reservoir.exparts:
            self.tag_list[expart] = []

        # TODO: restart

        return

    def run(self, worker, nattempts):
        """"""
        self.worker = worker

        # start info
        content = "===== Simulation Information @%s =====\n\n" % time.asctime( time.localtime(time.time()) )
        content += str(self.region)
        print(content)
        
        # - opt init structure
        start_index = 0 # TODO: for restart
        self.step_index = start_index
        # TODO: check if energy exists
        self.energy_stored, self.atoms = self.optimise(self.atoms)
        print("Cell Info: ", self.atoms.cell)
        print("energy_stored ", self.energy_stored)

        # add optimised initial structure
        print("\n\nrenew trajectory file")
        write(self.MCTRAJ, self.atoms, append=False)

        # - start monte carlo
        self.step_index = start_index + 1
        for idx in range(start_index+1, nattempts):
            self.step_index = idx
            print("\n\n===== MC Move %04d =====\n" %idx)
            # run standard MC move
            self.step()

            # TODO: save state
            self.atoms.info["step"] = idx
            write(self.MCTRAJ, self.atoms, append=True)

            # check uncertainty
        
        print("\n\nFINISHED PROPERLY @ %s." %time.asctime( time.localtime(time.time()) ))

        return

    def step(self):
        """ various actions
        [0]: move, [1]: exchange (insertion/deletion)
        """
        st = time.time()

        # - choose species
        expart = str(self.rng.choice(self.exparts)) # each element hase same prob to chooose
        print("selected particle: ", expart)
        rn_mcmove = self.rng.uniform()
        print("prob action", rn_mcmove)

        # step for selected type of particles
        nexatoms = len(self.tag_list[expart])
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

        et = time.time()
        print("step time: ", et - st)

        return
    
    def pick_random_atom(self, expart):
        """"""
        nexpart = len(self.tag_list[expart])
        if nexpart == 0:
            idx_pick = None
        else:
            idx_pick = self.rng.choice(self.tag_list[expart])
        #print(idx_pick, type(idx_pick))

        return idx_pick
    
    def update_exlist(self):
        """update the list of exchangeable particles"""
        print("number of particles: ")
        for expart, indices in self.tag_list.items():
            print("{:<4s}  {:<8d}".format(expart, len(indices)))
            print("species tag: ", indices)

        return
    
    def attempt_move_atom(self, expart):
        """"""
        # pick an atom
        self.update_exlist()
        tag_idx_pick = self.pick_random_atom(expart)
        if tag_idx_pick is not None:
            print("select species with tag %d" %tag_idx_pick)
        else:
            print("no exchangeable atoms...")
            return 
        
        # try move
        cur_atoms = self.atoms.copy()

        tags = cur_atoms.get_tags()
        species_indices = [i for i, t in enumerate(tags) if t==tag_idx_pick]

        cur_atoms = self.region.random_position_neighbour(
            cur_atoms, species_indices, operation="move"
        )

        if cur_atoms is None:
            print(f"failed to move after {self.region.MAX_RANDOM_ATTEMPTS} attempts...")
            return

        # move info
        print("origin position: ") 
        for si in species_indices:
            print(self.atoms[si].position)
        print("-> random position: ") 
        for si in species_indices:
            print(cur_atoms[si].position)

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        # - check opt pos
        print("-> optimised position: ") 
        for si in species_indices:
            print(opt_atoms[si].symbol, opt_atoms[si].position)

        # - acceptance ratio
        beta = self.beta[expart]
        cubic_wavelength = self.cubic_wavelength[expart]

        coef = 1.0
        energy_change = energy_after - self.energy_stored
        acc_ratio = np.min([1.0, coef * np.exp(-beta*(energy_change))])

        content = "\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n" %(
            self.acc_volume, len(self.tag_list[expart]), cubic_wavelength, coef
        )
        content += "Energy Change %.4f [eV]\n" %energy_change
        content += "Accept Ratio %.4f\n" %acc_ratio
        print(content)

        rn_motion = self.rng.uniform()
        if rn_motion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
        else:
            print("fail to move")
            pass
        
        print("Translation Probability %.4f" %rn_motion)

        print("energy_stored is %12.4f" %self.energy_stored)
        
        return
    
    def attempt_insert_atom(self, expart):
        """atomic insertion"""
        self.update_exlist()

        # - extend one species
        cur_atoms = self.atoms.copy()
        print("current natoms: ", len(cur_atoms))
        # --- build species and assign tag
        new_species = build_species(expart)

        expart_tag = (self.exparts.index(expart)+1)*self.MAX_NUM_PER_SPECIES
        tag_max = 0
        if len(self.tag_list[expart]) > 0:
            tag_max = np.max(self.tag_list[expart]) - expart_tag
        new_tag = int(expart_tag + tag_max + 1)
        new_species.set_tags(new_tag)

        cur_atoms.extend(new_species)

        print("natoms: ", len(cur_atoms))
        tag_idx_pick = new_tag
        print("try to insert species with tag ", tag_idx_pick)

        tags = cur_atoms.get_tags()
        species_indices = [i for i, t in enumerate(tags) if t==tag_idx_pick]

        cur_atoms = self.region.random_position_neighbour(
            cur_atoms, species_indices, operation="insert"
        )
        if cur_atoms is None:
            print("failed to insert after {self.region.MAX_RANDOM_ATTEMPS} attempts...")
            return
        
        # insert info
        print("inserted position: ")
        for si in species_indices:
            print(cur_atoms[si].symbol, cur_atoms[si].position)

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        # - check opt pos
        print("-> optimised position: ") 
        for si in species_indices:
            print(opt_atoms[si].symbol, opt_atoms[si].position)

        # - acceptance ratio
        nexatoms = len(self.tag_list[expart])
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
            self.tag_list[expart].append(new_tag)
        else:
            print('fail to insert...')
        print('Insertion Probability %.4f' %rn_insertion)

        print('energy_stored is %12.4f' %self.energy_stored)

        return

    def attempt_delete_atom(self, expart):
        """"""
        # pick an atom
        self.update_exlist()
        tag_idx_pick = self.pick_random_atom(expart)
        if tag_idx_pick is not None:
            print("select species with tag %d" %tag_idx_pick)
        else:
            print("no atom can be deleted...")
            return

        # try deletion
        cur_atoms = self.atoms.copy()
        
        tags = cur_atoms.get_tags()
        species_indices = [i for i, t in enumerate(tags) if t==tag_idx_pick]
        del cur_atoms[species_indices]

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        # - acceptance ratio
        nexatoms = len(self.tag_list[expart])
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
            new_expart_list = [t for t in self.tag_list[expart] if t != tag_idx_pick] 
            self.tag_list[expart] = new_expart_list
        else:
            pass

        print('Deletion Probability %.4f' %rn_deletion)

        print('energy_stored is %12.4f' %self.energy_stored)

        return


    def optimise(self, atoms):
        """"""
        print("\n----- DYNAMICS MIN INFO -----\n")
        driver = self.worker.driver
        with CustomTimer(name="run-dynamics"):
            # run minimisation
            tags = atoms.get_tags()
            print("n_cur_atoms: ", len(atoms))
            atoms = driver.run(atoms, 
                read_exists=False, extra_info = None
            )
            atoms.set_tags(tags)

            en = atoms.get_potential_energy()

        return en, atoms


if __name__ == "__main__":
    pass