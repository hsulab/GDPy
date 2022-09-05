#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import time
import logging
from pathlib import Path
import shutil

import numpy as np

#from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase import Atom, Atoms
from ase.io import read, write

from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from GDPy.builder.species import build_species
from GDPy.builder.region import Reservoir, ReducedRegion

from GDPy.utils.command import CustomTimer
from GDPy.md.md_utils import force_temperature


class GCMC():

    MCTRAJ = "./traj.xyz"
    MAX_NUM_PER_SPECIES = 10000

    pfunc = print

    _directory = Path.cwd()

    def __init__(
        self, 
        system: dict,
        restart: bool = False,
        probabilities: np.array = np.array([0.0,0.0]),
        preprocess = {},
        postprocess = {},
        run = {},
        random_seed = None,
        directory = Path.cwd()/"results",
        *args, **kwargs
    ):
        """
        """
        self.directory = directory
        self.restart = restart
        self._init_logger()

        self.preprocess_params = copy.deepcopy(preprocess)
        self.postprocess_params = copy.deepcopy(postprocess)
        self.run_params = copy.deepcopy(run)

        # set random generator TODO: move to run?
        self.set_rng(random_seed)
        self.pfunc(f"RANDOM SEED: {random_seed}")

        self._create_init_system(system)

        # transition probabilities: motion, insertion, deletion
        self.accum_probs = np.cumsum(probabilities) / np.sum(probabilities)

        return
    
    @property
    def directory(self):

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        # - create main dir
        directory_ = Path(directory_)
        if not directory_.exists():
            directory_.mkdir() # NOTE: ./tmp_folder
        else:
            pass
        self._directory = directory_

        return

    def _init_logger(self):
        """"""
        self.logger = logging.getLogger(__name__)

        log_level = logging.INFO

        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        working_directory = self.directory
        log_fpath = working_directory / (self.__class__.__name__+".out")

        if self.restart:
            fh = logging.FileHandler(filename=log_fpath, mode="a")
        else:
            fh = logging.FileHandler(filename=log_fpath, mode="w")

        fh.setLevel(log_level)
        #fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        #ch.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        self.pfunc = self.logger.info

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

        # NOTE: use tag_list to manipulate both atoms and molecules
        self.tag_list = self.region.tag_list

        # TODO: restart

        return

    def run(self, worker, nattempts):
        """ one step = preprocess + mc_attempts
            mc_attempt = move/exchange + postprocess + metropolis
        """
        # --- prepare drivers
        self.worker = worker

        if self.preprocess_params:
            self.pre_driver = self.worker.potter.create_driver(self.preprocess_params)
            assert self.pre_driver.init_params["task"] == "md", "Preprocess should be molecualr dynamics."
        else:
            self.pre_driver = None
        
        self.post_driver = self.worker.potter.create_driver(self.postprocess_params)
        if self.post_driver.get("steps") > 0:
            self.pfunc("Use biased monte carlo...")
            self.acc_volume = self.region.calc_acc_volume(self.atoms)
        else:
            self.pfunc("Use standard monte carlo...")
            self.acc_volume = self.region.volume

        self.MCTRAJ = self.directory/self.MCTRAJ

        # --- start info
        content = "===== Simulation Information @%s =====\n\n" % time.asctime( time.localtime(time.time()) )
        content += str(self.region)
        self.pfunc(content)
        
        # - opt init structure
        start_index = 0 # TODO: for restart
        self.step_index = start_index

        self.pfunc("\n\n===== Initial Minimisation =====")
        # TODO: check if energy exists
        self.energy_stored, self.atoms = self.optimise(self.atoms)
        #self.pfunc(f"Cell Info: {self.atoms.cell}")
        self.pfunc(f"energy_stored {self.energy_stored}")

        # add optimised initial structure
        self.pfunc("\n\nrenew trajectory file")
        write(self.MCTRAJ, self.atoms, append=False)

        # - start monte carlo
        self.step_index = start_index
        for idx in range(start_index, nattempts):
            self.step_index = idx
            self.pfunc(f"\n\n===== MC Cycle {idx:>8d} =====\n")

            # Molecular Dynamics, for hybrid GC-MD+MC
            self._pre_process()

            # run standard MC move/exchange
            for j in range(self.run_params.get("nmcattempts", 1)):
                self.pfunc(f"\n=== MC Attempt {j:>8d} ===")
                self.step()

                # Metropolis

                # TODO: save state
                self.atoms.info["mcstep"] = idx
                self.atoms.info["step"] = j
                write(self.MCTRAJ, self.atoms, append=True)

            # check uncertainty
        
        self.pfunc(f"\n\nFINISHED PROPERLY @ {time.asctime( time.localtime(time.time()) )}")

        return

    def step(self):
        """ various actions
        [0]: move, [1]: exchange (insertion/deletion)
        """
        self.pfunc("\n----- Monte Carlo INFO -----\n")

        st = time.time()

        # - choose species
        expart = str(self.rng.choice(self.exparts)) # each element hase same prob to chooose
        self.pfunc(f"selected particle: {expart}")

        # step for selected type of particles
        nexatoms = len(self.tag_list[expart])
        if nexatoms > 0:
            rn_mcmove = self.rng.uniform()
            self.pfunc(f"prob action {rn_mcmove}")
            if rn_mcmove < self.accum_probs[0]:
                # atomic motion
                self.pfunc("current attempt is *motion*")
                self.attempt_move_atom(expart)
            elif rn_mcmove < self.accum_probs[1]:
                # exchange (insertion/deletion)
                rn_ex = self.rng.uniform()
                self.pfunc(f"prob exchange {rn_ex}")
                if rn_ex < 0.5:
                    self.pfunc("current attempt is *insertion*")
                    self.attempt_insert_atom(expart)
                else:
                    self.pfunc("current attempt is *deletion*")
                    self.attempt_delete_atom(expart)
            else:
                pass # never execute here
        else:
            self.pfunc("current attempt is *insertion*")
            self.attempt_insert_atom(expart)

        et = time.time()
        self.pfunc(f"step time: {et-st}")

        return
    
    def pick_random_atom(self, expart):
        """"""
        nexpart = len(self.tag_list[expart])
        if nexpart == 0:
            idx_pick = None
        else:
            idx_pick = self.rng.choice(self.tag_list[expart])
        #self.pfunc(idx_pick, type(idx_pick))

        return idx_pick
    
    def update_exlist(self):
        """update the list of exchangeable particles"""
        self.pfunc("number of particles: ")
        for expart, indices in self.tag_list.items():
            self.pfunc("{:<4s} number: {:<8d}".format(expart, len(indices)))
            #self.pfunc(f"species tag: {indices}")

        return
    
    def attempt_move_atom(self, expart):
        """"""
        # pick an atom
        self.update_exlist()
        tag_idx_pick = self.pick_random_atom(expart)
        if tag_idx_pick is not None:
            self.pfunc("select species with tag %d" %tag_idx_pick)
        else:
            self.pfunc("no exchangeable atoms...")
            return 
        
        # try move
        cur_atoms = self.atoms.copy()

        tags = cur_atoms.get_tags()
        species_indices = [i for i, t in enumerate(tags) if t==tag_idx_pick]

        cur_atoms = self.region.random_position_neighbour(
            cur_atoms, species_indices, operation="move"
        )

        if cur_atoms is None:
            self.pfunc(f"failed to move after {self.region.MAX_RANDOM_ATTEMPTS} attempts...")
            return

        # move info
        self.pfunc("origin position: ") 
        for si in species_indices:
            self.pfunc(self.atoms[si].position)
        self.pfunc("-> random position: ") 
        for si in species_indices:
            self.pfunc(cur_atoms[si].position)

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        # - check opt pos
        self.pfunc("-> optimised position: ") 
        for si in species_indices:
            self.pfunc(f"{opt_atoms[si].symbol} {opt_atoms[si].position}")

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
        self.pfunc(content)

        rn_motion = self.rng.uniform()
        if rn_motion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
        else:
            self.pfunc("fail to move")
            pass
        
        self.pfunc("Translation Probability %.4f" %rn_motion)

        self.pfunc("energy_stored is %12.4f" %self.energy_stored)
        
        return
    
    def attempt_insert_atom(self, expart):
        """atomic insertion"""
        self.update_exlist()

        # - extend one species
        cur_atoms = self.atoms.copy()
        self.pfunc(f"current natoms: {len(cur_atoms)}")
        # --- build species and assign tag
        new_species = build_species(expart)
        #force_temperature(new_species, temperature=self.region.reservoir.temperature, unit="K")
        MaxwellBoltzmannDistribution(new_species, temperature_K=self.region.reservoir.temperature, rng=self.rng)

        expart_tag = (self.exparts.index(expart)+1)*self.MAX_NUM_PER_SPECIES
        tag_max = 0
        if len(self.tag_list[expart]) > 0:
            tag_max = np.max(self.tag_list[expart]) - expart_tag
        new_tag = int(expart_tag + tag_max + 1)
        new_species.set_tags(new_tag)

        cur_atoms.extend(new_species)

        self.pfunc(f"natoms: {len(cur_atoms)}")
        tag_idx_pick = new_tag
        self.pfunc(f"try to insert species with tag {tag_idx_pick}")
        self.pfunc("momenta:")
        momenta = new_species.get_momenta()
        self.pfunc(f"{momenta}")

        tags = cur_atoms.get_tags()
        species_indices = [i for i, t in enumerate(tags) if t==tag_idx_pick]

        cur_atoms = self.region.random_position_neighbour(
            cur_atoms, species_indices, operation="insert"
        )
        if cur_atoms is None:
            self.pfunc("failed to insert after {self.region.MAX_RANDOM_ATTEMPS} attempts...")
            return
        
        # insert info
        self.pfunc("inserted position: ")
        for si in species_indices:
            self.pfunc(f"{cur_atoms[si].symbol} {cur_atoms[si].position}")

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        # - check opt pos
        self.pfunc("-> optimised position: ") 
        for si in species_indices:
            self.pfunc(f"{opt_atoms[si].symbol} {opt_atoms[si].position}")

        # - acceptance ratio
        nexatoms = len(self.tag_list[expart])
        beta = self.beta[expart]
        chem_pot = self.chem_pot[expart]
        cubic_wavelength = self.cubic_wavelength[expart]

        # try insert
        coef = self.acc_volume/(nexatoms+1)/cubic_wavelength
        energy_change = energy_after-self.energy_stored-chem_pot
        acc_ratio = np.min([1.0, coef * np.exp(-beta*(energy_change))])

        content = "\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n" %(
            self.acc_volume, nexatoms, cubic_wavelength, coef
        )
        content += "Energy Change %.4f [eV]\n" %energy_change
        content += "Accept Ratio %.4f\n" %acc_ratio
        self.pfunc(content)

        rn_insertion = self.rng.uniform()
        if rn_insertion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
            # update exchangeable atoms
            self.tag_list[expart].append(new_tag)
        else:
            self.pfunc("fail to insert...")
        self.pfunc(f"Insertion Probability {rn_insertion}")

        self.pfunc(f"energy_stored is {self.energy_stored:12.4f}")

        return

    def attempt_delete_atom(self, expart):
        """"""
        # pick an atom
        self.update_exlist()
        tag_idx_pick = self.pick_random_atom(expart)
        if tag_idx_pick is not None:
            self.pfunc("select species with tag %d" %tag_idx_pick)
        else:
            self.pfunc("no atom can be deleted...")
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

        content = "\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n" %(
            self.acc_volume, nexatoms, cubic_wavelength, coef
        )
        content += "Energy Change %.4f [eV]\n" %energy_change
        content += "Accept Ratio %.4f\n" %acc_ratio
        self.pfunc(content)

        rn_deletion = self.rng.uniform()
        if rn_deletion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
            # reformat exchangeable atoms
            new_expart_list = [t for t in self.tag_list[expart] if t != tag_idx_pick] 
            self.tag_list[expart] = new_expart_list
        else:
            pass

        self.pfunc("Deletion Probability %.4f" %rn_deletion)

        self.pfunc("energy_stored is %12.4f" %self.energy_stored)

        return
    
    def _pre_process(self):
        """"""
        self.pfunc("\n----- run preprocess -----\n")
        driver = self.pre_driver
        if driver is not None:
            driver.directory = self.directory/"pre"
            atoms = self.atoms
            with CustomTimer(name="preprocess", func=self.pfunc):
                self.pfunc(f"n_cur_atoms: {len(atoms)}")
                tags = atoms.get_tags()
                new_atoms = driver.run(atoms, 
                    read_exists=False, extra_info = None
                )
                new_atoms.set_tags(tags)

                #en = atoms.get_potential_energy()
            self.atoms = new_atoms
            # - save traj
            traj_frames = driver.read_trajectory(add_step_info=True)
            for a in traj_frames:
                a.info["mcstep"] = self.step_index
            write(self.MCTRAJ, traj_frames, append=True)
        else:
            self.pfunc("\nno preprocess...\n")

        return

    def optimise(self, atoms):
        """"""
        self.pfunc("\n----- DYNAMICS MIN INFO -----\n")
        driver = self.post_driver
        with CustomTimer(name="run-dynamics"):
            driver.directory = self.directory/"post"
            # run minimisation
            tags = atoms.get_tags()
            self.pfunc(f"n_cur_atoms: {len(atoms)}")
            atoms = driver.run(atoms, 
                read_exists=False, extra_info = None
            )
            atoms.set_tags(tags)

            en = atoms.get_potential_energy()

        return en, atoms


if __name__ == "__main__":
    pass