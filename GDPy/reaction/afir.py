#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import logging
import pathlib
import time

from typing import NoReturn, List, Mapping

from itertools import combinations, product, chain, groupby

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.formula import Formula

from GDPy.builder import create_generator
from GDPy.builder.group import create_a_group, create_a_molecule_group

from GDPy.computation.utils import make_clean_atoms

from GDPy.graph.creator import find_product, find_molecules
from GDPy.reaction.utils import convert_index_to_formula


def find_target_fragments(atoms, target_commands: List[str]) -> Mapping[str,List[List[int]]]:
    """Find target fragments in the structure to react.

    This is a wrapper for group commands as there are several ways to defind
    a (atomic) group but we need a group of molecules here. Optional ways are 
    using `molecule` all, using `tag`...

    """
    fragments = {} # Mapping[str,List[List[int]]]
    target_molecules = None

    ngroups = len(target_commands)
    if ngroups == 1:
        atomic_indices = create_a_group(atoms, target_commands[0])
        fragments = find_molecules(atoms, atomic_indices)
    else:
        assert ngroups >= 2, "Need at least 2 groups..."
        for group_command in target_commands:
            fragments[group_command] = create_a_molecule_group(atoms, group_command)

    return fragments


class AFIRSearch():

    _directory = None
    _restart = False


    def __init__(
            self, target, mechanism: str="bi", gamma: List[float]=[0.5,2.5,1.0], generator=None,
            min_is=True, find_mep=False,
            seed=None, directory="./"
        ) -> NoReturn:
        """Define some basic parameters for the afir search.

        Args:
            min_is: Whether to minimise the initial structure.
                    Otherwise, the single-point energy is calculated.
        
        """
        self.target = target
        #assert len(self.target) == 2, "Target only supports two elements."

        self.mechanism = mechanism # default should be bimolecular reaction

        self.gamma = gamma
        
        self.generator = generator

        self.min_is = min_is
        self.find_mep = find_mep

        # - assign task dir
        self.directory = directory
        self._init_logger(self.directory)

        # - assign random seeds
        if seed is None:
            # TODO: need a random number to be logged
            self.rng = np.random.default_rng()
        elif isinstance(seed, int):
            self.rng = np.random.default_rng(seed)

        return

    @property
    def directory(self):

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        """"""
        # - create main dir
        directory_ = pathlib.Path(directory_)
        if not directory_.exists():
            directory_.mkdir() # NOTE: ./tmp_folder
        else:
            pass
        self._directory = directory_

        return

    def _init_logger(self, working_directory):
        """"""
        self.logger = logging.getLogger(__name__)

        log_level = logging.INFO

        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        self.logger.handlers = []
        # - screen
        # NOTE: jax's logger overwrite the setting of screen handler
        #if not self.logger.hasHandlers():
        #    ch = logging.StreamHandler()
        #    ch.setLevel(log_level)
        #    #ch.setFormatter(formatter)
        #    self.logger.addHandler(ch)
        #ch = logging.StreamHandler()
        #ch.setLevel(log_level)
        #ch.setFormatter(formatter)
        #self.logger.addHandler(ch)

        # - file
        working_directory = self.directory
        log_fpath = working_directory / (self.__class__.__name__+".out")

        if self._restart:
            fh = logging.FileHandler(filename=log_fpath, mode="a")
        else:
            fh = logging.FileHandler(filename=log_fpath, mode="w")
        fh.setLevel(log_level)
        #fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        return
    
    def run(self, worker, atoms_=None) -> NoReturn:
        """"""
        self.logger.info("START AFIR SEARCH")

        if self.generator is not None:
            # - get initial structures from the generator
            gen = create_generator(self.generator)
            frames = gen.run()

            assert len(frames) == 1, "Only one structure for now."
            atoms = frames[0]
        else:
            atoms = atoms_
        
        if atoms is None:
            raise RuntimeError(f"{self.__class__.__name__} needs atoms.")
        
        # - minimise input structure with target potential
        if self.min_is:
            self.logger.info("Minimise the initial state...")
            atoms = self._minimise_stable_state(atoms, worker, directory_=self.directory/"IS", steps=0)
        
        # - TODO: check molecules in the initial state?

        # - find possible reaction pairs
        fragments = find_target_fragments(atoms, self.target)
        # TODO: assert molecules in one group are the same type?
        content = "Found Target Fragments: \n"
        for k, v in fragments.items():
            content += "  {:<24s}:  {}\n".format(k, v)
        self.logger.info(content)

        frag_list = []
        for k, v in fragments.items():
            frag_list.append(v)
        possible_pairs = list(product(*frag_list))

        # - prepare afir bias
        # TODO: retain bias in the driver?
        bias_params = dict(
            method = "afir",
            gamma = None,
            groups = None
        )

        with open(self.directory/"rxn.dat", "w") as fopen:
            fopen.write(
                ("{:<8s}  "*2+"{:<12s}  "*6+"{:<12s}  "+"\n").format(
                    "#Pair", "nframes", "ene_is", "frc_is",
                    "ene_ts", "frc_ts", "ene_fs", "frc_fs",
                    "is_reacted"
                )
            )

        # - run each pair
        for i, pair in enumerate(possible_pairs):
            # -- start info
            self.logger.info(f"===== Pair {i} =====")
            bias_params["groups"] = pair 
            reax_indices = bias_params["groups"]
            reactants = convert_index_to_formula(atoms, reax_indices)
            self.logger.info("Reactants:")
            self.logger.info(reactants)
            self.logger.info(pair)

            # -- run pair
            directory_ = self.directory / f"p{i}"
            self._irun(atoms, worker, reactants, bias_params, directory_)

            # -- end info
            self.logger.info("\n\n")

        return
    
    def _irun(self, atoms_: Atoms, worker, reactants: List[str], bias_params_: dict, directory_):
        """Run a specific reaction pair in the structure."""
        ginit, gmax, gintv = self.gamma # eV
        # TODO: if ginit is none, then random one
        if ginit is None:
            cur_gamma = gmax*self.rng.random()
        else:
            cur_gamma = ginit
        self.logger.info(f"Initial Gamma: {cur_gamma}")

        driver = worker.driver

        reactants = sorted(reactants)

        # - run biased minimisation
        path_frames = [atoms_] # add initial state
        is_reacted = False

        ngamma = 0
        while cur_gamma <= gmax:
            self.logger.info(f"\nCurrent Gamma: {cur_gamma}")
            # TODO: make this part parallel, which maybe efficient for
            #       ab initio calculations...
            cur_atoms = atoms_.copy()
            # -- update worker's bias
            bias_params = copy.deepcopy(bias_params_)
            bias_params["gamma"] = cur_gamma
            #print(bias_params)
            driver.bias = driver._parse_bias([bias_params])
            #print(driver.bias)
            #print(driver.bias[0].gamma)

            # -- run calculation
            driver.directory = directory_ / f"g{ngamma}"
            end_atoms_ = driver.run(cur_atoms) # end frame of the minimisation

            results = dict(
                energy = end_atoms_.get_potential_energy(),
                forces = end_atoms_.get_forces(apply_constraint=False)
            )
            end_atoms = make_clean_atoms(end_atoms_, results)

            path_frames.append(end_atoms)

            # -- check if reaction happens?
            #    if so, break while
            prod_indices = find_product(end_atoms, bias_params["groups"])
            products = convert_index_to_formula(end_atoms, prod_indices)
            products = sorted(products)
            self.logger.info("Molecules: {}".format(" ".join(products)))
            if products == reactants:
                # TODO: index may change? molecule reconstruct?
                self.logger.info("nothing happens...")
                ...
            else:
                # TODO: resacale gamma if reaction were found in the second minimisation
                is_reacted = True
                self.logger.info("reaction happens...")
                self.logger.info(
                    "{} -> {}".format(" + ".join(reactants), " + ".join(products))
                )
                self.logger.info("minimise the last structure to the final state...")
                fs_atoms = self._minimise_stable_state(end_atoms, worker, directory_=directory_/"fs")
                path_frames.append(fs_atoms)
                break

            # -- update gamma
            cur_gamma += gintv
            ngamma += 1
        
        # - sort results
        write(directory_/"pseudo_path.xyz", path_frames)

        nframes = len(path_frames)
        energies = [a.get_potential_energy() for a in path_frames]
        ene_ts = max(energies)
        ts_idx = energies.index(ene_ts)

        maxforces = [np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in path_frames]
        frc_ts = maxforces[ts_idx]

        # info: nframes, IS, TS, FS, is_reacted?
        with open(self.directory/"rxn.dat", "a") as fopen:
            fopen.write(
                ("{:<8s}  "+"{:<8d}  "+"{:<12.4f}  "*6+"{:<12s}"+"\n").format(
                    directory_.name, nframes, energies[0], maxforces[0],
                    ene_ts, frc_ts, energies[-1], maxforces[-1], str(is_reacted)
                )
            )

        return
    
    def _minimise_stable_state(self, atoms_, worker, directory_, **kwargs) -> Atoms:
        """Re-minimise stable states (IS and FS).

        Initial state (input structure) may be minimised at an another theory 
        level or even not minimised at all.

        """
        # - prepare driver, remove bias
        driver = worker.driver
        driver._parse_bias([])
        driver.directory = directory_

        cur_atoms = atoms_.copy()
        end_atoms_ = driver.run(cur_atoms, **kwargs) # end frame of the minimisation

        results = dict(
            energy = end_atoms_.get_potential_energy(),
            forces = end_atoms_.get_forces(apply_constraint=False)
        )
        end_atoms = make_clean_atoms(end_atoms_, results)

        return end_atoms


if __name__ == "__main__":
    ...