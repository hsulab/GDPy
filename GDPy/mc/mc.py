#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import pathlib
from typing import NoReturn, List

import logging

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.computation.worker.drive import DriverBasedWorker
from GDPy.builder import create_generator
from GDPy.mc.operators import select_operator, parse_operators

"""This module tries to offer a base class for all MonteCarlo-like methods.
"""


class MonteCarlo():

    restart = False

    pfunc = print

    def __init__(
        self, system: dict, operators: List[dict], drivers: dict, 
        random_seed=1112, overwrite: bool=True, restart: bool=False, directory="./", 
        *args, **kwargs
    ) -> NoReturn:
        """Parameters for Monte Carlo.

        Args:
            overwrite: Whether overwrite calculation directory.
        
        """
        self.directory = directory

        self.overwrite = overwrite
        self.restart = restart

        # - parse system
        generator = create_generator(system)
        frames = generator.run()
        assert len(frames) == 1, "MC only accepts one structure."
        self.atoms = frames[0]


        # - parse operators
        self.operators, self.op_probs = parse_operators(operators)

        # - add parameters of drivers
        self.param_drivers = drivers

        # - set random seed that 
        self.set_rng(seed=random_seed)

        return

    @property
    def directory(self):

        return self._directory

    @directory.setter
    def directory(self, directory_):
        # - create main dir
        directory_ = pathlib.Path(directory_)
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
    
    def run(self, worker: DriverBasedWorker, nattempts: int=1):
        """Run MonteCarlo simulation."""
        # - prepare logger and output some basic info...
        self._init_logger()

        self.pfunc("===== MonteCarlo Operators (Modifiers) =====\n")
        for op in self.operators:
            self.pfunc(str(op))
        self.pfunc(f"normalised probabilities {self.op_probs}\n")

        # -- add print function to operators
        for op in self.operators:
            op.pfunc = self.pfunc

        # -- TODO: check if operators' regions are consistent
        #          though it works, unexpected results may occur
        
        # - prepare atoms
        self.pfunc("===== MonteCarlo Structure =====\n")
        tags = self.atoms.arrays.get("tags", None)
        if tags is None:
            # default is setting tags by elements
            symbols = self.atoms.get_chemical_symbols()
            type_list = sorted(list(set(symbols)))
            new_tags = [type_list.index(s)*10000+i for i, s in enumerate(symbols)]
            self.atoms.set_tags(new_tags)
            self.pfunc("set default tags by symbols...")
        else:
            self.pfunc("set attached tags from the structure...")

        # - prepare drivers
        #     TODO: broadcast scheduler to different drivers?
        worker.directory = self.directory
        worker.logger = self.logger

        drivers = {}
        driver_names = ["init", "post"]
        for name in driver_names:
            params = self.param_drivers.get(name, worker.driver.as_dict())
            driver = worker.potter.create_driver(params)
            drivers[name] = driver

        # - run init
        self.pfunc("===== MonteCarlo Initial Minimisation =====\n")
        # NOTE: atoms lost tags in optimisation
        #       TODO: move this part to driver?
        cur_tags = self.atoms.get_tags()

        self.atoms.info["confid"] = 0
        self.atoms.info["step"] = -1 # NOTE: remove step info

        worker.driver = drivers["init"]
        _ = worker.run([self.atoms])
        cur_frames = worker.retrieve()

        cur_atoms = cur_frames[0]
        energy_stored = cur_atoms.get_potential_energy()
        self.pfunc(f"ene: {energy_stored}")
        self.atoms = cur_atoms
        self.atoms.set_tags(cur_tags)
        write(self.directory/"mc.xyz", self.atoms)

        # -- log operator status
        with open(self.directory/"opstat.txt", "w") as fopen:
            fopen.write(
                "{:<24s}  {:<12s}  {:<12s}  {:<12s}  {:<12s}  \n".format(
                    "#Operator", "natoms", "Success", "prev_ene", "curr_ene"
                )
            )
        
        # - run mc steps
        # -- switch to post driver
        worker.driver = drivers["post"]
        # -- 
        for i in range(1,nattempts+1):
            self.pfunc(f"===== MC Step {i} =====")
            # -- operate atoms
            #    always return atoms even if no change is applied
            op = select_operator(self.operators, self.op_probs, self.rng)
            op_name = op.__class__.__name__
            self.pfunc(f"operator {op_name}")
            cur_atoms = op.run(self.atoms, self.rng)
            # -- postprocess?
            energy_operated = energy_stored
            if cur_atoms is not None:
                cur_tags = cur_atoms.get_tags()
                cur_atoms.info["confid"] = int(f"{i}")
                cur_atoms.info["step"] = -1 # NOTE: remove step info
                _ = worker.run([cur_atoms])
                cur_frames = worker.retrieve()
                cur_atoms = cur_frames[0]
                cur_atoms.set_tags(cur_tags)
                energy_operated = cur_atoms.get_potential_energy()
                self.pfunc(f"post ene: {energy_operated}")
                ...
                # -- metropolis
                success = op.metropolis(energy_stored, energy_operated, self.rng)
            else:
                success = False
            with open(self.directory/"opstat.txt", "a") as fopen:
                fopen.write(
                    "{:<24s}  {:<12d}  {:<12s}  {:<12.4f}  {:<12.4f}  \n".format(
                        op_name, len(self.atoms), str(success), energy_stored, energy_operated
                    )
                )
            # -- update atoms
            if success:
                energy_stored = energy_operated
                self.atoms = cur_atoms
                self.pfunc("success...")
            else:
                self.pfunc("failure...")
            write(self.directory/"mc.xyz", self.atoms, append=True)

            # -- remove
            if self.overwrite: # save current calculation...
                if (self.directory/f"cand{i-1}").exists() and i >= 2:
                    shutil.rmtree((self.directory/f"cand{i-1}"))

        return


if __name__ == "__main__":
    ...
