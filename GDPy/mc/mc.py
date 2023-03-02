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
        print(self.atoms)

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
        # - prepare workers...
        self._init_logger()
        #     TODO: broadcast scheduler to different drivers?
        worker.directory = self.directory
        worker.logger = self.logger

        drivers = {}
        driver_names = ["init", "post"]
        for name in driver_names:
            params = self.param_drivers.get(name, worker.driver.as_dict())
            driver = worker.potter.create_driver(params)
            drivers[name] = driver

        # - add print function to operators
        for op in self.operators:
            op.pfunc = self.pfunc

        # - run init
        worker.driver = drivers["init"]
        self.atoms.info["wdir"] = "cand0"
        _ = worker.run([self.atoms])
        cur_frames = worker.retrieve()
        cur_atoms = cur_frames[0]
        energy_stored = cur_atoms.get_potential_energy()
        self.pfunc(f"ene: {energy_stored}")
        self.atoms = cur_atoms
        write(self.directory/"mc.xyz", self.atoms)

        # -- log operator status
        with open(self.directory/"opstat.txt", "w") as fopen:
            fopen.write(
                "{:<24s}  {:<12s}  {:<12s}  {:<12s}  \n".format(
                    "#Operator", "Success", "prev_ene", "curr_ene"
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
                cur_atoms.info["wdir"] = f"cand{i}"
                _ = worker.run([cur_atoms])
                cur_frames = worker.retrieve()
                cur_atoms = cur_frames[0]
                energy_operated = cur_atoms.get_potential_energy()
                self.pfunc(f"post ene: {energy_operated}")
                ...
                # -- metropolis
                success = op.metropolis(energy_stored, energy_operated, self.rng)
            else:
                success = False
            with open(self.directory/"opstat.txt", "a") as fopen:
                fopen.write(
                    "{:<24s}  {:<12s}  {:<12.4f}  {:<12.4f}  \n".format(
                        op_name, str(success), energy_stored, energy_operated
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