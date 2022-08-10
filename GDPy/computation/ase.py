#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import time
import json
import shutil
import pathlib
from pathlib import Path
import warnings

import numpy as np

from ase import Atoms
from ase import units

from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from GDPy.computation.driver import AbstractDriver

from GDPy.md.md_utils import force_temperature
from GDPy.md.nosehoover import NoseHoover

from GDPy.builder.constraints import parse_constraint_info


""" TODO:
        add uncertainty quatification
"""


class AseDriver(AbstractDriver):

    # - defaults
    default_task = "bfgs"
    supported_tasks = ["bfgs", "ts", "nvt"]

    default_init_params = {
        "nvt": dict(
            timestep = 1.0, # fs
            temperature = 300, # Kelvin
            nvt_q = 334.,
            loginterval=1
        )
    }

    default_run_params = {
        "bfgs": dict(
            steps= 200,
            fmax = 0.05
        ),
        "ts": {},
        "nvt": dict(
            steps = 10
        )
    }

    # - other files
    log_fname = "dyn.log"
    traj_fname = "dyn.traj"

    saved_cards = [traj_fname]

    def __init__(
        self, calc=None, params: dict={}, directory="./"
    ):
        """"""
        self.calc = calc
        self.calc.reset()

        self._log_fpath = self.directory / self.log_fname
        self._traj_fpath = self.directory / self.traj_fname

        self.directory = directory

        self._parse_params(params)

        return
    
    def _parse_params(self, params):
        """ init dynamics object
        """
        super()._parse_params(params)

        if self.task == "bfgs":
            from ase.optimize import BFGS
            driver_cls = BFGS
        elif self.task == "ts":
            from sella import Sella, Constraints
            driver_cls = Sella
        elif self.task == "nvt":
            driver_cls = NoseHoover
        else:
            pass
        
        self.driver_cls = driver_cls

        return
    
    def update_params(self, **kwargs):

        return
    
    @property
    def log_fpath(self):

        return self._log_fpath
    
    @property
    def traj_fpath(self):

        return self._traj_fpath
    
    @AbstractDriver.directory.setter
    def directory(self, directory_):
        """"""
        # - main and calc
        super(AseDriver, AseDriver).directory.__set__(self, directory_)

        # - other files
        self._log_fpath = self.directory / self.log_fname
        self._traj_fpath = self.directory / self.traj_fname

        return 
    
    def run(self, atoms, **kwargs):
        """ run the driver
            parameters of calculator will not change since
            it still performs single-point calculation
        """
        # - set special keywords
        atoms.calc = self.calc

        # - prepare dir
        if not self.directory.exists():
            self.directory.mkdir(parents=True)
        
        # - overwrite 
        run_params = self.run_params.copy()
        #print(run_params)
        run_params.update(**kwargs)
        #print(run_params)

        # TODO: if have cons in kwargs overwrite current cons stored in atoms
        cons_text = run_params.pop("constraint", None)

        if cons_text is not None:
            atoms._del_constraints()
            mobile_indices, frozen_indices = parse_constraint_info(atoms, cons_text, ret_text=False)
            if frozen_indices:
                atoms.set_constraint(FixAtoms(indices=frozen_indices))
        #print(atoms.constraints)

        # - init driver
        if self.task == "bfgs":
            driver = self.driver_cls(
                atoms, 
                logfile=self.log_fpath,
                trajectory=str(self.traj_fpath)
            )
        elif self.task == "ts":
            driver = self.driver(
                atoms,
                order = 1,
                internal = False,
                logfile=self.log_fpath,
                trajectory=str(self.traj_fpath)
            )
        elif self.task == "nvt":
            # - velocity
            MaxwellBoltzmannDistribution(atoms, self.init_params["temperature"]*units.kB)
            force_temperature(atoms, self.init_params["temperature"])

            # - construct the driver
            driver = self.driver_cls(
                atoms = atoms,
                timestep = self.init_params["timestep"] * units.fs,
                temperature = self.init_params["temperature"] * units.kB,
                nvt_q = self.init_params["nvt_q"],
                logfile=self.log_fpath,
                trajectory=str(self.traj_fpath)
            )
        else:
            pass

        driver.run(**run_params)

        return atoms
    
    def minimise(self, atoms, repeat=1, extra_info=None, verbose=True, **kwargs) -> Atoms:
        """ return a new atoms with singlepoint calc
            input atoms wont be changed
        """
        # run dynamics
        cur_params = self.dyn_runparams.copy()
        for k, v in kwargs:
            if k in cur_params:
                cur_params[k] = v
        fmax = cur_params["fmax"]

        # TODO: add verbose
        content = f"\n=== Start minimisation maximum try {repeat} times ===\n"
        for i in range(repeat):
            content += f"--- attempt {i} ---\n"
            min_atoms = self.run(atoms, **cur_params)
            min_results = self.__read_min_results(self._logfile_path)
            content += min_results
            # NOTE: add few information
            # if extra_info is not None:
            #     min_atoms.info.update(extra_info)
            maxforce = np.max(np.fabs(min_atoms.get_forces(apply_constraint=True)))
            if maxforce <= fmax:
                break
            else:
                atoms = min_atoms
                print("backup old data...")
                for card in self.saved_cards:
                    card_path = self._directory_path / card
                    bak_fmt = ("bak.{:d}."+card)
                    idx = 0
                    while True:
                        bak_card = bak_fmt.format(idx)
                        if not Path(bak_card).exists():
                            saved_card_path = self._directory_path / bak_card
                            shutil.copy(card_path, saved_card_path)
                            break
                        else:
                            idx += 1
        else:
            warnings.warn(f"Not converged after {repeat} minimisations, and save the last atoms...", UserWarning)
        
        if verbose:
            print(content)

        return min_atoms

    def __read_min_results(self, fpath):
        """ compatibilty to lammps
        """
        with open(fpath, "r") as fopen:
            min_results = fopen.read()

        return min_results


if __name__ == '__main__':
    pass