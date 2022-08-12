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

from ase.io import read, write
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from GDPy.computation.driver import AbstractDriver

from GDPy.md.md_utils import force_temperature

from GDPy.builder.constraints import parse_constraint_info


""" TODO:
        add uncertainty quatification
"""


class AseDriver(AbstractDriver):

    # - defaults
    default_task = "bfgs"
    supported_tasks = ["bfgs", "ts", "md"]

    default_init_params = {
        # TODO: make md params consistent
        "md": {
            "md_style": "nvt",
            "velocity_seed": 1112,
            "timestep": 1.0, # fs
            "temperature_K": 300, # K
            "taut": 100, # fs
            "pressure": 1.0, # bar
            "taup": 100,
            "loginterval": 1
        }
    }

    default_run_params = {
        "bfgs": dict(
            steps= 200,
            fmax = 0.05
        ),
        "ts": {},
        "md": dict(
            steps = 10
        )
    }

    param_mapping = dict(
        temp = "temperature_K",
        taut = "Tdamp",
        pres = "pressure",
        taup = "Pdamp",
        dump_period = "loginterval"
    )

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
        elif self.task == "md":
            if self.init_params["md_style"] == "nve":
                from ase.md.verlet import VelocityVerlet as driver_cls
            elif self.init_params["md_style"] == "nvt":
                #from GDPy.md.nosehoover import NoseHoover as driver_cls
                from ase.md.nvtberendsen import NVTBerendsen as driver_cls
            elif self.init_params["md_style"] == "npt":
                from ase.md.nptberendsen import NPTBerendsen as driver_cls
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
        kwargs = self._map_params(kwargs)
        run_params = self.run_params.copy()
        run_params.update(**kwargs)

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
        elif self.task == "md":
            # - adjust params
            init_params_ = self.init_params.copy()
            velocity_seed = init_params_.pop("velocity_seed")
            rng = np.random.default_rng(velocity_seed)

            # - velocity
            MaxwellBoltzmannDistribution(atoms, temperature_K=init_params_["temperature_K"], rng=rng)
            force_temperature(atoms, init_params_["temperature_K"], unit="K") # NOTE: respect constraints

            # - prepare args
            md_style = init_params_.pop("md_style")
            if md_style == "nve":
                init_params_ = {k:v for k,v in init_params_.items() if k in ["loginterval", "timestep"]}
            elif md_style == "nvt":
                init_params_ = {
                    k:v for k,v in init_params_.items() 
                    if k in ["loginterval", "timestep", "temperature_K", "taut"]
                }
            elif md_style == "npt":
                init_params_ = {
                    k:v for k,v in init_params_.items() 
                    if k in ["loginterval", "timestep", "temperature_K", "taut", "pressure", "taup"]
                }
                init_params_["pressure"] *= (1./(160.21766208/0.000101325))

            # TODO: move this to parse_params?
            init_params_["timestep"] *= units.fs
            #print(init_params_)

            # - construct the driver
            driver = self.driver_cls(
                atoms = atoms,
                **init_params_,
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
    
    def read_trajectory(self, *args, **kwargs):
        """ read trajectory in the current working directory
        """
        traj_frames = read(self.traj_fpath, ":")

        return traj_frames


if __name__ == "__main__":
    pass