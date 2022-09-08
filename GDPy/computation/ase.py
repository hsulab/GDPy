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

def retrieve_and_save_deviation(atoms, devi_fpath):
    """"""
    results = copy.deepcopy(atoms.calc.results)
    devi_results = [(k,v) for k,v in results.items() if "devi" in k]
    if devi_results:
        devi_names = [x[0] for x in devi_results]
        devi_values = np.array([x[1] for x in devi_results]).reshape(1,-1)

        if devi_fpath.exists():
            with open(devi_fpath, "a") as fopen:
                np.savetxt(fopen, devi_values, fmt="%18.6e")
        else:
            with open(devi_fpath, "w") as fopen:
                np.savetxt(fopen, devi_values, fmt="%18.6e", header=" ".join(devi_names))

    return

def save_trajectory(atoms, log_fpath):
    """"""
    write(log_fpath, atoms, append=True)

    return


class AseDriver(AbstractDriver):

    name = "ase"

    # - defaults
    default_task = "min"
    supported_tasks = ["min", "ts", "md"]

    default_init_params = {
        "min": {
            "min_style": "bfgs",
            "min_modify": "integrator verlet tmax 4",
            "dump_period": 1
        },
        "md": {
            "md_style": "nvt",
            "velocity_seed": None,
            "timestep": 1.0, # fs
            "temp": 300, # K
            "Tdamp": 100, # fs
            "press": 1.0, # bar
            "Pdamp": 100,
            "dump_period": 1
        }
    }

    default_run_params = {
        "min": dict(
            steps= 0,
            fmax = 0.05
        ),
        "ts": {},
        "md": dict(
            steps = 0
        )
    }

    param_mapping = dict(
        temp = "temperature_K",
        Tdamp = "taut",
        pres = "pressure",
        Pdamp = "taup",
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
        super().__init__(calc, params, directory)

        self._log_fpath = self.directory / self.log_fname
        self._traj_fpath = self.directory / self.traj_fname

        return
    
    def _parse_params(self, params):
        """ init dynamics object
        """
        super()._parse_params(params)

        if self.task == "min":
            if self.init_params["min_style"] == "bfgs":
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
            raise NotImplementedError(f"{self.__class__.name} does not have {self.task} task.")
        
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
    
    def _create_dynamics(self, atoms, *args, **kwargs):
        """"""
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
            mobile_indices, frozen_indices = parse_constraint_info(
                atoms, cons_text, ignore_ase_constraints=True, ret_text=False
            )
            if frozen_indices:
                atoms.set_constraint(FixAtoms(indices=frozen_indices))
        #print(atoms.constraints)

        # - init driver
        if self.task == "min":
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
            init_params_ = copy.deepcopy(self.init_params)
            velocity_seed = init_params_.pop("velocity_seed", np.random.randint(0,10000))
            rng = np.random.default_rng(velocity_seed)

            # - velocity
            if atoms.get_kinetic_energy() > 0.:
                # atoms have momenta
                pass
            else:
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
            raise NotImplementedError(f"Unknown task {self.task}.")
        
        return driver, run_params

    def run(self, atoms_, *args, **kwargs):
        """ run the driver
            parameters of calculator will not change since
            it still performs single-point calculation
        """
        #atoms = Atoms(
        #    symbols=copy.deepcopy(atoms_.get_chemical_symbols()),
        #    positions=copy.deepcopy(atoms_.get_positions()),
        #    cell=copy.deepcopy(atoms_.get_cell(complete=True)),
        #    pbc=copy.deepcopy(atoms_.get_pbc())
        #)
        atoms = copy.deepcopy(atoms_)
        dynamics, run_params = self._create_dynamics(atoms, *args, **kwargs)

        # NOTE: traj file not stores properties (energy, forces) properly
        dynamics.attach(
            save_trajectory, interval=self.init_params["loginterval"],
            atoms=atoms, log_fpath=self.directory/"traj.xyz"
        )
        # NOTE: retrieve deviation info
        dynamics.attach(
            retrieve_and_save_deviation, interval=self.init_params["loginterval"], 
            atoms=atoms, devi_fpath=self.directory/"model_devi-ase.dat"
        )
        dynamics.run(**run_params)

        return atoms
    
    def read_trajectory(self, add_step_info=True, *args, **kwargs):
        """ read trajectory in the current working directory
        """
        traj_frames = read(self.directory/"traj.xyz", index=":")

        if add_step_info:
            data = np.loadtxt(self.directory/"dyn.log", dtype=float, skiprows=1)
            timesteps = data[:, 0] # ps
            steps = timesteps*1000/self.init_params["timestep"]
            for step, atoms in zip(steps, traj_frames):
                atoms.info["step"] = int(step)

        # - read deviation, similar to lammps
        devi_fpath = self.directory / "model_devi-ase.dat"
        if devi_fpath.exists():
            with open(devi_fpath, "r") as fopen:
                lines = fopen.readlines()
            dkeys = ("".join([x for x in lines[0] if x != "#"])).strip().split()
            dkeys = [x.strip() for x in dkeys][1:]
            data = np.loadtxt(devi_fpath, dtype=float)
            ncols = data.shape[-1]
            data = data.reshape(-1,ncols)
            data = data.transpose()[1:,:len(traj_frames)]

            for i, atoms in enumerate(traj_frames):
                for j, k in enumerate(dkeys):
                    atoms.info[k] = data[j,i]

        return traj_frames


if __name__ == "__main__":
    pass