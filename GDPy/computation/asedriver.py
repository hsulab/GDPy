#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import dataclasses
import shutil
from pathlib import Path
from typing import NoReturn, List, Tuple

import numpy as np

from ase import Atoms
from ase import units

from ase.io import read, write
import ase.constraints
from ase.constraints import Filter, FixAtoms
from ase.optimize.optimize import Dynamics
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution, Stationary, ZeroRotation
)

from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.computation.driver import AbstractDriver, DriverSetting
from GDPy.computation.bias import create_bias_list

from GDPy.md.md_utils import force_temperature

from GDPy.builder.constraints import parse_constraint_info


def retrieve_and_save_deviation(atoms, devi_fpath) -> NoReturn:
    """Read model deviation and add results to atoms.info if the file exists."""
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
                np.savetxt(
                    fopen, devi_values, fmt="%18.6e", header=("{:<18s}"*len(devi_names)).format(*devi_names)
                )

    return

def save_trajectory(atoms, log_fpath) -> NoReturn:
    """Create a clean atoms from the input and save simulation trajectory."""
    atoms_ = Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions().copy(),
        cell=atoms.get_cell().copy(),
        pbc=copy.deepcopy(atoms.get_pbc())
    )

    results = dict(
        energy = atoms.get_potential_energy(),
        forces = copy.deepcopy(atoms.get_forces())
    )
    spc = SinglePointCalculator(atoms, **results)
    atoms_.calc = spc

    write(log_fpath, atoms_, append=True)

    return

@dataclasses.dataclass
class AseDriverSetting(DriverSetting):

    driver_cls: Dynamics = None
    filter_cls: Filter = None

    fmax: float = 0.05 # eV/Ang

    def __post_init__(self):
        """"""
        # - task-specific params
        if self.task == "md":
            self._internals.update(
                velocity_seed = self.velocity_seed,
                md_style = self.md_style,
                timestep = self.timestep,
                temperature_K = self.temp,
                taut = self.Tdamp,
                pressure = self.press,
                taup = self.Pdamp,
            )
            # TODO: provide a unified class for thermostat
            if self.md_style == "nve":
                from ase.md.verlet import VelocityVerlet as driver_cls
            elif self.md_style == "nvt":
                #from GDPy.md.nosehoover import NoseHoover as driver_cls
                from ase.md.nvtberendsen import NVTBerendsen as driver_cls
            elif self.md_style == "npt":
                from ase.md.nptberendsen import NPTBerendsen as driver_cls
        
        if self.task == "min":
           # - to opt atomic positions
            from ase.optimize import BFGS
            if self.min_style == "bfgs":
                driver_cls = BFGS
            # - to opt unit cell
            #   UnitCellFilter, StrainFilter, ExpCellFilter
            # TODO: add filter params
            filter_names = ["unitCellFilter", "StrainFilter", "ExpCellFilter"]
            if self.min_style in filter_names:
                driver_cls = BFGS
                self.filter_cls = getattr(ase.constraints, self.min_style)

        if self.task == "ts":
            # TODO: move to reactor
            try:
                from sella import Sella, Constraints
                driver_cls = Sella
            except:
                raise NotImplementedError(f"Sella is not installed.")
            ...
        
        try:
            self.driver_cls = driver_cls
        except:
            raise RuntimeError("Ase Driver Class is not defined.")

        # - shared params
        self._internals.update(
            loginterval = self.dump_period
        )

        # NOTE: There is a bug in ASE as it checks `if steps` then fails when spc.
        if self.steps == 0:
            self.steps = -1

        return
    
    def get_run_params(self, *args, **kwargs) -> dict:
        """"""
        run_params = dict(
            steps = kwargs.get("steps", self.steps),
            constraint = kwargs.get("constraint", None)
        )
        if self.task == "min" or self.task == "ts":
            run_params.update(
                fmax = kwargs.get("fmax", self.fmax),
            )
        run_params.update(**kwargs)

        return run_params


class AseDriver(AbstractDriver):

    name = "ase"

    # - defaults
    default_task = "min"
    supported_tasks = ["min", "ts", "md"]

    # - other files
    log_fname = "dyn.log"
    traj_fname = "dyn.traj"

    #: List of output files would be saved when restart.
    saved_cards: List[str] = [traj_fname]

    def __init__(
        self, calc=None, params: dict={}, directory="./", *args, **kwargs
    ):
        """"""
        super().__init__(calc, params, directory, *args, **kwargs)

        self.setting = AseDriverSetting(**params)

        self._log_fpath = self.directory / self.log_fname
        self._traj_fpath = self.directory / self.traj_fname

        return
    
    @property
    def log_fpath(self):
        """File path of the simulation log."""

        return self._log_fpath
    
    @property
    def traj_fpath(self):
        """File path of the simulation trajectory."""

        return self._traj_fpath
    
    @AbstractDriver.directory.setter
    def directory(self, directory_):
        """Set log and traj path regarding to the working directory."""
        # - main and calc
        super(AseDriver, AseDriver).directory.__set__(self, directory_)

        # - other files
        self._log_fpath = self.directory / self.log_fname
        self._traj_fpath = self.directory / self.traj_fname

        return 
    
    def _create_dynamics(self, atoms, *args, **kwargs) -> Tuple[Dynamics,dict]:
        """Create the correct class of this simulation with running parameters."""
        # - set special keywords
        atoms.calc = self.calc

        # - prepare dir
        if not self.directory.exists():
            self.directory.mkdir(parents=True)
        
        # - overwrite 
        run_params = self.setting.get_run_params(*args, **kwargs)

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
        if self.setting.task == "min":
            if self.setting.filter_cls:
                atoms = self.setting.filter_cls(atoms)
            driver = self.setting.driver_cls(
                atoms, 
                logfile=self.log_fpath,
                trajectory=str(self.traj_fpath)
            )
        elif self.setting.task == "ts":
            driver = self.setting.driver_cls(
                atoms,
                order = 1,
                internal = False,
                logfile=self.log_fpath,
                trajectory=str(self.traj_fpath)
            )
        elif self.setting.task == "md":
            # - adjust params
            init_params_ = copy.deepcopy(self.setting.get_init_params())
            velocity_seed = init_params_.pop("velocity_seed", np.random.randint(0,10000))
            rng = np.random.default_rng(velocity_seed)

            # - velocity
            if atoms.get_kinetic_energy() > 0.:
                # atoms have momenta
                ...
            else:
                MaxwellBoltzmannDistribution(
                    atoms, temperature_K=init_params_["temperature_K"], rng=rng
                )
                # TODO: make this optional
                ZeroRotation(atoms, preserve_temperature=True)
                Stationary(atoms, preserve_temperature=True)
                force_temperature(atoms, init_params_["temperature_K"], unit="K") # NOTE: respect constraints

            # - prepare args
            # TODO: move this part to setting post_init?
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

            init_params_["timestep"] *= units.fs
            #print(init_params_)

            # - construct the driver
            driver = self.setting.driver_cls(
                atoms = atoms,
                **init_params_,
                logfile=self.log_fpath,
                trajectory=str(self.traj_fpath)
            )
        else:
            raise NotImplementedError(f"Unknown task {self.task}.")
        
        return driver, run_params

    def run(self, atoms_, read_exists: bool=True, extra_info: dict=None, *args, **kwargs):
        """Run the driver.

        Additional output files would be generated, namely a xyz-trajectory and
        a deviation file if the calculator could estimate uncertainty.

        Note:
            Calculator's parameters will not change since it still performs 
            single-point calculations as the simulation goes.

        """
        atoms = copy.deepcopy(atoms_) # TODO: make minimal atoms object?
        dynamics, run_params = self._create_dynamics(atoms, *args, **kwargs)
        print("run_params: ", run_params)

        # NOTE: traj file not stores properties (energy, forces) properly
        init_params = self.setting.get_init_params()
        dynamics.attach(
            save_trajectory, interval=init_params["loginterval"],
            atoms=atoms, log_fpath=self.directory/"traj.xyz"
        )
        # NOTE: retrieve deviation info
        dynamics.attach(
            retrieve_and_save_deviation, interval=init_params["loginterval"], 
            atoms=atoms, devi_fpath=self.directory/"model_devi-ase.dat"
        )
        dynamics.run(**run_params)

        return atoms
    
    def read_trajectory(self, add_step_info=True, *args, **kwargs):
        """Read trajectory in the current working directory."""
        traj_frames = read(self.directory/"traj.xyz", index=":")

        # TODO: log file will not be overwritten when restart
        init_params = self.setting.get_init_params()
        if add_step_info:
            if self.setting.task == "md":
                data = np.loadtxt(self.directory/"dyn.log", dtype=float, skiprows=1)
                if len(data.shape) == 1:
                    data = data[np.newaxis,:]
                timesteps = data[:, 0] # ps
                steps = timesteps*1000/init_params["timestep"]
            elif self.setting.task == "min":
                data = np.loadtxt(self.directory/"dyn.log", dtype=str, skiprows=1)
                if len(data.shape) == 1:
                    data = data[np.newaxis,:]
                steps = [int(s) for s in data[:, 1]]
            assert len(steps) == len(traj_frames), "Number of steps and number of frames are inconsistent..."
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


class BiasedAseDriver(AseDriver):

    """Run dynamics with external forces (bias).
    """

    def __init__(
        self, calc=None, params: dict={}, directory="./",
        *args, **kwargs
    ):
        """"""
        super().__init__(calc, params, directory)

        # - check bias
        self.bias = self._parse_bias(params["bias"])

        return
    
    def _parse_bias(self, params_list: List[dict]):
        """"""
        bias_list = create_bias_list(params_list)

        return bias_list

    def run(self, atoms_, *args, **kwargs):
        """Run the driver with bias."""
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

        # - set bias to atoms
        for bias in self.bias:
            bias.attach_atoms(atoms)

        # - set steps
        dynamics.max_steps = run_params["steps"]
        dynamics.fmax = run_params["fmax"]

        # - mimic the behaviour of ase dynamics irun
        natoms = len(atoms)

        for _ in self._irun(atoms, dynamics):
            ...

        return atoms
    
    def _irun(self, atoms, dynamics):
        """Mimic the behaviour of ase dynamics irun."""
        # -- compute original forces
        cur_forces = atoms.get_forces(apply_constraint=True).copy()
        ext_forces = np.zeros((len(atoms),3)) + np.inf
        cur_forces += ext_forces # avoid convergence at 0 step

        yield False

        if dynamics.nsteps == 0: # log first step
            dynamics.log(forces=cur_forces)
            dynamics.call_observers()

        while (
            not dynamics.converged(forces=cur_forces) and 
            dynamics.nsteps < dynamics.max_steps
        ):
            # -- compute original forces
            cur_forces = atoms.get_forces(apply_constraint=True).copy()

            # -- compute external forces
            ext_forces = self._compute_external_forces(atoms)
            #print(ext_forces)
            cur_forces += ext_forces

            # -- run step
            dynamics.step(f=cur_forces)
            dynamics.nsteps += 1

            yield False

            # log the step
            dynamics.log(forces=cur_forces)
            dynamics.call_observers()
        
        yield dynamics.converged(forces=cur_forces)
    
    def _compute_external_forces(self, atoms):
        """Compute external forces based on bias params."""
        # TODO: replace this with a bias mixer
        natoms = len(atoms)
        ext_forces = np.zeros((natoms,3))
        for cur_bias in self.bias:
            ext_forces += cur_bias.compute()

        return ext_forces


if __name__ == "__main__":
    pass