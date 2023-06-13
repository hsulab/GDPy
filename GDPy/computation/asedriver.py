#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import dataclasses
import shutil
from pathlib import Path
from typing import NoReturn, List, Tuple
import warnings

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
from ase.calculators.mixing import MixedCalculator

from GDPy.computation.driver import AbstractDriver, DriverSetting
from GDPy.computation.bias import create_bias_list
from GDPy.data.trajectory import Trajectory

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
    """Create a clean atoms from the input and save simulation trajectory.

    We need an explicit copy of atoms as some calculators may not return all 
    necessary information. For example, schnet only returns required properties.
    If only energy is required, there are no forces.

    """
    # - save atoms
    atoms_to_save = Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions().copy(),
        cell=atoms.get_cell().copy(),
        pbc=copy.deepcopy(atoms.get_pbc())
    )
    if "tags" in atoms.arrays:
        atoms_to_save.set_tags(atoms.get_tags())
    results = dict(
        energy = atoms.get_potential_energy(),
        forces = copy.deepcopy(atoms.get_forces())
    )
    spc = SinglePointCalculator(atoms, **results)
    atoms_to_save.calc = spc

    # - check special metadata
    calc = atoms.calc
    if isinstance(calc, MixedCalculator):
        atoms_to_save.info["energy_contributions"] = copy.deepcopy(calc.results["energy_contributions"])
        atoms_to_save.arrays["force_contributions"] = copy.deepcopy(calc.results["force_contributions"])

    # - append to traj
    write(log_fpath, atoms_to_save, append=True)

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
                ignore_atoms_velocities = self.ignore_atoms_velocities,
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

        if self.task == "rxn":
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
            constraint = kwargs.get("constraint", self.constraint)
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
    supported_tasks = ["min", "rxn", "md"]

    # - other files
    log_fname = "dyn.log"
    traj_fname = "dyn.traj"
    xyz_fname = "traj.xyz"
    devi_fname = "model_devi-ase.dat"

    #: List of output files would be saved when restart.
    saved_fnames: List[str] = [xyz_fname, devi_fname]

    #: List of output files would be removed when restart.
    removed_fnames: List[str] = [log_fname, traj_fname, xyz_fname, devi_fname]

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
            if (not init_params_["ignore_atoms_velocties"] and atoms.get_kinetic_energy() > 0.):
                # atoms have momenta
                ...
            else:
                MaxwellBoltzmannDistribution(
                    atoms, temperature_K=init_params_["temperature_K"], rng=rng
                )
                if self.setting.remove_rotation:
                    ZeroRotation(atoms, preserve_temperature=False)
                if self.setting.remove_translation:
                    Stationary(atoms, preserve_temperature=False)
                # NOTE: respect constraints
                #       ase code does not consider constraints
                force_temperature(atoms, init_params_["temperature_K"], unit="K") 

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

        # - get run_params, respect steps and fmax from kwargs
        _dynamics, run_params = self._create_dynamics(atoms, *args, **kwargs)

        converged, trajectory = self._continue(atoms, run_params, read_exists=read_exists, *args, **kwargs)
        if trajectory:
            new_atoms = trajectory[-1]
            if extra_info is not None:
                new_atoms.info.update(**extra_info)
            # TODO: set a hard limit of min steps
            #       since some terrible structures may not converged anyway
            # - check convergence of forace evaluation (e.g. SCF convergence)
            try:
                scf_convergence = self.calc.read_convergence()
            except:
                # -- cannot read scf convergence then assume it is ok
                scf_convergence = True
            if not scf_convergence:
                warnings.warn(f"{self.name} at {self.directory} failed to converge at SCF.", RuntimeWarning)
            if not converged:
                warnings.warn(f"{self.name} at {self.directory} failed to converge.", RuntimeWarning)
        else:
            raise RuntimeError(f"{self.name} at {self.directory} doesnot have a trajectory.")
        
        # - reset calc params

        return new_atoms

    def _set_dynamics(self, atoms: Atoms, *args, **kwargs):
        """"""
        # - set calculator
        atoms.calc = self.calc

        # - set dynamics
        dynamics, run_params = self._create_dynamics(atoms, *args, **kwargs)

        # NOTE: traj file not stores properties (energy, forces) properly
        init_params = self.setting.get_init_params()
        dynamics.attach(
            save_trajectory, interval=init_params["loginterval"],
            atoms=atoms, log_fpath=self.directory/self.xyz_fname
        )
        # NOTE: retrieve deviation info
        dynamics.attach(
            retrieve_and_save_deviation, interval=init_params["loginterval"], 
            atoms=atoms, devi_fpath=self.directory/self.devi_fname
        )

        return dynamics
    
    def _continue(self, atoms, run_params: dict, read_exists=True, *args, **kwargs):
        """Check whether continue unfinished calculation

        Restart driver from the last accessible atoms.

        """
        trajectory = []
        converged = False
        try:
            if read_exists:
                if (self.directory/self.xyz_fname).exists():
                    trajectory = self.read_trajectory(add_step_info=True)
                    prev_atoms = trajectory[-1]
                    converged = self.read_convergence(trajectory, run_params)
                else:
                    prev_atoms = atoms
                if not converged:
                    # backup output files and continue with lastest atoms
                    # dyn.log and dyn.traj are created when init so dont backup them
                    for fname in self.saved_fnames:
                        curr_fpath = self.directory/fname
                        if curr_fpath.exists():
                            # TODO: check if file is empty?
                            backup_fmt = ("gbak.{:d}."+fname)
                            # --- check backups
                            idx = 0
                            while True:
                                backup_fpath = self.directory/(backup_fmt.format(idx))
                                if not Path(backup_fpath).exists():
                                    shutil.copy(curr_fpath, backup_fpath)
                                    break
                                else:
                                    idx += 1
                    # remove unnecessary files and start all over
                    # retain calculator-related files
                    for fname in self.removed_fnames:
                        curr_fpath = self.directory/fname
                        if curr_fpath.exists():
                            curr_fpath.unlink()
                    # run dynamics again
                    dynamics = self._set_dynamics(prev_atoms)
                    dynamics.run(**run_params)
            else:
                # restart calculation from the scratch
                prev_atoms = atoms
                # remove unnecessary files and start all over
                # retain calculator-related files
                for fname in self.removed_fnames:
                    curr_fpath = self.directory/fname
                    if curr_fpath.exists():
                        curr_fpath.unlink()
                # run dynamics again
                dynamics = self._set_dynamics(prev_atoms)
                dynamics.run(**run_params)
            # -- check convergence again
            trajectory = self.read_trajectory(add_step_info=True)
            converged = self.read_convergence(trajectory, run_params)
        except Exception as e:
            print(f"Exception of {self.__class__.__name__} is {e}.")
        print(f"{self.name} {self.directory}")
        print("converged: ", converged)

        return converged, trajectory
    
    def read_trajectory(self, add_step_info=True, *args, **kwargs):
        """Read trajectory in the current working directory."""
        traj_frames = []
        target_fpath = self.directory/self.xyz_fname
        if target_fpath.exists() and target_fpath.stat().st_size != 0:
            # TODO: concatenate all trajectories
            traj_frames = read(self.directory/self.xyz_fname, index=":")

            # - check the convergence of the force evaluation
            try:
                scf_convergence = self.calc.read_convergence()
            except:
                # -- cannot read scf convergence then assume it is ok
                scf_convergence = True
            if not scf_convergence:
                warnings.warn(f"{self.name} at {self.directory} failed to converge at SCF.", RuntimeWarning)
                traj_frames[0].info["error"] = f"Unconverged SCF at {self.directory}."

            # TODO: log file will not be overwritten when restart
            init_params = self.setting.get_init_params()
            if add_step_info:
                if self.setting.task == "md":
                    data = np.loadtxt(self.directory/"dyn.log", dtype=float, skiprows=1)
                    if len(data.shape) == 1:
                        data = data[np.newaxis,:]
                    timesteps = data[:, 0] # ps
                    steps = [int(s) for s in timesteps*1000/init_params["timestep"]]
                    for time, atoms in zip(timesteps, traj_frames):
                        atoms.info["time"] = time*1000.
                elif self.setting.task == "min":
                    # Method - Step - Time - Energy - fmax
                    # BFGS:    0 22:18:46    -1024.329999        3.3947
                    data = np.loadtxt(self.directory/"dyn.log", dtype=str, skiprows=1)
                    if len(data.shape) == 1:
                        data = data[np.newaxis,:]
                    steps = [int(s) for s in data[:, 1]]
                    fmaxs = [float(fmax) for fmax in data[:, 4]]
                    for fmax, atoms in zip(fmaxs, traj_frames):
                        atoms.info["fmax"] = fmax
                assert len(steps) == len(traj_frames), "Number of steps and number of frames are inconsistent..."
                for step, atoms in zip(steps, traj_frames):
                    atoms.info["step"] = int(step)

            # - read deviation, similar to lammps
            # TODO: concatenate all deviations
            devi_fpath = self.directory / self.devi_fname
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
        else:
            ...

        return Trajectory(images=traj_frames, driver_config=dataclasses.asdict(self.setting))


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
            atoms=atoms, log_fpath=self.directory/self.xyz_fname
        )
        # NOTE: retrieve deviation info
        dynamics.attach(
            retrieve_and_save_deviation, interval=self.init_params["loginterval"], 
            atoms=atoms, devi_fpath=self.directory/self.devi_fname
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