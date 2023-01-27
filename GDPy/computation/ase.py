#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import shutil
from pathlib import Path
from typing import NoReturn, List, Tuple

import numpy as np

from ase import Atoms
from ase import units

from ase.io import read, write
import ase.constraints
from ase.constraints import FixAtoms
from ase.optimize.optimize import Dynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.computation.driver import AbstractDriver
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
                np.savetxt(fopen, devi_values, fmt="%18.6e", header=" ".join(devi_names))

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
            steps= -1, # for spc, steps=0 would fail to step dynamics.max_steps in ase 3.22.1
            fmax = 0.05
        ),
        "ts": {},
        "md": dict(
            steps = -1
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

    #: List of output files would be saved when restart.
    saved_cards: List[str] = [traj_fname]

    def __init__(
        self, calc=None, params: dict={}, directory="./"
    ):
        """"""
        super().__init__(calc, params, directory)

        self._log_fpath = self.directory / self.log_fname
        self._traj_fpath = self.directory / self.traj_fname

        return
    
    def _parse_params(self, params):
        """Parse different tasks, and prepare init and run params."""
        super()._parse_params(params)

        self.driver_cls, self.filter_cls = None, None
        if self.task == "min":
            # - to opt atomic positions
            from ase.optimize import BFGS
            if self.init_params["min_style"] == "bfgs":
                driver_cls = BFGS
            # - to opt unit cell
            #   UnitCellFilter, StrainFilter, ExpCellFilter
            # TODO: add filter params
            filter_names = ["unitCellFilter", "StrainFilter", "ExpCellFilter"]
            if self.init_params["min_style"] in filter_names:
                driver_cls = BFGS
                self.filter_cls = getattr(ase.constraints, self.init_params["min_style"])
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
            if self.filter_cls:
                atoms = self.filter_cls(atoms)
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

    def run(self, atoms_, read_exists: bool=True, extra_info: dict=None, *args, **kwargs):
        """Run the driver.

        Additional output files would be generated, namely a xyz-trajectory and
        a deviation file if the calculator could estimate uncertainty.

        Note:
            Calculator's parameters will not change since it still performs 
            single-point calculations as the simulation goes.

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
        """Read trajectory in the current working directory."""
        traj_frames = read(self.directory/"traj.xyz", index=":")

        if add_step_info:
            if self.task == "md":
                data = np.loadtxt(self.directory/"dyn.log", dtype=float, skiprows=1)
                timesteps = data[:, 0] # ps
                steps = timesteps*1000/self.init_params["timestep"]
            elif self.task == "min":
                data = np.loadtxt(self.directory/"dyn.log", dtype=str, skiprows=1)
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