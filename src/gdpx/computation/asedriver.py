#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import dataclasses
import io
import shutil
import pathlib
import tarfile
import traceback
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
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from ase.calculators.singlepoint import SinglePointCalculator

from .. import config as GDPCONFIG
from ..builder.constraints import parse_constraint_info
from .driver import AbstractDriver, DriverSetting
from .md.md_utils import force_temperature
from ..potential.calculators.mixer import EnhancedCalculator

from .plumed import set_plumed_state


def retrieve_and_save_deviation(atoms, devi_fpath) -> None:
    """Read model deviation and add results to atoms.info if the file exists."""
    results = copy.deepcopy(atoms.calc.results)
    # devi_results = [(k,v) for k,v in results.items() if "devi" in k]
    devi_results = [
        (k, v) for k, v in results.items() if k in GDPCONFIG.VALID_DEVI_FRAME_KEYS
    ]
    if devi_results:
        devi_names = [x[0] for x in devi_results]
        devi_values = np.array([x[1] for x in devi_results]).reshape(1, -1)

        if devi_fpath.exists():
            with open(devi_fpath, "a") as fopen:
                np.savetxt(fopen, devi_values, fmt="%18.6e")
        else:
            with open(devi_fpath, "w") as fopen:
                np.savetxt(
                    fopen,
                    devi_values,
                    fmt="%18.6e",
                    header=("{:>18s}" * len(devi_names)).format(*devi_names),
                )

    return


def save_trajectory(atoms, log_fpath) -> None:
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
        pbc=copy.deepcopy(atoms.get_pbc()),
    )
    if "tags" in atoms.arrays:
        atoms_to_save.set_tags(atoms.get_tags())
    if atoms.get_kinetic_energy() > 0.0:
        atoms_to_save.set_momenta(atoms.get_momenta())
    results = dict(
        energy=atoms.get_potential_energy(), forces=copy.deepcopy(atoms.get_forces())
    )
    spc = SinglePointCalculator(atoms, **results)
    atoms_to_save.calc = spc

    # - save special keys and arrays from calc
    natoms = len(atoms)

    # -- add deviation
    for k, v in atoms.calc.results.items():
        if k in GDPCONFIG.VALID_DEVI_FRAME_KEYS:
            atoms_to_save.info[k] = v
    for k, v in atoms.calc.results.items():
        if k in GDPCONFIG.VALID_DEVI_ATOMIC_KEYS:
            atoms_to_save.arrays[k] = np.reshape(v, (natoms, -1))
    # print(f"keys: {atoms.calc.results.keys()}")

    # -- check special metadata
    calc = atoms.calc
    if isinstance(calc, EnhancedCalculator):
        atoms_to_save.info["host_energy"] = copy.deepcopy(calc.results["host_energy"])
        atoms_to_save.info["bias_energy"] = (
            results["energy"] - calc.results["host_energy"]
        )
        atoms_to_save.arrays["host_forces"] = copy.deepcopy(calc.results["host_forces"])

    # - append to traj
    write(log_fpath, atoms_to_save, append=True)

    return


@dataclasses.dataclass
class AseDriverSetting(DriverSetting):

    driver_cls: Dynamics = None
    filter_cls: Filter = None

    thermostat: str = None
    friction: float = 0.01
    friction_seed: int = None

    fix_cm: bool = False

    fmax: float = 0.05  # eV/Ang

    def __post_init__(self):
        """"""
        # - task-specific params
        if self.task == "md":
            self._internals.update(
                # - helper params
                velocity_seed=self.velocity_seed,
                ignore_atoms_velocities=self.ignore_atoms_velocities,
                md_style=self.md_style,  # ensemble...
                thermostat=self.thermostat,
                # -
                fixcm=self.fix_cm,
                timestep=self.timestep,
                temperature_K=self.temp,
                taut=self.Tdamp,
                pressure=self.press,
                taup=self.Pdamp,
            )
            # TODO: provide a unified class for thermostat
            if self.md_style == "nve":
                from ase.md.verlet import VelocityVerlet as driver_cls
            elif self.md_style == "nvt":
                thermostat = (
                    self.thermostat if self.thermostat is not None else "berendsen"
                )
                if thermostat == "berendsen":
                    from ase.md.nvtberendsen import NVTBerendsen as driver_cls

                    thermostat_params = dict(taut=self._internals["taut"])
                elif thermostat == "langevin":
                    from ase.md.langevin import Langevin as driver_cls

                    if self.friction_seed is not None:
                        friction_seed = self.friction_seed
                    else:
                        friction_seed = np.random.randint(0, 100000000)
                    thermostat_params = dict(
                        friction=self.friction / units.fs,
                        # TODO: use driver's seed instead!
                        rng=np.random.default_rng(seed=friction_seed),
                    )
                elif thermostat == "nose_hoover":
                    from .md.nosehoover import NoseHoover as driver_cls
                else:
                    raise RuntimeError(f"Unknown thermostat {thermostat}.")
                self._internals["thermostat"] = thermostat
                self._internals["thermostat_params"] = thermostat_params
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
        self._internals.update(loginterval=self.dump_period)

        # NOTE: There is a bug in ASE as it checks `if steps` then fails when spc.
        if self.steps == 0:
            self.steps = -1

        return

    def get_run_params(self, *args, **kwargs) -> dict:
        """"""
        run_params = dict(
            steps=kwargs.get("steps", self.steps),
            constraint=kwargs.get("constraint", self.constraint),
        )
        if self.task == "min" or self.task == "ts":
            run_params.update(
                fmax=kwargs.get("fmax", self.fmax),
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
    xyz_fname = "traj.xyz"
    devi_fname = "model_devi-ase.dat"

    #: List of output files would be saved when restart.
    saved_fnames: List[str] = [log_fname, xyz_fname, devi_fname]

    #: List of output files would be removed when restart.
    removed_fnames: List[str] = [log_fname, xyz_fname, devi_fname]

    def __init__(self, calc=None, params: dict = {}, directory="./", *args, **kwargs):
        """"""
        super().__init__(calc, params, directory, *args, **kwargs)

        self.setting = AseDriverSetting(**params)

        self._log_fpath = self.directory / self.log_fname

        return

    @property
    def log_fpath(self):
        """File path of the simulation log."""

        return self._log_fpath

    @AbstractDriver.directory.setter
    def directory(self, directory_):
        """Set log and traj path regarding to the working directory."""
        # - main and calc
        super(AseDriver, AseDriver).directory.__set__(self, directory_)

        # - other files
        self._log_fpath = self.directory / self.log_fname

        return

    def _create_dynamics(self, atoms, *args, **kwargs) -> Tuple[Dynamics, dict]:
        """Create the correct class of this simulation with running parameters.

        Respect `steps` and `fmax` as restart.

        """
        # - overwrite
        run_params = self.setting.get_run_params(*args, **kwargs)

        # NOTE: if have cons in kwargs overwrite current cons stored in atoms
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
                atoms, logfile=self.log_fpath, trajectory=None
            )
        elif self.setting.task == "ts":
            driver = self.setting.driver_cls(
                atoms, order=1, internal=False, logfile=self.log_fpath, trajectory=None
            )
        elif self.setting.task == "md":
            # - adjust params
            init_params_ = copy.deepcopy(self.setting.get_init_params())
            # NOTE: every dynamics will have a new rng...
            self._print(f"MD Driver's random_seed: {self.random_seed}")
            rng = np.random.default_rng(self.random_seed)

            # - velocity
            if (
                not init_params_["ignore_atoms_velocities"]
                and atoms.get_kinetic_energy() > 0.0
            ):
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
                init_params_ = {
                    k: v
                    for k, v in init_params_.items()
                    if k in ["loginterval", "timestep"]
                }
            elif md_style == "nvt":
                init_params_ = {
                    k: v
                    for k, v in init_params_.items()
                    if k in ["loginterval", "fixcm", "timestep", "temperature_K"]
                }
                init_params_.update(**self.setting._internals["thermostat_params"])
            elif md_style == "npt":
                init_params_ = {
                    k: v
                    for k, v in init_params_.items()
                    if k
                    in [
                        "loginterval",
                        "fixcm",
                        "timestep",
                        "temperature_K",
                        "taut",
                        "pressure",
                        "taup",
                    ]
                }
                init_params_["pressure"] *= 1.0 / (160.21766208 / 0.000101325)

            init_params_["timestep"] *= units.fs

            # NOTE: plumed
            set_plumed_state(
                self.calc,
                timestep=init_params_["timestep"],
                stride=init_params_["loginterval"],
            )

            # - construct the driver
            driver = self.setting.driver_cls(
                atoms=atoms, **init_params_, logfile=self.log_fpath, trajectory=None
            )
        else:
            raise NotImplementedError(f"Unknown task {self.task}.")

        return driver, run_params

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """"""
        verified = super()._verify_checkpoint()
        if verified:
            asetraj = self.directory / self.xyz_fname
            if asetraj.exists() and asetraj.stat().st_size != 0:
                temp_frames = read(asetraj, ":")
                try:
                    _ = temp_frames[0].get_forces()
                except:  # `RuntimeError: Atoms object has no calculator.`
                    verified = False
            else:
                verified = False
        else:
            ...

        return verified

    def _irun(self, atoms: Atoms, ckpt_wdir=None, *args, **kwargs):
        """Run the simulation."""
        try:
            # To restart, velocities are always retained
            prev_ignore_atoms_velocities = self.setting.ignore_atoms_velocities
            if ckpt_wdir is None:  # start from the scratch
                curr_params = {}
                curr_params["random_seed"] = self.random_seed
                curr_params["init"] = self.setting.get_init_params()
                if self.setting.task == "md":  # check rng for MD simulations...
                    if "rng" in curr_params["init"]["thermostat_params"]:  # langevin...
                        rng = curr_params["init"]["thermostat_params"]["rng"]
                        curr_params["init"]["thermostat_params"][
                            "rng"
                        ] = rng._bit_generator.state
                else:
                    ...
                curr_params["run"] = self.setting.get_run_params()
                import yaml

                with open(self.directory / "params.yaml", "w") as fopen:
                    yaml.safe_dump(curr_params, fopen, indent=2)
            else:  # restart ...
                traj = self.read_trajectory()
                nframes = len(traj)
                assert nframes > 0, "AseDriver restarts with a zero-frame trajectory."
                atoms = traj[-1]
                # --- update run_params in settings
                dump_period = self.setting.get_init_params()["loginterval"]
                target_steps = self.setting.get_run_params(*args, **kwargs)["steps"]
                if target_steps > 0:
                    steps = target_steps + dump_period - nframes * dump_period
                assert steps > 0, "Steps should be greater than 0."
                kwargs.update(steps=steps)

                # To restart, velocities are always retained
                self.setting.ignore_atoms_velocities = False

            # - set calculator
            atoms.calc = self.calc

            # - set dynamics
            dynamics, run_params = self._create_dynamics(atoms, *args, **kwargs)

            # NOTE: traj file not stores properties (energy, forces) properly
            init_params = self.setting.get_init_params()
            dynamics.attach(
                save_trajectory,
                interval=init_params["loginterval"],
                atoms=atoms,
                log_fpath=self.directory / self.xyz_fname,
            )
            # NOTE: retrieve deviation info
            dynamics.attach(
                retrieve_and_save_deviation,
                interval=init_params["loginterval"],
                atoms=atoms,
                devi_fpath=self.directory / self.devi_fname,
            )
            dynamics.run(**run_params)
            # NOTE: check if the last frame is properly stored
            loginterval = init_params["loginterval"]
            if loginterval > 1:
                if self.setting.task == "min":
                    data = np.loadtxt(self.directory/"dyn.log", dtype=str, skiprows=1)
                    if len(data.shape) == 1:
                        data = data[np.newaxis,:]
                    nsteps = data.shape[0]
                    if nsteps > 0 and (nsteps - 1) % loginterval != 0:
                        save_trajectory(atoms, self.directory / self.xyz_fname)
                        retrieve_and_save_deviation(
                            atoms, self.directory / self.devi_fname
                        )
                else:  # TODO: If MD breaks due to some errors?
                    ...
            else:
                ...

            # - Some interactive calculator needs kill processes after finishing,
            #   e.g. VaspInteractive...
            if hasattr(self.calc, "finalize"):
                self.calc.finalize()
            # To restart, velocities are always retained
            self.setting.ignore_atoms_velocities = prev_ignore_atoms_velocities
        except Exception as e:
            self._debug(f"Exception of {self.__class__.__name__} is {e}.")
            self._debug(
                f"Exception of {self.__class__.__name__} is {traceback.format_exc()}."
            )

        return

    def read_force_convergence(self, *args, **kwargs) -> bool:
        """Check if the force is converged.

        Sometimes DFT failed to converge SCF due to improper structure.

        """
        # - check convergence of forace evaluation (e.g. SCF convergence)
        scf_convergence = False
        try:
            scf_convergence = self.calc.read_convergence()
        except:
            # -- cannot read scf convergence then assume it is ok
            scf_convergence = True
        if not scf_convergence:
            warnings.warn(
                f"{self.name} at {self.directory} failed to converge at SCF.",
                RuntimeWarning,
            )
        # if not converged:
        #    warnings.warn(f"{self.name} at {self.directory} failed to converge.", RuntimeWarning)

        return scf_convergence

    def _read_a_single_trajectory(
        self, wdir: pathlib.Path, archive_path: pathlib.Path = None, *args, **kwargs
    ):
        """"""
        self._debug(f"archive_path: {archive_path}")
        self._debug(f"wdir: {wdir}")
        if archive_path is None:
            frames = read(wdir / self.xyz_fname, ":")  # TODO: check traj existence?
        else:
            target_name = str(
                (wdir / self.xyz_fname).relative_to(self.directory.parent)
            )
            with tarfile.open(archive_path, "r:gz") as tar:
                for tarinfo in tar:
                    if tarinfo.name == target_name:
                        fobj = io.StringIO(
                            tar.extractfile(tarinfo.name).read().decode()
                        )
                        frames = read(fobj, ":", format="extxyz")
                        fobj.close()
                        break
                else:  # TODO: if not find target traj?
                    ...

        return frames

    def read_trajectory(self, archive_path=None, *args, **kwargs) -> List[Atoms]:
        """Read trajectory in the current working directory."""
        # -
        prev_wdirs = sorted(self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"))
        self._debug(f"prev_wdirs: {prev_wdirs}")

        traj_list = []
        for w in prev_wdirs:
            curr_frames = self._read_a_single_trajectory(w, archive_path=archive_path)
            traj_list.append(curr_frames)

        # Even though xyz file may be empty, the read can give a empty list...
        curr_frames = self._read_a_single_trajectory(
            self.directory, archive_path=archive_path
        )
        traj_list.append(curr_frames)

        # -- concatenate
        traj_frames, ntrajs = [], len(traj_list)
        if ntrajs > 0:
            traj_frames.extend(traj_list[0])
            for i in range(1, ntrajs):
                assert np.allclose(
                    traj_list[i - 1][-1].positions, traj_list[i][0].positions
                ), f"Traj {i-1} and traj {i} are not consecutive."
                traj_frames.extend(traj_list[i][1:])
        else:
            ...

        # - add some info
        init_params = self.setting.get_init_params()
        if self.setting.task == "md":
            # Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]
            # 0.0000           3.4237       2.8604       0.5633   272.4
            # data = np.loadtxt(self.directory/"dyn.log", dtype=float, skiprows=1)
            # if len(data.shape) == 1:
            #    data = data[np.newaxis,:]
            # timesteps = data[:, 0] # ps
            # steps = [int(s) for s in timesteps*1000/init_params["timestep"]]
            # ... infer from input settings
            for i, atoms in enumerate(traj_frames):
                atoms.info["time"] = (
                    i * init_params["timestep"] * init_params["loginterval"]
                )
        elif self.setting.task == "min":
            # Method - Step - Time - Energy - fmax
            # BFGS:    0 22:18:46    -1024.329999        3.3947
            # data = np.loadtxt(self.directory/"dyn.log", dtype=str, skiprows=1)
            # if len(data.shape) == 1:
            #    data = data[np.newaxis,:]
            # steps = [int(s) for s in data[:, 1]]
            # fmaxs = [float(fmax) for fmax in data[:, 4]]
            for atoms in traj_frames:
                atoms.info["fmax"] = np.max(
                    np.fabs(atoms.get_forces(apply_constraint=True))
                )
        # assert len(steps) == len(traj_frames), f"Number of steps {len(steps)} and number of frames {len(traj_frames)} are inconsistent..."
        for step, atoms in enumerate(traj_frames):
            atoms.info["step"] = int(step) * init_params["loginterval"]

        # - deviation stored in traj, no need to read from file

        nframes = len(traj_frames)

        # calculation happens but some errors in calculation
        if self.directory.exists():
            # - check the convergence of the force evaluation
            try:
                scf_convergence = self.calc.read_convergence()
            except:
                # -- cannot read scf convergence then assume it is ok
                scf_convergence = True
            if not scf_convergence:
                warnings.warn(
                    f"{self.name} at {self.directory.name} failed to converge at SCF.",
                    RuntimeWarning,
                )
                if nframes > 0:  # for compat
                    traj_frames[0].info[
                        "error"
                    ] = f"Unconverged SCF at {self.directory}."
                traj_frames.error = True
        else:
            # TODO: How about archived data?
            ...

        return traj_frames


if __name__ == "__main__":
    ...
