#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses
import io
import pathlib
import tarfile
import traceback
from typing import Optional, List, Tuple
import warnings

import numpy as np

import yaml

from ase import Atoms
from ase import units

from ase.io import read, write
import ase.constraints
from ase.constraints import Filter
from ase.optimize.optimize import Dynamics
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from .. import config as GDPCONFIG
from ..potential.calculators.mixer import EnhancedCalculator
from .driver import AbstractDriver, DriverSetting, Controller
from .md.md_utils import force_temperature


def set_calc_state(calc: Calculator, timestep: float, stride: int):
    """Some calculators need driver information e.g. PLUMED."""
    if calc.name == "plumed":
        calc.timestep = timestep
        calc.stride = stride
    if hasattr(calc, "calcs"):
        for subcalc in calc.calcs:
            set_calc_state(subcalc, timestep, stride)
    else:
        ...

    return


def update_atoms_info(atoms: Atoms, dyn: Dynamics, start_step: int = 0) -> None:
    """Update step in atoms.info."""
    atoms.info["step"] = dyn.nsteps + start_step

    return


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

    # - save atoms info...
    atoms_to_save.info["step"] = atoms.info["step"]

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


def save_checkpoint(
    dyn: Dynamics, atoms: Atoms, wdir: pathlib.Path, start_step: int = 0
):
    """"""
    ckpt_wdir = wdir / f"checkpoint.{dyn.nsteps+start_step}"
    ckpt_wdir.mkdir(parents=True, exist_ok=True)

    # write(ckpt_wdir/"structure.xyz", atoms)
    save_trajectory(atoms=atoms, log_fpath=ckpt_wdir / "structures.xyz")

    # For some optimisers and dynamics, they use random generator.
    if hasattr(dyn, "rng"):
        with open(ckpt_wdir / "rng_state.yaml", "w") as fopen:
            yaml.safe_dump(dyn.rng.bit_generator.state, fopen)

    # For some mixed calculator, save information, for example, PLUMED...
    if hasattr(atoms.calc, "calcs"):
        for calc in atoms.calc.calcs:
            if hasattr(calc, "_save_checkpoint"):
                calc._save_checkpoint(ckpt_wdir)

    return


@dataclasses.dataclass
class BerendsenThermostat(Controller):

    name: str = "berendsen"

    def __post_init__(
        self,
    ):
        """"""
        taut = self.params.get("Tdamp", 100.0)
        assert taut is not None

        self.conv_params = dict(taut=taut * units.fs)

        return


@dataclasses.dataclass
class LangevinThermostat(Controller):

    name: str = "langevin"

    def __post_init__(
        self,
    ):
        """"""
        friction = self.params.get("friction", None)  # fs^-1
        assert friction is not None

        self.conv_params = dict(
            friction=friction / units.fs,
            # NOTE: The rng that generates friction normal distribution
            #       is set in `create_dynamics` by the driver's random_seed
            rng=None,
        )

        return


@dataclasses.dataclass
class NoseHooverThermostat(Controller):

    name: str = "nosehoover"

    def __post_init__(
        self,
    ):
        """"""
        nvt_q = self.params.get("nvt_q", None)
        assert nvt_q is not None

        self.conv_params = dict(nvt_q=nvt_q)  # thermostat mass

        return


@dataclasses.dataclass
class BerendsenBarostat(Controller):

    name: str = "berendsen"

    def __post_init__(
        self,
    ):
        """"""
        taut = self.params.get("Tdamp", 100.0)  # fs
        assert taut is not None

        taup = self.params.get("Pdamp", 100.0)  # fs
        assert taup is not None

        compressibility = self.params.get("compressibility", 4.57e-5)  # bar^-1

        self.conv_params = dict(
            taut=taut * units.fs,
            taup=taup * units.fs,
            compressibility_au=compressibility / units.bar,
        )

        return


controllers = dict(
    # - nvt
    berendsen_nvt=BerendsenThermostat,
    langevin_nvt=LangevinThermostat,
    nosehoover_nvt=NoseHooverThermostat,
    # - npt
    berendsen_npt=BerendsenBarostat,
)


@dataclasses.dataclass
class AseDriverSetting(DriverSetting):

    driver_cls: Optional[Dynamics] = None
    filter_cls: Optional[Filter] = None

    ensemble: str = "nve"

    controller: dict = dataclasses.field(default_factory=dict)

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
            )
            if self.ensemble == "nve":
                from ase.md.verlet import VelocityVerlet as driver_cls

                _init_md_params = dict(
                    timestep=self.timestep * units.fs,
                )
            elif self.ensemble == "nvt":
                if self.controller:
                    thermo_cls_name = self.controller["name"] + "_" + self.ensemble
                    thermo_cls = controllers[thermo_cls_name]
                else:
                    thermo_cls = BerendsenThermostat
                thermostat = thermo_cls(**self.controller)
                if thermostat.name == "berendsen":
                    from ase.md.nvtberendsen import NVTBerendsen as driver_cls
                elif thermostat.name == "langevin":
                    from ase.md.langevin import Langevin as driver_cls
                elif thermostat.name == "nose_hoover":
                    from .md.nosehoover import NoseHoover as driver_cls
                else:
                    raise RuntimeError(f"Unknown thermostat {thermostat}.")
                thermo_params = thermostat.conv_params
                _init_md_params = dict(
                    fixcm=self.fix_cm,
                    timestep=self.timestep * units.fs,
                    temperature_K=self.temp,
                )
                _init_md_params.update(**thermo_params)
            elif self.ensemble == "npt":
                if self.controller:
                    baro_cls_name = self.controller["name"] + "_" + self.ensemble
                    baro_cls = controllers[baro_cls_name]
                else:
                    baro_cls = BerendsenBarostat
                barostat = baro_cls(**self.controller)
                if barostat.name == "berendsen":
                    from ase.md.nptberendsen import NPTBerendsen as driver_cls
                else:
                    raise RuntimeError(f"Unknown barostat {barostat}.")
                baro_params = barostat.conv_params
                _init_md_params = dict(
                    fixcm=self.fix_cm,
                    timestep=self.timestep * units.fs,
                    temperature_K=self.temp,
                    pressure_au=self.press * (1e5 * units.Pascal),
                )
                _init_md_params.update(**baro_params)

            self._internals.update(**_init_md_params)

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

    def _create_dynamics(self, atoms: Atoms, *args, **kwargs) -> Tuple[Dynamics, dict]:
        """Create the correct class of this simulation with running parameters.

        Respect `steps` and `fmax` as restart.

        """
        # - overwrite
        run_params = self.setting.get_run_params(*args, **kwargs)

        # -
        self._preprocess_constraints(atoms, run_params)

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
            init_params_ = self.setting.get_init_params()

            # - velocity
            # NOTE: every dynamics will have a new rng...
            velocity_seed = init_params_.pop("velocity_seed")
            if velocity_seed is None:
                self._print(f"MD Driver's velocity_seed: {self.random_seed}")
                vrng = np.random.Generator(np.random.PCG64(self.random_seed))
            else:
                self._print(f"MD Driver's velocity_seed: {velocity_seed}")
                # vrng = np.random.default_rng(velocity_seed)
                vrng = np.random.Generator(np.random.PCG64(velocity_seed))

            ignore_atoms_velocities = init_params_.pop("ignore_atoms_velocities")
            if not ignore_atoms_velocities and atoms.get_kinetic_energy() > 0.0:
                # atoms have momenta
                ...
            else:
                MaxwellBoltzmannDistribution(
                    atoms, temperature_K=init_params_["temperature_K"], rng=vrng
                )
                if self.setting.remove_rotation:
                    ZeroRotation(atoms, preserve_temperature=False)
                if self.setting.remove_translation:
                    Stationary(atoms, preserve_temperature=False)
                # NOTE: respect constraints
                #       ase code does not consider constraints
                force_temperature(atoms, init_params_["temperature_K"], unit="K")

            # - some dynamics need rng
            if "rng" in init_params_:
                self._print(f"MD Driver's rng: {self.rng.bit_generator.state}")
                init_params_["rng"] = self.rng

            # - other callbacks
            set_calc_state(
                self.calc,
                timestep=init_params_["timestep"],
                stride=init_params_["loginterval"],
            )

            # - construct the driver
            driver = self.setting.driver_cls(
                atoms=atoms, **init_params_, logfile=self.log_fpath, trajectory=None
            )
        else:
            raise NotImplementedError(f"Unknown task {self.setting.task}.")

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

    def _find_latest_checkpoint(self, wdir: pathlib.Path):
        """"""
        ckpt_dirs = sorted(
            wdir.glob("checkpoint.*"), key=lambda x: int(x.name.split(".")[-1])
        )
        latest_ckpt_dir = ckpt_dirs[-1]

        return latest_ckpt_dir

    def _load_checkpoint(self, ckpt_dir: pathlib.Path):
        """"""
        atoms = read(ckpt_dir / "structures.xyz", ":")[-1]

        rng_state_fpath = ckpt_dir / "rng_state.yaml"
        if rng_state_fpath.exists():
            with open(ckpt_dir / "rng_state.yaml", "r") as fopen:
                rng_state = yaml.safe_load(fopen)
        else:
            rng_state = None

        return atoms, rng_state

    def _irun(
        self,
        atoms: Atoms,
        ckpt_wdir=None,
        cache_traj: List[Atoms] = None,
        *args,
        **kwargs,
    ):
        """Run the simulation."""
        prev_wdir = ckpt_wdir
        try:
            # To restart, velocities are always retained
            prev_ignore_atoms_velocities = self.setting.ignore_atoms_velocities
            if prev_wdir is None:  # start from the scratch
                start_step = 0
                rng_state = None

                curr_params = {}
                curr_params["random_seed"] = self.random_seed
                curr_params["init"] = self.setting.get_init_params()
                curr_params["run"] = self.setting.get_run_params()

                with open(self.directory / "params.yaml", "w") as fopen:
                    yaml.safe_dump(curr_params, fopen, indent=2)
            else:  # restart ...
                ckpt_wdir = self._find_latest_checkpoint(prev_wdir)
                atoms, rng_state = self._load_checkpoint(ckpt_wdir)
                start_step = atoms.info["step"]
                if hasattr(self.calc, "calcs"):
                    for calc in self.calc.calcs:
                        if hasattr(calc, "_load_checkpoint"):
                            calc._load_checkpoint(ckpt_wdir, start_step=start_step)
                # --- update run_params in settings
                target_steps = self.setting.get_run_params(*args, **kwargs)["steps"]
                if target_steps > 0:
                    steps = target_steps - start_step
                assert steps > 0, "Steps should be greater than 0."
                kwargs.update(steps=steps)

                # To restart, velocities are always retained
                self.setting.ignore_atoms_velocities = False

            # - set calculator
            atoms.calc = self.calc

            # - set dynamics
            dynamics, run_params = self._create_dynamics(atoms, *args, **kwargs)
            # dynamics.nsteps = start_step
            if hasattr(dynamics, "rng") and rng_state is not None:
                dynamics.rng.bit_generator.state = rng_state

            # --- callback functions
            dynamics.insert_observer(
                update_atoms_info,
                dyn=dynamics,
                atoms=atoms,
                start_step=start_step,
            )
            # NOTE: traj file not stores properties (energy, forces) properly
            init_params = self.setting.get_init_params()
            dynamics.attach(
                save_checkpoint,
                interval=self.setting.ckpt_period,
                dyn=dynamics,
                atoms=atoms,
                wdir=self.directory,
                start_step=start_step,
            )
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
            dump_period = self.setting.dump_period
            assert init_params["loginterval"] == dump_period

            if dump_period > 1:
                should_dump_last = False
                if self.setting.task == "min":
                    # optimiser dumps every step to log...
                    data = np.loadtxt(self.directory / "dyn.log", dtype=str, skiprows=1)
                    if len(data.shape) == 1:
                        data = data[np.newaxis, :]
                    nsteps = data.shape[0]
                    if nsteps > 0 and (nsteps - 1) % dump_period != 0:
                        should_dump_last = True
                elif self.setting.task == "md":
                    if dump_period > self.setting.steps:
                        should_dump_last = True
                if should_dump_last:
                    update_atoms_info(atoms, dynamics, start_step=start_step)
                    save_checkpoint(
                        dynamics, atoms, self.directory, start_step=start_step
                    )
                    save_trajectory(atoms, self.directory / self.xyz_fname)
                    retrieve_and_save_deviation(atoms, self.directory / self.devi_fname)
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
            if (wdir / self.xyz_fname).exists():
                frames = read(wdir / self.xyz_fname, ":")
            else:
                frames = []
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
                else:
                    frames = []

        return frames

    def read_trajectory(self, archive_path=None, *args, **kwargs) -> List[Atoms]:
        """Read trajectory in the current working directory."""
        # - read trajectory
        traj_frames = self._aggregate_trajectories(archive_path=archive_path)

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
