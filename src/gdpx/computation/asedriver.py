#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses
import functools
import io
import json
import pathlib
import shutil
import tarfile
import traceback
import warnings
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.md.md import MolecularDynamics
from ase.optimize.optimize import Dynamics

from .. import config as GDPCONFIG
from ..potential.calculators.mixer import EnhancedCalculator
from .driver import EARLYSTOP_KEY, AbstractDriver, Controller, DriverSetting
from .observer import create_an_observer


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


def update_atoms_info(atoms: Atoms, dyn: Dynamics) -> None:
    """Update step in atoms.info."""
    atoms.info["step"] = dyn.nsteps

    return


def update_target_temperature(dyn: MolecularDynamics, dtemp: float) -> None:
    """Update thermostat's target temperature at each step.

    Args:
        dyn: Dynamics object.
        dtemp: The delta temperature at each step.

    """
    temperature = None
    try:  # berendsen_nvt
        temperature = dyn.get_temperature()
    except:  # langevin
        temperature = dyn.temp / units.kB
    finally:
        assert temperature is not None

    target_temperature = temperature + dtemp
    dyn.set_temperature(temperature_K=target_temperature)

    return


def update_target_pressure(dyn: MolecularDynamics, dpres: float) -> None:
    """Update barostat's target pressure at each step.

    Args:
        dyn: Dynamics object.
        dpres: The delta pressure at each step.

    """
    pressure = dyn.get_pressure() / (1e5 * units.Pascal)

    target_pressure = pressure + dpres
    dyn.pressure = target_pressure * 1e5 * units.Pascal

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


def save_trajectory(atoms, traj_fpath) -> None:
    """Create a clean atoms from the input and save simulation trajectory.

    We need an explicit copy of atoms as some calculators may not return all
    necessary information. For example, schnet only returns required properties.
    If only energy is required, there are no forces.

    """
    # save atoms
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
    try:
        results.update(stress=atoms.get_stress())
    except:
        # Some calculators may not have stress data.
        # Also, some calcs also give stress when input atoms have pbc.
        # If a mixer calc is used and one of its calcs do not have stress,
        # the entire save_trajectory failed.
        # Maybe we should tell this function only cmin must have stress info.
        ...

    spc = SinglePointCalculator(atoms_to_save, **results)
    atoms_to_save.calc = spc

    # - save atoms info...
    atoms_to_save.info["step"] = atoms.info["step"]
    if EARLYSTOP_KEY in atoms.info:
        atoms_to_save.info[EARLYSTOP_KEY] = atoms.info[EARLYSTOP_KEY]

    # - save special keys and arrays from calc
    num_atoms = len(atoms)

    # -- add deviation
    for k, v in atoms.calc.results.items():
        if k in GDPCONFIG.VALID_DEVI_FRAME_KEYS:
            atoms_to_save.info[k] = v
    for k, v in atoms.calc.results.items():
        if k in GDPCONFIG.VALID_DEVI_ATOMIC_KEYS:
            atoms_to_save.arrays[k] = np.reshape(v, (num_atoms, -1))
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
    write(traj_fpath, atoms_to_save, append=True)

    return


def save_checkpoint(
    dyn: Dynamics, atoms: Atoms, wdir: pathlib.Path, ckpt_number: int = 3
):
    """"""
    if dyn.nsteps > 0:
        ckpt_wdir = wdir / f"checkpoint.{dyn.nsteps}"
        ckpt_wdir.mkdir(parents=True, exist_ok=True)

        # write(ckpt_wdir/"structure.xyz", atoms)
        save_trajectory(atoms=atoms, traj_fpath=ckpt_wdir / "structures.xyz")

        # For some optimisers and dynamics, they use random generator.
        if hasattr(dyn, "rng"):
            with open(ckpt_wdir / "rng_state.json", "w") as fopen:
                json.dump(dyn.rng.bit_generator.state, fopen, indent=2)

        # For some mixed calculator, save information, for example, PLUMED...
        if hasattr(atoms.calc, "calcs"):
            for calc in atoms.calc.calcs:
                if hasattr(calc, "_save_checkpoint"):
                    calc._save_checkpoint(ckpt_wdir)

        # remove checkpoints if the number is over ckpt_number
        ckpt_wdirs = sorted(wdir.glob("checkpoint*"), key=lambda x: int(x.name[11:]))
        num_ckpts = len(ckpt_wdirs)
        if num_ckpts > ckpt_number:
            for w in ckpt_wdirs[:-ckpt_number]:
                shutil.rmtree(w)
    else:
        # Do not save checkpoint at step 0.
        # Sometime it may be useful to save a ckpt at step 0 if
        # an expensive potential is used. However, we normally
        # use ase-backend for very quick potentials.
        ...

    return


def monit_and_intervene(
    atoms: Atoms, dynamics: Dynamics, observer, print_func=print
) -> None:
    """"""
    if dynamics.nsteps >= observer.patience:
        if observer.run(atoms):
            dynamics.max_steps = 0
            atoms.info[EARLYSTOP_KEY] = True
            print_func(f"EARLY STOPPED at step {dynamics.nsteps}!!")

    return


@dataclasses.dataclass
class BFGSMinimiser(Controller):

    name: str = "bfgs"

    def __post_init__(self):
        """"""
        from ase.optimize import BFGS

        maxstep = self.params.get("maxstep", 0.2)  # Ang
        assert maxstep is not None

        self.params.update(driver_cls=functools.partial(BFGS, maxstep=maxstep))

        return


@dataclasses.dataclass
class BFGSCellMinimiser(Controller):

    name: str = "bfgs"

    def __post_init__(self):
        """"""
        from ase.optimize import BFGS as min_cls

        maxstep = self.params.get("maxstep", 0.2)  # Ang
        assert maxstep is not None

        isotropic = self.params.get("isotropic", False)  # Ang
        assert isotropic is not None

        pressure = self.params.get("pressure", 1.0)  # bar
        assert pressure is not None
        pressure *= 1e-4 / 160.21766208  # bar -> eV/Ang^3

        # TODO: StrainFilter, FrechetCellFilter
        from ase.filters import UnitCellFilter as filter_cls

        def combine_filter_and_minimiser(atoms, **kwargs):
            """"""
            new_filter_cls = functools.partial(
                filter_cls, hydrostatic_strain=isotropic, scalar_pressure=pressure
            )

            return min_cls(atoms=new_filter_cls(atoms), maxstep=maxstep, **kwargs)  # type: ignore

        self.params.update(driver_cls=combine_filter_and_minimiser)

        return


@dataclasses.dataclass
class MDController(Controller):

    #: Controller name.
    name: str = "md"

    #: Timestep in fs.
    timestep: float = 1.0

    #: Temperature at beginning in Kelvin.
    temperature: float = 300.0

    #: Temperature at end in Kelvin.
    temperature_end: Optional[float] = None

    #: Pressure in bar.
    pressure: float = 1.0

    #: Pressure in bar.
    pressure_end: Optional[float] = None

    #: Whether fix center of mass.
    fix_com: bool = True

    def __post_init__(self):
        """"""

        self.timestep *= units.fs
        self.pressure *= 1e5 * units.Pascal
        if self.pressure_end is not None:
            self.pressure_end *= 1e5 * units.Pascal

        return


@dataclasses.dataclass
class Verlet(MDController):

    def __post_init__(self):
        """"""
        super().__post_init__()

        if self.temperature_end is not None:
            raise Exception("AseDriver verlet_nve does not `tend`.")

        from ase.md.verlet import VelocityVerlet

        driver_cls = functools.partial(VelocityVerlet, timestep=self.timestep)
        self.params.update(driver_cls=driver_cls)

        return


@dataclasses.dataclass
class BerendsenThermostat(MDController):

    name: str = "berendsen"

    def __post_init__(
        self,
    ):
        super().__post_init__()

        taut = self.params.get("Tdamp", 100.0)
        taut *= units.fs
        assert taut is not None

        from ase.md.nvtberendsen import NVTBerendsen

        driver_cls = functools.partial(
            NVTBerendsen,
            timestep=self.timestep,
            temperature=self.temperature,
            fixcm=self.fix_com,
            taut=taut,
        )
        self.params.update(driver_cls=driver_cls)

        return


@dataclasses.dataclass
class LangevinThermostat(MDController):

    name: str = "langevin"

    def __post_init__(
        self,
    ):
        """"""
        super().__post_init__()

        # NOTE: The rng that generates friction normal distribution
        #       is set in `create_dynamics` by the driver's random_seed
        friction = self.params.get("friction", 0.01)  # fs^-1
        friction *= 1.0 / units.fs
        assert friction is not None

        from ase.md.langevin import Langevin

        driver_cls = functools.partial(
            Langevin,
            timestep=self.timestep,
            temperature_K=self.temperature,
            fixcm=self.fix_com,
            friction=friction,
        )
        self.params.update(driver_cls=driver_cls)

        return


@dataclasses.dataclass
class NoseHooverThermostat(MDController):

    name: str = "nose_hoover"

    def __post_init__(
        self,
    ):
        """"""
        super().__post_init__()

        if self.temperature_end is not None:
            raise Exception("AseDriver nose_hoover_nvt does not `tend`.")

        qmass = self.params.get("nvt_q", 334.0)  # a.u.
        assert qmass is not None

        from .md.nosehoover import NoseHoover

        driver_cls = functools.partial(
            NoseHoover,
            timestep=self.timestep,
            temperature=self.temperature * units.kB,
            nvt_q=qmass,
        )
        self.params.update(driver_cls=driver_cls)

        return


@dataclasses.dataclass
class BerendsenBarostat(MDController):

    name: str = "berendsen"

    def __post_init__(
        self,
    ):
        """"""
        super().__post_init__()

        taut = self.params.get("Tdamp", 100.0)  # fs
        taut *= units.fs
        assert taut is not None

        taup = self.params.get("Pdamp", 100.0)  # fs
        taup *= units.fs
        assert taup is not None

        # The default value is the one of water.
        compressibility = self.params.get("compressibility", 4.57e-5)  # bar^-1
        compressibility *= compressibility / units.bar

        from ase.md.nptberendsen import NPTBerendsen

        driver_cls = functools.partial(
            NPTBerendsen,
            timestep=self.timestep,
            temperature=self.temperature,
            pressure=self.pressure,
            fixcm=self.fix_com,
            taut=taut,
            taup=taup,
            compressibility_au=compressibility,
        )
        self.params.update(driver_cls=driver_cls)

        return


@dataclasses.dataclass
class MonteCarloController(MDController):

    name: str = "monte_carlo"

    def __post_init__(self):
        """"""
        super().__post_init__()

        if self.temperature_end is not None:
            raise Exception("AseDriver monte_carlo_nvt does not `tend`.")

        maxstepsize = self.params.get("maxstepsizes", 0.2)  # Ang
        assert maxstepsize is not None

        from .mc.tfmc import TimeStampedMonteCarlo

        driver_cls = functools.partial(
            TimeStampedMonteCarlo,
            timestep=self.timestep,
            temperature=self.temperature,
            fixcm=self.fix_com,
            maxstepsize=maxstepsize,
        )
        self.params.update(driver_cls=driver_cls)

        return


controllers = dict(
    # - min
    bfgs_min=BFGSMinimiser,
    # - cmin
    bfgs_cmin=BFGSCellMinimiser,
    # - nve
    verlet_nve=Verlet,
    # - nvt
    berendsen_nvt=BerendsenThermostat,
    langevin_nvt=LangevinThermostat,
    nose_hoover_nvt=NoseHooverThermostat,
    monte_carlo_nvt=MonteCarloController,
    # - npt
    berendsen_npt=BerendsenBarostat,
)

default_controllers = dict(
    min=BFGSMinimiser,
    cmin=BFGSCellMinimiser,
    nve=Verlet,
    nvt=BerendsenThermostat,
    npt=BerendsenBarostat,
)


@dataclasses.dataclass
class AseDriverSetting(DriverSetting):

    #: MD ensemble.
    ensemble: str = "nve"

    #: Dynamics controller.
    controller: dict = dataclasses.field(default_factory=dict)

    #: Force tolerance.
    fmax: Optional[float] = 0.05  # eV/Ang

    #: Whether fix com to the its initial position.
    fix_com: bool = False

    driver_cls: Optional[Dynamics] = None

    def __post_init__(self):
        """"""
        _init_params = {}
        if self.task == "md":
            suffix = self.ensemble
            _init_params.update(
                timestep=self.timestep,
                temperature=self.temp,
                pressure=self.press,
                fix_com=self.fix_com,
            )
        else:
            suffix = self.task
        _init_params.update(**self.controller)

        if self.controller:
            cont_cls_name = self.controller["name"] + "_" + suffix
            if cont_cls_name in controllers:
                cont_cls = controllers[cont_cls_name]
            else:
                raise RuntimeError(f"Unknown controller {cont_cls_name}.")
        else:
            cont_cls = default_controllers[suffix]

        cont = cont_cls(**_init_params)
        self.driver_cls = cont.params.pop("driver_cls")

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

        return run_params


class AseDriver(AbstractDriver):

    #: Driver name.
    name = "ase"

    #: Log filename.
    log_fname: str = "dyn.log"

    #: Trajectory filename.
    xyz_fname: str = "traj.xyz"

    #: Model deviation fielname.
    devi_fname: str = "model_devi-ase.dat"

    #: Class for setting.
    setting_cls: type[DriverSetting] = AseDriverSetting

    @property
    def log_fpath(self):
        """File path of the simulation log."""

        return self.directory / self.log_fname

    def _create_dynamics(
        self, atoms: Atoms, start_step: int = 0, *args, **kwargs
    ) -> Tuple[Dynamics, dict]:
        """Create the correct class of this simulation with running parameters.

        Respect `steps` and `fmax` as restart.

        """
        # - overwrite
        run_params = self.setting.get_run_params(*args, **kwargs)

        # -
        self._preprocess_constraints(atoms, run_params)

        # - init driver
        if self.setting.task == "min":
            driver = self.setting.driver_cls(
                atoms, logfile=self.log_fpath, trajectory=None
            )
        elif self.setting.task == "cmin":
            driver = self.setting.driver_cls(
                atoms, logfile=self.log_fpath, trajectory=None
            )
        elif self.setting.task == "md":
            # velocity
            self._prepare_velocities(
                atoms, self.setting.velocity_seed, self.setting.ignore_atoms_velocities
            )

            # other callbacks
            set_calc_state(
                self.calc,
                timestep=self.setting.timestep,
                stride=self.setting.dump_period,
            )

            # construct the driver
            driver = self.setting.driver_cls(
                atoms=atoms, logfile=self.log_fpath, trajectory=None
            )

            # check if the simulation is annealing
            if self.setting.tend is not None:
                dtemp = (self.setting.tend - self.setting.temp) / self.setting.steps
                driver.set_temperature(
                    temperature_K=self.setting.temp + (start_step - 1) * dtemp
                )
                driver.attach(
                    update_target_temperature, dyn=driver, dtemp=dtemp, interval=1
                )
            if self.setting.pend is not None:
                dpres = (self.setting.pend - self.setting.press) / self.setting.steps
                # ase-v3.23.0 hase a bug in berendsen_npt _process_pressure
                driver.pressure = (
                    (self.setting.press + (start_step - 1) * dpres) * 1e5 * units.Pascal
                )
                driver.attach(
                    update_target_pressure, dyn=driver, dpres=dpres, interval=1
                )

            # override rng
            if hasattr(driver, "rng"):
                # Langevin needs this!
                self._print(f"MD Driver uses rng: {self.rng.bit_generator.state}")
                driver.rng = self.rng
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

        rng_state_fpath = ckpt_dir / "rng_state.json"
        if rng_state_fpath.exists():
            with open(ckpt_dir / "rng_state.json", "r") as fopen:
                rng_state = json.load(fopen)
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
        # To restart, velocities are always retained
        prev_ignore_atoms_velocities = self.setting.ignore_atoms_velocities
        if prev_wdir is None:  # start from the scratch
            start_step = 0
            rng_state = None

            curr_params = {}
            curr_params["random_seed"] = self.random_seed
            curr_params["init"] = self.setting.get_init_params()
            curr_params["run"] = self.setting.get_run_params()

            with open(self.directory / "params.json", "w") as fopen:
                json.dump(curr_params, fopen, indent=2)
        else:  # restart ...
            ckpt_wdir = self._find_latest_checkpoint(prev_wdir)
            atoms, rng_state = self._load_checkpoint(ckpt_wdir)
            write(self.directory / self.xyz_fname, atoms)
            start_step = atoms.info["step"]
            if hasattr(self.calc, "calcs"):
                for calc in self.calc.calcs:
                    if hasattr(calc, "_load_checkpoint"):
                        calc._load_checkpoint(ckpt_wdir, start_step=start_step)
            # --- update run_params in settings
            target_steps = self.setting.get_run_params(*args, **kwargs)["steps"]
            if target_steps > 0:
                if self.setting.task == "md":
                    steps = target_steps - start_step
                else:
                    # ase v3.22.1 opt will reset max_steps to steps in run
                    steps = target_steps
            assert steps > 0, "Steps should be greater than 0."
            kwargs.update(steps=steps)

            # To restart, velocities are always retained
            self.setting.ignore_atoms_velocities = False

        # - set calculator
        atoms.calc = self.calc

        # - set dynamics
        dynamics, run_params = self._create_dynamics(
            atoms, start_step=start_step, *args, **kwargs
        )
        dynamics.nsteps = start_step
        dynamics.max_steps = self.setting.steps
        if hasattr(dynamics, "rng") and rng_state is not None:
            dynamics.rng.bit_generator.state = rng_state

        # callback functions
        init_params = self.setting.get_init_params()

        # update atoms.info at the beginning of each step
        dynamics.attach(
            update_atoms_info,
            dyn=dynamics,
            atoms=atoms,
        )

        # HACK: add some observers to early stop the simulation
        #       make sure this must be added before save_trajectory
        #       as it adds `earlystop` in atoms.info and
        #       BE CAREFUL it changes some attributes affect convergence
        if self.setting.observers is not None:
            observers = []
            for ob_params in self.setting.observers:
                observers.append(create_an_observer(ob_params))
            for ob in observers:
                dynamics.attach(
                    monit_and_intervene,
                    interval=self.setting.dump_period,
                    atoms=atoms,
                    dynamics=dynamics,
                    observer=ob,
                    print_func=self._print,
                )

        # traj file not stores properties (energy, forces) properly
        dynamics.attach(
            save_checkpoint,
            interval=self.setting.ckpt_period,
            dyn=dynamics,
            atoms=atoms,
            wdir=self.directory,
            ckpt_number=self.setting.ckpt_number,
        )
        dynamics.attach(
            save_trajectory,
            interval=self.setting.dump_period,
            atoms=atoms,
            traj_fpath=self.directory / self.xyz_fname,
        )
        dynamics.attach(
            retrieve_and_save_deviation,
            interval=self.setting.dump_period,
            atoms=atoms,
            devi_fpath=self.directory / self.devi_fname,
        )

        # run simulation
        try:
            dynamics.run(**run_params)
        except Exception as e:
            self._debug(f"Exception of {self.__class__.__name__} is {e}.")
            self._debug(
                f"Exception of {self.__class__.__name__} is {traceback.format_exc()}."
            )

        # make sure the max_steps are the same as input even if
        # it is set by earlystop observer
        dynamics.max_steps = self.setting.steps

        # NOTE: check if the last frame is properly stored
        dump_period = self.setting.dump_period
        ckpt_period = self.setting.ckpt_period

        should_dump_last, should_ckpt_last = False, False
        # task min optimiser dumps every step to log but we control saved structures
        # by dump_period
        nsteps = atoms.info["step"] + 1
        if nsteps > 0 and (nsteps - 1) % dump_period != 0:
            should_dump_last = True
            if atoms.info.get(EARLYSTOP_KEY, False):
                should_dump_last = True
        if nsteps > 0 and (nsteps - 1) % ckpt_period != 0:
            should_ckpt_last = True

        if should_dump_last:
            self._debug("dump the last frame...")
            update_atoms_info(atoms, dynamics)
            save_trajectory(atoms, self.directory / self.xyz_fname)
            retrieve_and_save_deviation(atoms, self.directory / self.devi_fname)

        if should_ckpt_last:
            self._debug("ckpt the last frame...")
            save_checkpoint(
                dynamics,
                atoms,
                self.directory,
                ckpt_number=self.setting.ckpt_number,
            )

        # - Some interactive calculator needs kill processes after finishing,
        #   e.g. VaspInteractive...
        if hasattr(self.calc, "finalize"):
            self.calc.finalize()
        # To restart, velocities are always retained
        self.setting.ignore_atoms_velocities = prev_ignore_atoms_velocities

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

        # HACK: No need cut frames to the checkpoint like other driver backends
        #       as we always dump the last frame as a checkpoint
        # NOTE: Actually, if the simulation stopped in the middle, we do not have
        #       the checkpoint of the last frame thus the trajectories are not
        #       consecutive. So we concatenate trajectoies by atoms.info["step"]!!!

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
                    i * self.setting.timestep * self.setting.dump_period
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
