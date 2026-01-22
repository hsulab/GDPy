import copy
import dataclasses
import io
import os
import pathlib
import pickle
import tarfile
import traceback
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, all_changes
from ase.calculators.lammps import Prism, unitconvert
from ase.calculators.mixing import LinearCombinationCalculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses, atomic_numbers
from ase.io import read
from ase.io.lammpsdata import write_lammps_data

from gdpx import config
from gdpx.backend.lammps import add_model_deviation_to_atoms_info, parse_thermo_data_by_pattern
from gdpx.backend.plumed import (
    add_colvar_to_atoms_info,
    clap_plumed_file_by_number,
    clap_plumed_file_by_simulations,
    find_input_key_value,
    write_plumed_input_file,
)
from gdpx.group import evaluate_constraint_expression, evaluate_group_expression
from gdpx.utils.strconv import integers_to_string

from .driver import BaseDriver, Controller, DriverSetting
from .observer import create_an_observer


@dataclasses.dataclass(frozen=True)
class AseLammpsSettings:
    """File names."""

    inputstructure_filename: str = "stru.data"
    trajectory_filename: str = "traj.dump"
    input_fname: str = "in.lammps"
    # log_filename: str = "log.lammps"
    log_filename: str = "lmp.out"
    deviation_filename: str = "model_devi.out"
    prism_filename: str = "ase-prism.bindat"


#: Instance.
ASELMPCONFIG = AseLammpsSettings()


def parse_type_list(atoms):
    """Parse the type list based on input atoms."""
    # elements
    type_list = list(set(atoms.get_chemical_symbols()))
    type_list.sort()  # by alphabet

    return type_list


@dataclasses.dataclass
class CGMinimiser(Controller):
    name: str = "cg"

    def __post_init__(self):
        """"""
        maxstep = self.params.get("maxstep", 0.2)  # Ang
        maxstep = unitconvert.convert(maxstep, "distance", "metal", self.units)

        input_line = "min_style  cg\n"
        input_line += f"min_modify dmax {maxstep}"

        self.conv_params = dict(
            input_line=input_line,
        )

        return


@dataclasses.dataclass
class FireMinimizer(Controller):
    name: str = "fire"

    def __post_init__(self):
        """"""
        integrator = self.params.get("integrator", "verlet")
        tmax = self.params.get("tmax", 4)

        input_line = "min_style  fire\n"
        input_line += f"min_modify integrator {integrator} tmax {tmax}"

        self.conv_params = dict(
            input_line=input_line,
        )

        return


@dataclasses.dataclass
class MDController(Controller):
    #: Controller name.
    name: str = "md"

    #: Timestep in fs.
    timestep: float = 1.0

    #: Temperature in Kelvin.
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
        # convert timestep from fs to units
        self.timestep = unitconvert.convert(self.timestep, "time", "real", self.units)

        # convert temperature from Kelvin to units
        self.temperature = unitconvert.convert(self.temperature, "temperature", "real", self.units)

        if self.temperature_end is not None:
            self.temperature_end = unitconvert.convert(self.temperature_end, "temperature", "real", self.units)
        else:
            self.temperature_end = self.temperature

        if not (self.temperature > 0.0):
            raise Exception(f"MDController temperature `{self.temperature}` must be greater than 0.")

        assert self.temperature_end is not None
        if not (self.temperature_end > 0.0):
            raise Exception(f"MDController temperature_end `{self.temperature_end}` must be greater than 0.")

        # convert pressure from bar to units
        self.pressure = unitconvert.convert(self.pressure, "pressure", "metal", self.units)

        if self.pressure_end is not None:
            self.pressure_end = unitconvert.convert(self.pressure_end, "pressure", "metal", self.units)
        else:
            self.pressure_end = self.pressure

        input_line = ""
        if self.fix_com:
            input_line += "fix  fix_com {group} recenter INIT INIT INIT\n"
        input_line += f"\ntimestep {self.timestep}\n"
        self.conv_params = dict(input_line=input_line)

        return


@dataclasses.dataclass
class Verlet(MDController):
    name: str = "verlet"

    def __post_init__(self):
        """"""
        super().__post_init__()

        input_line = "fix {fix_id:>24s} {group} nve"
        self.conv_params["input_line"] = input_line + self.conv_params["input_line"]

        return


@dataclasses.dataclass
class LangevinThermostat(MDController):
    name: str = "langevin"

    def __post_init__(self):
        """"""
        super().__post_init__()

        friction = self.params.get("friction", 0.01)  # fs^-1
        assert friction is not None
        # Lammps uses the reciprocal of the friction coefficient
        # with the time unit.
        damp = unitconvert.convert(1.0 / friction, "time", "real", self.units)

        friction_seed = self.params.get("friction_seed", None)

        input_line = "fix {fix_id:>24s}0 {group} nve\n"
        input_line += "fix {fix_id:>24s}1 {group} langevin "
        input_line += f"{self.temperature} {self.temperature_end} {damp} "

        if friction_seed is not None:
            input_line += f"{friction_seed}"
        else:
            input_line += "{seed}"

        self.conv_params["input_line"] = input_line + self.conv_params["input_line"]

        return


@dataclasses.dataclass
class NoseHooverChainThermostat(MDController):
    name: str = "nose_hoover_chain"

    def __post_init__(self):
        """"""
        super().__post_init__()

        Tdamp = self.params.get(
            "Tdamp",
            unitconvert.convert(self.timestep * 100.0, "time", self.units, "real"),
        )  # fs, timestep have been converted previously
        assert Tdamp is not None
        Tdamp = unitconvert.convert(Tdamp, "time", "real", self.units)

        input_line = "fix {fix_id:>24s} {group} nvt temp "
        input_line += f"{self.temperature} {self.temperature_end} {Tdamp}"

        self.conv_params["input_line"] = input_line + self.conv_params["input_line"]

        return


@dataclasses.dataclass
class ParrinelloRahmanBarostat(MDController):
    name: str = "parrinello_rahman"

    def __post_init__(self):
        """"""
        super().__post_init__()

        Tdamp = self.params.get(
            "Tdamp",
            unitconvert.convert(self.timestep * 100.0, "time", self.units, "real"),
        )  # fs, timestep have been converted previously
        assert Tdamp is not None
        Tdamp = unitconvert.convert(Tdamp, "time", "real", self.units)

        Pdamp = self.params.get(
            "Pdamp",
            unitconvert.convert(self.timestep * 1000.0, "time", self.units, "real"),
        )  # fs, timestep have been converted previously
        assert Pdamp is not None
        Pdamp = unitconvert.convert(Pdamp, "time", "real", self.units)

        isotropic = self.params.get("isotropic", True)
        assert isotropic is not None
        isotropic = "iso" if isotropic else "aniso"

        input_line = "fix {fix_id:>24s} {group} npt temp "
        input_line += f"{self.temperature} {self.temperature_end} {Tdamp} "
        input_line += f"{isotropic} {self.pressure} {self.pressure_end} {Pdamp}"

        self.conv_params["input_line"] = input_line + self.conv_params["input_line"]

        return


controllers = dict(
    # min
    cg_min=CGMinimiser,
    fire_min=FireMinimizer,
    # nve
    verlet_nve=Verlet,
    # nvt
    langevin_nvt=LangevinThermostat,
    nose_hoover_chain_nvt=NoseHooverChainThermostat,
    # npt
    parrinello_rahman_npt=ParrinelloRahmanBarostat,
)

default_controllers = dict(
    min=FireMinimizer,
    nve=Verlet,
    nvt=LangevinThermostat,
    npt=ParrinelloRahmanBarostat,
)


@dataclasses.dataclass
class LmpDriverSetting(DriverSetting):
    #: LAMMPS units.
    units: str = "metal"

    #: MD ensemble.
    ensemble: str = "nve"

    #: Driver detailed controller setting.
    controller: dict = dataclasses.field(default_factory=dict)

    #: Whether fix com to the its initial position.
    fix_com: bool = False

    #: Whether initialise velocties internally by LAMMPS.
    use_lammps_vinit: bool = True

    #: Energy tolerance in minimisation, 1e-5 [eV].
    emax: Optional[float] = 0.0

    #: Force tolerance in minimisation, 5e-2 eV/Ang.
    fmax: Optional[float] = 0.05

    #: Neighbor list.
    neighbor: str = "2.0 bin"

    #: Neighbor list setting.
    neigh_modify: Optional[str] = "every 10 check yes"

    #: More custom LAMMPS fixes.
    extra_fix: list[str] = dataclasses.field(default_factory=list)

    #: PLUMED setting.
    plumed: Optional[str] = None

    def __post_init__(self):
        """"""
        if self.task == "min":
            self._internals.update(
                etol=self.emax,
                ftol=self.fmax,
            )

        if self.task == "md":
            self._internals.update(
                plumed=self.plumed,
            )

        # - special params
        self._internals.update(
            neighbor=self.neighbor,
            neigh_modify=self.neigh_modify,
            extra_fix=self.extra_fix,
        )

        return

    def get_simulation_inputs(self, random_seed: int, group: str = "mobile") -> list[str]:
        """Convert parameters into lammps input lines."""
        _init_params = {}

        if self.task == "min":
            suffix = self.task
        elif self.task == "md":
            suffix = self.ensemble
            _init_params.update(
                timestep=self.timestep,
                temperature=self.temp,
                temperature_end=self.tend if self.tend is not None else self.temp,
                pressure=self.press,
                pressure_end=self.pend if self.pend is not None else self.press,
                fix_com=self.fix_com,
            )
        else:
            suffix = self.task

        if self.controller:
            cont_cls_name = self.controller["name"] + "_" + suffix
            if cont_cls_name in controllers:
                cont_cls = controllers[cont_cls_name]
            else:
                raise RuntimeError(f"Unknown controller {cont_cls_name}.")
        else:
            cont_cls = default_controllers[suffix]

        _init_params.update(**self.controller)
        controller = cont_cls(units=self.units, **_init_params)

        # The input inline should be a f-string with placeholders that accept system-specific parameters.
        _init_placeholders = dict(
            fix_id="controller",
            group=group,
            seed=random_seed,  # langevin needs this
        )
        input_line = controller.conv_params["input_line"].format(**_init_placeholders)
        lines = [input_line]

        return lines

    def get_run_params(self, *args, **kwargs):
        """"""
        # convergence criteria
        fmax_ = kwargs.pop("fmax", self.fmax)
        emax_ = kwargs.pop("emax", self.emax)
        if emax_ is None:
            emax_ = 0.0
        if fmax_ is None:
            fmax_ = 0.0

        steps_ = kwargs.pop("steps", self.steps)

        run_params = dict(
            steps=steps_,
            constraint=kwargs.get("constraint", self.constraint),
            etol=emax_,
            ftol=fmax_,
        )

        # - add extra parameters
        run_params.update(**kwargs)

        return run_params


class LmpDriver(BaseDriver):
    """Use lammps to perform dynamics.

    Minimisation and/or molecular dynamics.

    """

    name = "lammps"

    default_task = "min"
    supported_tasks = ["min", "md"]

    #: Class for setting.
    setting_cls: type[DriverSetting] = LmpDriverSetting

    def __init__(self, calc, params: dict, directory="./", *args, **kwargs):
        """"""
        calc, params = self._canonicalise_calculator(calc=calc, params=params)
        params.update(units=calc.units)

        super().__init__(calc, params, directory=directory, *args, **kwargs)

        return

    def _canonicalise_calculator(self, calc, params: dict):
        """Canonicalise the calculator and its parameters.

        We check whether the calculator is a pure LAMMPS or a combination of LAMMPS and PLUMED.

        """
        # TODO: We should better move this to potential_manager.
        units = "metal"
        try:
            from ..potential.managers.plumed.calculators.plumed2 import Plumed

            new_calc, new_params = calc, params
            if isinstance(calc, LinearCombinationCalculator):
                ncalcs = len(calc.calcs)
                assert ncalcs == 2, "Number of calculators should be 2."
                if isinstance(calc.calcs[0], Lammps) and isinstance(calc.calcs[1], Plumed):
                    new_calc = calc.calcs[0]
                    new_params = copy.deepcopy(params)
                    new_params["plumed"] = "".join(calc.calcs[1].input)
                    units = calc.calcs[0].units
        except ImportError:
            new_calc = calc
            new_params = params
            units = calc.units
        new_params.update(units=units)

        return new_calc, new_params

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """"""
        verified = super()._verify_checkpoint(*args, **kwargs)
        if verified:
            if self.setting.ckpt_period >= self.setting.steps:
                # The computation may finish without saving any restart files.
                verified = self.read_convergence_from_logfile()
            else:
                checkpoints = list(self.directory.glob("restart.*"))
                self._debug(f"checkpoints: {checkpoints}")
                if not checkpoints:
                    verified = False
        else:
            ...

        return verified

    def _create_dynamics(self, atoms: Atoms, *args, **kwargs):
        """Convert parameters into lammps input lines."""
        lines = []
        if self.setting.task == "md":
            # Velocities by ASE may lose precision as
            # they are first written to data file and read by lammps then
            if self.setting.use_lammps_vinit:
                velocity_seed = self.setting.velocity_seed
                if velocity_seed is None:
                    velocity_seed = self.random_seed
                self._print(f"MD Driver's velocity_seed: {velocity_seed}")
                line = f"velocity        mobile create {self.setting.temp} {velocity_seed} dist gaussian "
                if self.setting.remove_translation:
                    line += "mom yes "
                if self.setting.remove_rotation:
                    line += "rot yes "
                if atoms.get_kinetic_energy() > 0.0:
                    if self.setting.ignore_atoms_velocities:
                        atoms.set_momenta(np.zeros(atoms.positions.shape))
                        lines.append(line)
                    else:
                        lines.append("# Use atoms' velocities.")
                else:
                    atoms.set_momenta(np.zeros(atoms.positions.shape))
                    lines.append(line)
            else:
                self._prepare_velocities(
                    atoms,
                    self.setting.velocity_seed,
                    self.setting.ignore_atoms_velocities,
                )
        else:
            ...

        dynamics = self.setting.get_simulation_inputs(random_seed=self.random_seed, group="mobile")
        lines.extend(dynamics)

        return lines

    def _irun(self, atoms: Atoms, ckpt_wdir=None, *args, **kwargs):
        """"""
        run_params = self.setting.get_init_params()
        run_params.update(**self.setting.get_run_params(**kwargs))

        prev_temperature, prev_pressure = self.setting.temp, self.setting.press

        is_continue = ckpt_wdir is not None
        if not is_continue:  # start from the scratch
            curr_temperature, curr_pressure = (
                self.setting.temp,
                self.setting.press,
            )
            finish_steps = 0  # For plumed
        else:
            checkpoints = sorted(
                list(ckpt_wdir.glob("restart.*")),
                key=lambda x: int(x.name.split(".")[1]),
            )
            self._debug(f"checkpoints to restart: {checkpoints}")
            target_steps = run_params["steps"]
            finish_steps = int(checkpoints[-1].name.split(".")[1])
            remain_steps = target_steps - finish_steps
            run_params.update(read_restart=str(checkpoints[-1].resolve()), steps=remain_steps)
            if self.setting.tend is not None:
                curr_temperature = (
                    self.setting.temp + (self.setting.tend - self.setting.temp) / target_steps * finish_steps
                )
            else:
                curr_temperature = self.setting.temp
            if self.setting.pend is not None:
                curr_pressure = (
                    self.setting.press + (self.setting.pend - self.setting.press) / target_steps * finish_steps
                )
            else:
                curr_pressure = self.setting.press

        # In case of temperature/pressure annealing
        self.setting.temp = curr_temperature
        self.setting.press = curr_pressure

        dynamics = self._create_dynamics(atoms, *args, **kwargs)

        if self.calc.plumed is not None:
            if is_continue:  # clean up COLVAR and HILLS
                assert finish_steps > 0
                required_num_lines = int(finish_steps / self.setting.dump_period)
                assert required_num_lines - finish_steps / self.setting.dump_period == 0, (
                    "The finished steps must be multiple of dump_period."
                )
                # Find files with names starting with COLVAR
                colvar_files = list(ckpt_wdir.glob("COLVAR*"))
                for fpath in colvar_files:
                    clap_plumed_file_by_number(
                        fpath,
                        self.directory / fpath.name,
                        required_num_lines,
                        0,
                    )
                # Find enhanced sampling-related files, HILLS or KERNELS
                hills_fpath = ckpt_wdir / "HILLS"
                if hills_fpath.exists():
                    clap_plumed_file_by_number(
                        hills_fpath,
                        self.directory / "HILLS",
                        required_num_lines,
                        0,
                    )
                kernels_fpath = ckpt_wdir / "KERNELS"
                if kernels_fpath.exists():
                    opes_sigma = find_input_key_value(
                        self.calc.plumed,
                        "OPES_METAD",
                        "SIGMA",
                    )
                    adaptive_sigma_stride = find_input_key_value(
                        self.calc.plumed,
                        "OPES_METAD",
                        "ADAPTIVE_SIGMA_STRIDE",
                    )
                    if opes_sigma is None or opes_sigma == "ADAPTIVE":
                        adaptive_sigma_stride = (
                            int(adaptive_sigma_stride) if adaptive_sigma_stride is not None else 10
                        )  # opes_metad use 10xpace to estimate sigmas
                    else:
                        assert adaptive_sigma_stride is None, (
                            "If SIGMA is initialised, ADAPTIVE_SIGMA_STRIDE should not be set."
                        )
                        adaptive_sigma_stride = 1
                    clap_plumed_file_by_number(
                        kernels_fpath,
                        self.directory / "KERNELS",
                        required_num_lines,
                        # TODO: check adaptive_sigma_stride
                        -adaptive_sigma_stride + 1,  # opes_metd use 10xpace to estimate sigmas
                    )
                # copy STATE if we have the exact state at the checkpoint
                state_fpath = ckpt_wdir / "STATE"
                if state_fpath.exists():
                    finished_time = finish_steps * self.setting.timestep / 1000.0  # in ps
                    clap_plumed_file_by_simulations(
                        state_fpath,
                        self.directory / "STATE",
                        finished_time,
                    )
            write_plumed_input_file(
                pathlib.Path(self.directory),
                copy.deepcopy(self.calc.plumed),
                dict(
                    dump_period=self.setting.dump_period,
                    ckpt_period=self.setting.ckpt_period,
                    temperature=curr_temperature,
                ),
                is_continue=is_continue,
            )

        self.setting.temp = prev_temperature
        self.setting.press = prev_pressure

        halt = ""
        if self.setting.observers is not None:
            observers = []
            for ob_params in self.setting.observers:
                observers.append(create_an_observer(ob_params))
            for i, ob in enumerate(observers):
                if hasattr(ob, "get_lammps_command"):
                    halt += ob.get_lammps_command(f"observer_{i:>02d}", "all", dump_period=self.setting.dump_period)

        # Update calculator parameters
        self.calc.set(
            task=self.setting.task,
            dump_period=self.setting.dump_period,
            ckpt_period=self.setting.ckpt_period,
            dynamics=dynamics,
            steps=run_params["steps"],
            constraint=run_params["constraint"],
            etol=run_params["etol"],
            ftol=run_params["ftol"],
            # misc
            read_restart=run_params.get("read_restart", None),
            extra_fix=run_params["extra_fix"],  # e.g. fixcm
            neighbor=run_params["neighbor"],
            neigh_modify=run_params["neigh_modify"],
            halt=halt,  # for earlystop
        )
        atoms.calc = self.calc

        # run simulation
        try:
            _ = atoms.get_forces()
        except Exception as e:
            self._debug(traceback.format_exc())
        finally:
            # restore some temporary parameters.
            ...

        return

    @staticmethod
    def _read_a_single_trajectory(
        wdir: pathlib.Path,
        mdir,
        units: str,
        archive_path: Optional[pathlib.Path] = None,
        print_func=config._print,
        debug_func=config._debug,
        *args,
        **kwargs,
    ):
        """"""
        # Get all file handles
        traj_io, log_io = None, None
        if archive_path is None:
            traj_io = open(wdir / ASELMPCONFIG.trajectory_filename, "r")
            log_io = open(wdir / ASELMPCONFIG.log_filename, "r")
            prism_file = wdir / ASELMPCONFIG.prism_filename
            if prism_file.exists():
                prism_io = open(prism_file, "rb")
            else:
                prism_io = None
            devi_path = wdir / (ASELMPCONFIG.deviation_filename)
            if devi_path.exists():
                devi_io = open(devi_path, "r")
            else:
                devi_io = None
            colvar_path = wdir / "COLVAR"
            if colvar_path.exists():
                colvar_io = open(colvar_path, "r")
            else:
                colvar_io = None
        else:
            rpath = wdir.relative_to(mdir.parent)
            traj_tarname = str(rpath / ASELMPCONFIG.trajectory_filename)
            prism_tarname = str(rpath / ASELMPCONFIG.prism_filename)
            log_tarname = str(rpath / ASELMPCONFIG.log_filename)
            devi_tarname = str(rpath / ASELMPCONFIG.deviation_filename)
            colvar_tarname = str(rpath / "COLVAR")
            prism_io, devi_io, colvar_io = None, None, None
            with tarfile.open(archive_path, "r:gz") as tar:
                for tarinfo in tar:
                    if tarinfo.name.startswith(wdir.name):
                        if tarinfo.name == traj_tarname:
                            traj_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                        elif tarinfo.name == prism_tarname:
                            prism_io = io.BytesIO(tar.extractfile(tarinfo.name).read())
                        elif tarinfo.name == log_tarname:
                            log_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                        elif tarinfo.name == devi_tarname:
                            devi_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                        elif tarinfo.name == colvar_tarname:
                            colvar_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                        else:
                            ...
                    else:
                        continue
                else:  # TODO: if not find target traj?
                    ...

        # Must have (traj, log), others (prism, devi, colvar) are optional
        assert traj_io is not None
        assert log_io is not None

        # Get timesteps
        timesteps = []
        while True:
            line = traj_io.readline()
            if "TIMESTEP" in line:
                timesteps.append(int(traj_io.readline().strip()))
            if not line:
                break
        traj_io.seek(0)

        # - read structure trajectory
        if prism_io is not None:
            prismobj = pickle.load(prism_io)
        else:
            prismobj = None

        curr_traj_frames_ = read(
            traj_io,
            index=":",
            format="lammps-dump-text",
            prismobj=prismobj,
            units=units,
        )
        nframes_traj = len(curr_traj_frames_)
        timesteps = timesteps[:nframes_traj]  # avoid incomplete structure

        # Read thermo data
        thermo_dict = parse_thermo_data_by_pattern(log_io.readlines(), print_func=print_func, debug_func=debug_func)

        # The last frame would not be dumpped if timestep not equals multiple*dump_period
        # or if there were any error,
        pot_energies = [unitconvert.convert(p, "energy", units, "ASE") for p in thermo_dict["PotEng"]]
        nframes_thermo = len(pot_energies)
        nframes = min([nframes_traj, nframes_thermo])
        debug_func(f"nframes in lammps: {nframes} traj {nframes_traj} thermo {nframes_thermo}")

        curr_traj_frames, curr_energies = [], []
        for i, t in enumerate(timesteps):
            if t in thermo_dict["Step"]:
                curr_atoms = curr_traj_frames_[i]
                curr_atoms.info["step"] = t
                curr_traj_frames.append(curr_atoms)
                curr_energies.append(pot_energies[thermo_dict["Step"].tolist().index(t)])

        for pot_eng, atoms in zip(curr_energies, curr_traj_frames):
            forces = atoms.get_forces()
            # forces have already been converted in ase read, so are velocities
            sp_calc = SinglePointCalculator(atoms, energy=pot_eng, forces=forces)
            atoms.calc = sp_calc

        # Check model_devi.out if any, frames may not have deviations as the simulation can stop in the middle
        _ = (
            add_model_deviation_to_atoms_info(
                devi_io,
                curr_traj_frames,
                units=units,
            )
            if devi_io is not None
            else None
        )

        # Add COLVAR if any, frames may not have colvars as the simulation can stop in the middle
        _ = (
            add_colvar_to_atoms_info(
                colvar_io,
                curr_traj_frames,
                ignored_columns=["time"],
            )
            if colvar_io is not None
            else None
        )

        # Close all file handles
        traj_io.close()
        log_io.close()
        if prism_io is not None:
            prism_io.close()
        if devi_io is not None:
            devi_io.close()
        if colvar_io is not None:
            colvar_io.close()

        return curr_traj_frames

    def read_trajectory(
        self,
        type_list=None,
        archive_path: Optional[pathlib.Path] = None,
        *args,
        **kwargs,
    ) -> list[Atoms]:
        """Read trajectory in the current working directory."""
        if type_list is not None:
            self.calc.type_list = type_list
        curr_units = self.calc.units

        traj_frames = self._aggregate_trajectories(
            units=curr_units,
            mdir=self.directory,
            check_energy=True,
            archive_path=archive_path,
        )

        return traj_frames

    def read_convergence_from_logfile(self, *args, **kwargs):
        """"""
        converged = False
        log_fpath = self.directory / ASELMPCONFIG.log_filename
        if log_fpath.exists() and log_fpath.stat().st_size != 0:
            with open(log_fpath, "r") as fopen:
                lines = fopen.readlines()
            end_line = lines[-1].strip()
            if end_line.startswith("Total wall time:"):
                converged = True
            elif end_line.startswith("Last command: run"):
                with open(self.directory / "EARLYSTOP", "w") as fopen:
                    fopen.write("")
                converged = True
            else:
                self._print(f"LAMMPS ENDLINE: {end_line}")
        else:
            ...

        return converged


class Lammps(FileIOCalculator):
    #: Calculator name.
    name: str = "Lammps"

    #: Implemented properties.
    implemented_properties: list[str] = ["energy", "forces", "stress"]

    #: Default calculator parameters, NOTE which have ase units.
    default_parameters: dict = dict(
        # ase prepared parameters
        task="min",
        dump_period=1,
        ckpt_period=100,
        dynamics="",
        steps=0,
        constraint=None,  # index of atoms, start from 0
        etol=0.0,
        ftol=0.05,
        is_classic=False,
        # lmp params ---
        read_restart=None,
        units="metal",
        atom_style="atomic",
        atom_modify=None,
        processors=None,
        # boundary = "p p p",
        newton=None,
        pair_style=None,
        pair_coeff=None,
        pair_modify=None,
        kspace_style=None,
        neighbor="2.0 bin",
        neigh_modify="every 10 check yes",
        mass="* 1.0",
        # extra fix
        halt="",
        extra_fix=[],
        # externals
        plumed=None,
    )

    #: Symbol to integer.
    type_list: Optional[list[str]] = None

    #: Cached trajectory of the previous simulation.
    cached_traj_frames: Optional[list[Atoms]] = None

    def __init__(self, command="lmp", label=name, **kwargs):
        """"""
        FileIOCalculator.__init__(self, command=command, label=label, **kwargs)

        # complete command
        command_ = self.profile.command
        if "-in" in command_:
            ...
        else:
            command_ += f" -in in.lammps 2>&1 > lmp.out"
        self.profile.command = command_

        # - check potential
        assert self.pair_style is not None, "pair_style is not set."

        return

    def __getattr__(self, key):
        """Corresponding getattribute-function."""
        if key != "parameters" and key in self.parameters:
            return self.parameters[key]
        return object.__getattribute__(self, key)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """Run calculation."""
        # TODO: should use user-custom type_list from potential manager
        #       move this part to driver?
        self.type_list = parse_type_list(atoms)

        # init for creating the directory
        FileIOCalculator.calculate(self, atoms, properties, system_changes)

        return

    def write_input(self, atoms, properties=None, system_changes=None) -> None:
        """Write input file and input structure."""
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # Check velocities
        write_velocities = False
        if atoms.get_kinetic_energy() > 0.0:
            write_velocities = True

        # Write structure
        prismobj = Prism(atoms.get_cell())
        prism_file = os.path.join(self.directory, ASELMPCONFIG.prism_filename)
        with open(prism_file, "wb") as fopen:
            pickle.dump(prismobj, fopen)
        stru_data = os.path.join(self.directory, ASELMPCONFIG.inputstructure_filename)

        assert self.type_list is not None
        write_lammps_data(
            stru_data,
            atoms,
            specorder=self.type_list,
            force_skew=False,
            prismobj=prismobj,
            velocities=write_velocities,
            units=self.units,
            atom_style=self.atom_style,
        )

        # Write input
        self._write_input(atoms, prismobj)

        return

    def _is_finished(self):
        """Check whether the simulation finished or failed.

        Return wall time if the simulation finished.

        """

        is_finished, end_info = False, "not finished"
        log_filepath = pathlib.Path(os.path.join(self.directory, ASELMPCONFIG.log_filename))

        if log_filepath.exists():
            ERR_FLAG = "ERROR: "
            END_FLAG = "Total wall time:"
            with open(log_filepath, "r") as fopen:
                lines = fopen.readlines()

            for line in lines:
                if line.strip().startswith(ERR_FLAG):
                    is_finished = True
                    end_info = " ".join(line.strip().split()[1:])
                    break
                if line.strip().startswith(END_FLAG):
                    is_finished = True
                    end_info = " ".join(line.strip().split()[1:])
                    break
            else:
                is_finished = False
        else:
            is_finished = False

        return is_finished, end_info

    def read_results(self):
        """ASE read results."""
        # obtain results
        self.results = {}

        # - Be careful with UNITS
        # read forces from dump file
        curr_wdir = pathlib.Path(self.directory)
        self.cached_traj_frames = LmpDriver._read_a_single_trajectory(
            mdir=curr_wdir,
            wdir=curr_wdir,
            units=self.units,
            print_func=lambda _: "",
            debug_func=lambda _: "",
        )
        converged_frame = self.cached_traj_frames[-1]

        self.results["forces"] = converged_frame.get_forces().copy()
        self.results["energy"] = converged_frame.get_potential_energy()

        # - add deviation info
        for k, v in converged_frame.info.items():
            if "devi" in k:
                self.results[k] = v

        return

    def _write_input(self, atoms, prismobj) -> None:
        """Write input file in.lammps"""
        # Global settings
        content = f"restart         {self.ckpt_period}  restart.*.data\n\n"
        content += "units           %s\n" % self.units
        content += "atom_style      %s\n" % self.atom_style
        if self.atom_modify is not None:
            content += f"atom_modify {self.atom_modify}\n"

        # Parallel settings
        if self.processors is not None:
            content += f"processors  {self.processors}\n"  # if 2D simulation

        # Simulation system
        pbc = atoms.get_pbc()
        if "boundary" in self.parameters:
            content += "boundary {0} \n".format(self.parameters["boundary"])
        else:
            content += "boundary {0} {1} {2} \n".format(
                *tuple("fp"[int(x)] for x in pbc)  # sometimes s failed to wrap all atoms
            )
        content += "\n"
        if self.newton:
            content += f"newton  {self.newton}\n"
        if self.read_restart is None:
            content += "read_data	    %s\n" % ASELMPCONFIG.inputstructure_filename
        else:
            content += f"read_restart    {self.read_restart}\n"

        if prismobj.is_skewed():
            content += "box             tilt large\n"
            content += "change_box      all triclinic\n"

        # Particle masses
        assert self.type_list is not None
        mass_line = "".join(
            "mass %d %f\n" % (idx + 1, atomic_masses[atomic_numbers[elem]]) for idx, elem in enumerate(self.type_list)
        )
        content += mass_line
        content += "\n"

        # Particle charges
        if self.atom_style == "charge" and self.type_charges:
            for itype, charge in enumerate(self.type_charges):
                content += f"set type {itype + 1} charge {charge}\n"
            content += "\n"

        # Pair styles
        type_list_str = " ".join(self.type_list)
        pair_dicts = dict(
            pair_style=dict(
                type_list=type_list_str,
                out_freq=self.dump_period,  # deepmd
            ),
            pair_coeff=dict(
                type_list=type_list_str,
            ),
        )
        if self.is_classic:
            # assert (
            #     self.atom_style == "charge"
            # ), "For now, classic potentials need charge information."
            content += f"pair_style  {self.pair_style}\n"
            if isinstance(self.pair_coeff, str):
                pair_coeff = [self.pair_coeff]
            else:
                pair_coeff = self.pair_coeff
            for coeff in pair_coeff:
                content += f"pair_coeff  {coeff}\n"
        else:
            # Some potentials need system-specific information such as type_list,
            # so we treat them separately.
            potential = self.pair_style.strip().split()[0]
            if potential == "reax/c":
                assert self.atom_style == "charge", "reax/c should have charge atom_style"
                content += f"pair_style  {self.pair_style}\n"
                content += f"pair_coeff  {self.pair_coeff} {type_list_str}\n"
                content += "fix             reaxqeq all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
            elif potential == "eann":
                pot_data = self.pair_style.strip().split()[1:]
                endp = len(pot_data)
                for ip, p in enumerate(pot_data):
                    if p == "out_freq":
                        endp = ip
                        break
                pot_data = pot_data[:endp]
                if len(pot_data) > 1:
                    pair_style = "eann {} out_freq {}".format(" ".join(pot_data), self.dump_period)
                else:
                    pair_style = "eann {}".format(" ".join(pot_data))
                content += "pair_style  {}\n".format(pair_style)
                # NOTE: make out_freq consistent with dump_period
                if self.pair_coeff is None:
                    pair_coeff = "double * *"
                else:
                    pair_coeff = self.pair_coeff
                content += f"pair_coeff	{pair_coeff} {type_list_str}\n"
            else:
                # PotentialManager should give f-strings for pair_style and pair_coeff that
                # system-specific information can be applied below.
                content += "pair_style  " + self.pair_style.format(**pair_dicts["pair_style"]) + "\n"
                content += "pair_coeff  " + self.pair_coeff.format(**pair_dicts["pair_coeff"]) + "\n"

        if self.pair_modify is not None:
            content += f"pair_modify {self.pair_modify}\n"

        content += "\n"

        # TODO: kspace should be set with pair at the same time
        if self.kspace_style is not None:
            content += f"kspace_style  {self.kspace_style}\n\n"

        # Neighbour list settings
        content += f"neighbor        {self.neighbor}\n"
        if self.neigh_modify:
            content += f"neigh_modify    {self.neigh_modify}\n"
        content += "\n"

        # Whether fix atoms
        mobile_indices, frozen_indices = evaluate_constraint_expression(atoms, self.constraint)
        if mobile_indices:  # Sometimes all atoms are fixed.
            mobile_text = integers_to_string(mobile_indices, inp_convention="ase")
            content += f"group mobile id {mobile_text}\n"
            content += "\n"
        if frozen_indices:  # Sometimes no atoms are fixed.
            frozen_text = integers_to_string(frozen_indices, inp_convention="ase")
            content += f"group frozen id {frozen_text}\n"
            content += "fix cons frozen setforce 0.0 0.0 0.0\n"
        content += "\n"

        # Thermodynamics outputs
        # TODO: use more flexible notations
        if self.task == "min":
            content += "thermo_style    custom step pe ke etotal temp press vol fmax fnorm\n"
        elif self.task == "md":
            content += "compute mobileTemp mobile temp\n"
            content += "thermo_style    custom step c_mobileTemp pe ke etotal press vol lx ly lz xy xz yz\n"
        else:
            pass
        content += f"thermo         {self.dump_period}\n"
        content += "thermo_modify   flush yes\n"

        # total energy is not stored in dump so we need read from log.lammps
        if self.atom_style == "atomic":
            content += "dump		1 all custom {} {} id type element x y z fx fy fz vx vy vz\n".format(
                self.dump_period, ASELMPCONFIG.trajectory_filename
            )
        elif self.atom_style == "charge":
            content += "dump		1 all custom {} {} id type element q x y z fx fy fz vx vy vz\n".format(
                self.dump_period, ASELMPCONFIG.trajectory_filename
            )
        else:
            ...
        assert self.type_list is not None
        content += f"dump_modify 1 element {' '.join(self.type_list)} flush yes\n"
        content += "\n"

        # Add additional fixes
        for i, fix_info in enumerate(self.extra_fix):
            if isinstance(fix_info, str):  # fix ID command
                content += "{:<24s}  {:<24s}  {:<s}\n".format("fix", f"extra{i}", fix_info)
            else:  # fix ID group-ID command
                group_indices = evaluate_group_expression(atoms, fix_info[0])
                group_text = integers_to_string(
                    group_indices, inp_convention="ase"
                )  # ase-index-list -> lmp-index-text
                content += "{:<24s}  {:<24s}  id  {:<s}  \n".format("group", f"extra_group_{i}", group_text)
                content += "{:<24s}  {:<24s}  {:<s}  {:<s}\n".format(
                    "fix", f"extra{i}", f"extra_group_{i}", fix_info[1]
                )
        content += "\n"

        # Add earlystop fixes
        if self.halt:
            content += self.halt
            content += "\n"

        # Simulation tasks
        if self.task == "min":
            content += "\n".join(self.dynamics) + "\n"

            content += "minimize        {:f} {:f} {:d} {:d}\n".format(
                unitconvert.convert(self.etol, "energy", "ASE", self.units),
                unitconvert.convert(self.ftol, "force", "ASE", self.units),
                self.steps,
                2 * self.steps,
            )
        elif self.task == "md":
            if self.read_restart is not None:
                # pop up velocity line
                self.dynamics[0] = "#  use velocities in restart"
            content += "\n".join(self.dynamics) + "\n"

            if self.plumed is not None:
                content += "fix             metad all plumed plumedfile plumed.inp outfile plumed.out\n"
            content += f"run             {self.steps}\n"
        else:
            # TODO: NEB?
            ...

        # Write the input file
        in_file = os.path.join(self.directory, ASELMPCONFIG.input_fname)
        with open(in_file, "w") as fopen:
            fopen.write(content)

        return
