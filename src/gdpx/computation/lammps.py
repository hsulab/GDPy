#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses
import io
import itertools
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

from .. import config
from ..backend.lammps import parse_thermo_data_by_pattern
from ..builder.constraints import convert_indices, parse_constraint_info
from ..builder.group import create_a_group
from .driver import AbstractDriver, Controller, DriverSetting


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
class FireMinimizer(Controller):

    name: str = "fire"

    def __post_init__(self):
        """"""
        self.conv_params = dict(
            min_style="fire",
            min_modify=self.params.get("min_modify", "integrator verlet tmax 4"),
        )

        return


@dataclasses.dataclass
class LangevinThermostat(Controller):

    name: str = "langevin"

    def __post_init__(self):
        """"""
        friction = self.params.get("friction", 0.01)  # fs^-1
        assert friction is not None

        friction_seed = self.params.get("friction_seed", None)

        self.conv_params = dict(
            damp=unitconvert.convert(1.0 / friction, "time", "real", self.units)
        )
        if friction_seed is not None:
            self.conv_params.update(seed=friction_seed)

        return


@dataclasses.dataclass
class NoseHooverChainThermostat(Controller):

    name: str = "nose_hoover_chain"

    def __post_init__(self):
        """"""
        Tdamp = self.params.get("Tdamp", 100.0)
        assert Tdamp is not None

        self.conv_params = dict(
            Tdamp=unitconvert.convert(Tdamp, "time", "real", self.units)
        )

        return


@dataclasses.dataclass
class ParrinelloRahmanBarostat(Controller):

    name: str = "parrinello_rahman"

    def __post_init__(self):
        """"""
        Tdamp = self.params.get("Tdamp", 100.0)
        assert Tdamp is not None

        Pdamp = self.params.get("Pdamp", 100.0)
        assert Pdamp is not None

        isotropic = self.params.get("isotropic", True)
        assert isotropic is not None

        self.conv_params = dict(
            Tdamp=unitconvert.convert(Tdamp, "time", "real", self.units),
            Pdamp=unitconvert.convert(Pdamp, "time", "real", self.units),
            isotropic="iso" if isotropic else "aniso",
        )

        return


controllers = dict(
    # nvt
    langevin_nvt=LangevinThermostat,
    nose_hoover_chain_nvt=NoseHooverChainThermostat,
    parrinello_rahman_npt=ParrinelloRahmanBarostat,
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
    use_lmpvel: bool = True

    #: Energy tolerance in minimisation, 1e-5 [eV].
    emax: Optional[float] = 0.0

    #: Force tolerance in minimisation, 5e-2 eV/Ang.
    fmax: Optional[float] = 0.05

    #: Neighbor list.
    neighbor: str = "0.0 bin"

    #: Neighbor list setting.
    neigh_modify: Optional[str] = None

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

    def get_minimisation_inputs(self, random_seed, group: str = "mobile") -> list[str]:
        """"""
        """Convert parameters into lammps input lines."""
        MIN_FIX_ID: str = "controller"
        _init_min_params = dict(
            fix_id=MIN_FIX_ID,
            group=group,
        )
        if self.controller:
            min_cls_name = self.controller["name"] + "_min"
            min_cls = controllers[min_cls_name]
        else:
            min_cls = FireMinimizer

        minimiser = min_cls(units=self.units, **self.controller)
        _init_min_params.update(**minimiser.conv_params)

        if minimiser.name == "fire":
            min_line = "min_style  {min_style}\nmin_modify {min_modify}".format(
                **_init_min_params
            )
        else:
            raise RuntimeError(f"Unknown minimiser {minimiser}.")

        lines = [min_line]

        return lines

    def get_molecular_dynamics_inputs(
        self, random_seed, group: str = "mobile"
    ) -> list[str]:
        """Convert parameters into lammps input lines."""
        MD_FIX_ID: str = "controller"
        _init_md_params = dict(
            fix_id=MD_FIX_ID,
            group=group,
            timestep=unitconvert.convert(self.timestep, "time", "real", self.units),
        )

        if self.ensemble == "nve":
            lines = ["fix {fix_id:>24s} {group} nve".format(**_init_md_params)]
        elif self.ensemble == "nvt":
            _init_md_params.update(
                Tstart=self.temp,
                Tstop=self.tend if self.tend else self.temp,
            )
            if self.controller:
                thermo_cls_name = self.controller["name"] + "_" + self.ensemble
                thermo_cls = controllers[thermo_cls_name]
            else:
                thermo_cls = LangevinThermostat
            thermostat = thermo_cls(units=self.units, **self.controller)
            if thermostat.name == "langevin":
                _init_md_params.update(
                    seed=random_seed,
                )
                _init_md_params.update(**thermostat.conv_params)
                thermo_line = "fix {fix_id:>24s}0 {group} nve\n".format(
                    **_init_md_params
                )
                thermo_line += "fix {fix_id:>24s}1 {group} langevin {Tstart} {Tstop} {damp} {seed}".format(
                    **_init_md_params
                )
            elif thermostat.name == "nose_hoover_chain":
                _init_md_params.update(**thermostat.conv_params)
                thermo_line = "fix {fix_id:>24s} {group} nvt temp {Tstart} {Tstop} {Tdamp}".format(
                    **_init_md_params
                )
            else:
                raise RuntimeError(f"Unknown thermostat {thermostat}.")
            lines = [thermo_line]
        elif self.ensemble == "npt":
            _init_md_params.update(
                Tstart=self.temp,
                Tstop=self.tend if self.tend else self.temp,
                Pstart=self.press,
                Pstop=self.pend if self.pend else self.press,
            )
            if self.controller:
                baro_cls_name = self.controller["name"] + "_" + self.ensemble
                baro_cls = controllers[baro_cls_name]
            else:
                baro_cls = ParrinelloRahmanBarostat
            barostat = baro_cls(units=self.units, **self.controller)
            if barostat.name == "parrinello_rahman":
                _init_md_params.update(**barostat.conv_params)
                baro_line = "fix {fix_id:>24s} {group} npt temp {Tstart} {Tstop} {Tdamp} {isotropic} {Pstart} {Pstop} {Pdamp}".format(
                    **_init_md_params
                )
            else:
                raise RuntimeError(f"Unknown barostat {barostat}.")
            lines = [baro_line]
        else:
            raise RuntimeError(f"Unknown ensemble {self.ensemble}.")

        if self.fix_com:
            com_line = "fix  fix_com {group} recenter INIT INIT INIT".format(
                **_init_md_params
            )
            lines.append(com_line)

        lines.append(f"timestep {_init_md_params['timestep']}")

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


class LmpDriver(AbstractDriver):
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
                if isinstance(calc.calcs[0], Lammps) and isinstance(
                    calc.calcs[1], Plumed
                ):
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
        if self.setting.task == "min":
            dynamics = self.setting.get_minimisation_inputs(
                random_seed=self.random_seed
            )
            lines.extend(dynamics)
        else:  # assume md
            # NOTE: Velocities by ASE may lose precision as
            #       they are first written to data file and read by lammps then
            if self.setting.use_lmpvel:
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
            dynamics = self.setting.get_molecular_dynamics_inputs(
                random_seed=self.random_seed
            )
            lines.extend(dynamics)

        return lines

    def _irun(self, atoms: Atoms, ckpt_wdir=None, *args, **kwargs):
        """"""
        run_params = self.setting.get_init_params()
        run_params.update(**self.setting.get_run_params(**kwargs))

        prev_temperature, prev_pressure = self.setting.temp, self.setting.press
        if ckpt_wdir is None:  # start from the scratch
            curr_temperature, curr_pressure = self.setting.temp, self.setting.press
        else:
            checkpoints = sorted(
                list(ckpt_wdir.glob("restart.*")),
                key=lambda x: int(x.name.split(".")[1]),
            )
            self._debug(f"checkpoints to restart: {checkpoints}")
            target_steps = run_params["steps"]
            finish_steps = int(checkpoints[-1].name.split(".")[1])
            remain_steps = target_steps - finish_steps
            run_params.update(
                read_restart=str(checkpoints[-1].resolve()), steps=remain_steps
            )
            # shutil.move(
            #     checkpoints[-1].parent / "traj.dump", self.directory / "traj.dump"
            # )
            if self.setting.tend is not None:
                curr_temperature = (
                    self.setting.temp
                    + (self.setting.tend - self.setting.temp)
                    / target_steps
                    * finish_steps
                )
            else:
                curr_temperature = self.setting.temp
            if self.setting.pend is not None:
                curr_pressure = (
                    self.setting.press
                    + (self.setting.pend - self.setting.press)
                    / target_steps
                    * finish_steps
                )
            else:
                curr_pressure = self.setting.press

        # In case of temperature/pressure annealing
        self.setting.temp = curr_temperature
        self.setting.press = curr_pressure

        dynamics = self._create_dynamics(atoms, *args, **kwargs)

        self.setting.temp = prev_temperature
        self.setting.press = prev_pressure

        # - check constraint
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
        # - get FileIO
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
                            traj_io = io.StringIO(
                                tar.extractfile(tarinfo.name).read().decode()
                            )
                        elif tarinfo.name == prism_tarname:
                            prism_io = io.BytesIO(tar.extractfile(tarinfo.name).read())
                        elif tarinfo.name == log_tarname:
                            log_io = io.StringIO(
                                tar.extractfile(tarinfo.name).read().decode()
                            )
                        elif tarinfo.name == devi_tarname:
                            devi_io = io.StringIO(
                                tar.extractfile(tarinfo.name).read().decode()
                            )
                        elif tarinfo.name == colvar_tarname:
                            colvar_io = io.StringIO(
                                tar.extractfile(tarinfo.name).read().decode()
                            )
                        else:
                            ...
                    else:
                        continue
                else:  # TODO: if not find target traj?
                    ...

        # - read timesteps
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

        # - read thermo data
        thermo_dict = parse_thermo_data_by_pattern(
            log_io.readlines(), print_func=print_func, debug_func=debug_func
        )

        # NOTE: last frame would not be dumpped if timestep not equals multiple*dump_period
        #       if there were any error,
        pot_energies = [
            unitconvert.convert(p, "energy", units, "ASE")
            for p in thermo_dict["PotEng"]
        ]
        nframes_thermo = len(pot_energies)
        nframes = min([nframes_traj, nframes_thermo])
        debug_func(
            f"nframes in lammps: {nframes} traj {nframes_traj} thermo {nframes_thermo}"
        )

        # NOTE: check whether steps in thermo and traj are consistent
        # pot_energies = pot_energies[:nframes]
        # curr_traj_frames = curr_traj_frames[:nframes]
        # assert len(pot_energies) == len(curr_traj_frames), f"Number of pot energies and frames are inconsistent at {str(wdir)}."

        curr_traj_frames, curr_energies = [], []
        for i, t in enumerate(timesteps):
            if t in thermo_dict["Step"]:
                curr_atoms = curr_traj_frames_[i]
                curr_atoms.info["step"] = t
                curr_traj_frames.append(curr_atoms)
                curr_energies.append(
                    pot_energies[thermo_dict["Step"].tolist().index(t)]
                )

        for pot_eng, atoms in zip(curr_energies, curr_traj_frames):
            forces = atoms.get_forces()
            # NOTE: forces have already been converted in ase read, so velocities are
            sp_calc = SinglePointCalculator(atoms, energy=pot_eng, forces=forces)
            atoms.calc = sp_calc

        # - check model_devi.out
        # TODO: convert units?
        if devi_io is not None:
            lines = devi_io.readlines()
            if "#" in lines[0]:  # the first file
                dkeys = ("".join([x for x in lines[0] if x != "#"])).strip().split()
                dkeys = [x.strip() for x in dkeys][1:]
            else:
                ...
            devi_io.seek(0)
            data = np.loadtxt(devi_io, dtype=float)
            ncols = data.shape[-1]
            data = data.reshape(-1, ncols)
            # NOTE: For some minimisers, dp gives several deviations as
            #       multiple force evluations are performed in one step.
            #       Thus, we only take the last occurance of the deviation in each step.
            step_indices = []
            steps = data[:, 0].astype(np.int32).tolist()
            for k, v in itertools.groupby(enumerate(steps), key=lambda x: x[1]):
                v = sorted(v, key=lambda x: x[0])
                step_indices.append(v[-1][0])
            data = data.transpose()[1:, step_indices[:nframes]]

            for i, atoms in enumerate(curr_traj_frames):
                for j, k in enumerate(dkeys):
                    try:
                        atoms.info[k] = data[j, i]
                    except IndexError:
                        # NOTE: Some potentials donot print last frames of min
                        #       for example, lammps
                        atoms.info[k] = 0.0
        else:
            ...

        # - check COLVAR
        if colvar_io is not None:
            # - read latest COLVAR Files
            names = colvar_io.readline().split()[2:]
            colvar_io.seek(0)
            colvars = np.loadtxt(colvar_io)
            # print("colvars: ", colvars.shape)
            curr_colvars = colvars[-nframes_traj:, :]
            for i, atoms in enumerate(curr_traj_frames):
                for k, v in zip(names, curr_colvars[i, :]):
                    atoms.info[k] = v

        # - Close IO
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
        archive_path: pathlib.Path = None,
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
        if log_fpath:
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

        # - check velocities
        write_velocities = False
        if atoms.get_kinetic_energy() > 0.0:
            write_velocities = True

        # write structure
        prismobj = Prism(atoms.get_cell())  # TODO: nonpbc?
        prism_file = os.path.join(self.directory, ASELMPCONFIG.prism_filename)
        with open(prism_file, "wb") as fopen:
            pickle.dump(prismobj, fopen)
        stru_data = os.path.join(self.directory, ASELMPCONFIG.inputstructure_filename)
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

        # write input
        self._write_input(atoms, prismobj)

        return

    def _is_finished(self):
        """Check whether the simulation finished or failed.

        Return wall time if the simulation finished.

        """

        is_finished, end_info = False, "not finished"
        log_filepath = pathlib.Path(
            os.path.join(self.directory, ASELMPCONFIG.log_filename)
        )

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
        # - write in.lammps
        content = f"restart         {self.ckpt_period}  restart.*.data\n\n"
        content += "units           %s\n" % self.units
        content += "atom_style      %s\n" % self.atom_style

        # - mpi settings
        if self.processors is not None:
            content += "processors {}\n".format(self.processors)  # if 2D simulation

        # - simulation box
        pbc = atoms.get_pbc()
        if "boundary" in self.parameters:
            content += "boundary {0} \n".format(self.parameters["boundary"])
        else:
            content += "boundary {0} {1} {2} \n".format(
                *tuple(
                    "fp"[int(x)] for x in pbc
                )  # sometimes s failed to wrap all atoms
            )
        content += "\n"
        if self.newton:
            content += "newton {}\n".format(self.newton)
        if self.read_restart is None:
            content += "read_data	    %s\n" % ASELMPCONFIG.inputstructure_filename
        else:
            content += f"read_restart    {self.read_restart}\n"

        if prismobj.is_skewed():
            content += "box             tilt large\n"
            content += "change_box      all triclinic\n"

        # particle masses
        mass_line = "".join(
            "mass %d %f\n" % (idx + 1, atomic_masses[atomic_numbers[elem]])
            for idx, elem in enumerate(self.type_list)
        )
        content += mass_line
        content += "\n"

        # particle charges
        if self.atom_style == "charge" and self.type_charges:
            for itype, charge in enumerate(self.type_charges):
                content += f"set type {itype+1} charge {charge}\n"
            content += "\n"

        # pair, MLIP specific settings
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
            potential = self.pair_style.strip().split()[0]
            if potential == "reax/c":
                assert (
                    self.atom_style == "charge"
                ), "reax/c should have charge atom_style"
                content += "pair_style  {}\n".format(self.pair_style)
                content += "pair_coeff {} {}\n".format(
                    self.pair_coeff, " ".join(self.type_list)
                )
                content += (
                    "fix             reaxqeq all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
                )
            elif potential == "eann":
                pot_data = self.pair_style.strip().split()[1:]
                endp = len(pot_data)
                for ip, p in enumerate(pot_data):
                    if p == "out_freq":
                        endp = ip
                        break
                pot_data = pot_data[:endp]
                if len(pot_data) > 1:
                    pair_style = "eann {} out_freq {}".format(
                        " ".join(pot_data), self.dump_period
                    )
                else:
                    pair_style = "eann {}".format(" ".join(pot_data))
                content += "pair_style  {}\n".format(pair_style)
                # NOTE: make out_freq consistent with dump_period
                if self.pair_coeff is None:
                    pair_coeff = "double * *"
                else:
                    pair_coeff = self.pair_coeff
                content += "pair_coeff	{} {}\n".format(
                    pair_coeff, " ".join(self.type_list)
                )
            elif potential == "deepmd":
                content += "pair_style  {} out_freq {}\n".format(
                    self.pair_style, self.dump_period
                )
                content += "pair_coeff	{} {}\n".format(
                    self.pair_coeff, " ".join(self.type_list)
                )
            elif potential == "nequip":
                content += "pair_style  {}\n".format(self.pair_style)
                content += "pair_coeff	{} {}\n".format(
                    self.pair_coeff, " ".join(self.type_list)
                )
            else:
                content += "pair_style {}\n".format(self.pair_style)
                # content += "pair_coeff {} {}\n".format(self.pair_coeff, " ".join(self.type_list))
                content += "pair_coeff {}\n".format(self.pair_coeff)

        if self.pair_modify is not None:
            content += f"pair_modify {self.pair_modify}\n"

        content += "\n"

        # TODO: kspace should be set with pair at the same time
        if self.kspace_style is not None:
            content += f"kspace_style  {self.kspace_style}\n\n"

        # neighbor
        content += "neighbor        {}\n".format(self.neighbor)
        if self.neigh_modify:
            content += "neigh_modify        {}\n".format(self.neigh_modify)
        content += "\n"

        # - constraint
        mobile_text, frozen_text = parse_constraint_info(atoms, self.constraint)
        if mobile_text:  # NOTE: sometimes all atoms are fixed
            content += "group mobile id %s\n" % mobile_text
            content += "\n"
        if frozen_text:  # not empty string
            # content += "region bottom block INF INF INF INF 0.0 %f\n" %zmin # unit A
            content += "group frozen id %s\n" % frozen_text
            content += "fix cons frozen setforce 0.0 0.0 0.0\n"
        content += "\n"

        # - outputs
        # TODO: use more flexible notations
        if self.task == "min":
            content += (
                "thermo_style    custom step pe ke etotal temp press vol fmax fnorm\n"
            )
        elif self.task == "md":
            content += "compute mobileTemp mobile temp\n"
            content += "thermo_style    custom step c_mobileTemp pe ke etotal press vol lx ly lz xy xz yz\n"
        else:
            pass
        content += "thermo          {}\n".format(self.dump_period)
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

        # add extra fix
        for i, fix_info in enumerate(self.extra_fix):
            if isinstance(fix_info, str):  # fix ID command
                content += "{:<24s}  {:<24s}  {:<s}\n".format(
                    "fix", f"extra{i}", fix_info
                )
            else:  # fix ID group-ID command
                group_indices = create_a_group(atoms, fix_info[0])
                group_text = convert_indices(
                    group_indices, index_convention="py"
                )  # py-index -> lmp-index text
                content += "{:<24s}  {:<24s}  id  {:<s}  \n".format(
                    "group", f"extra_group_{i}", group_text
                )
                content += "{:<24s}  {:<24s}  {:<s}  {:<s}\n".format(
                    "fix", f"extra{i}", f"extra_group_{i}", fix_info[1]
                )
        content += "\n"

        # run type
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
                # TODO: We should better move this to driver setting.
                try:
                    from ..potential.managers.plumed.calculators.plumed2 import (
                        update_stride_and_file,
                    )

                    plumed_inp = update_stride_and_file(
                        self.plumed, wdir=str(self.directory), stride=self.dump_period
                    )
                    with open(os.path.join(self.directory, "plumed.inp"), "w") as fopen:
                        fopen.write("".join(plumed_inp))
                    content += "fix             metad all plumed plumedfile plumed.inp outfile plumed.out\n"
                except:
                    raise RuntimeError(
                        "Plumed Bias is included but cannot be imported."
                    )
            content += f"run             {self.steps}\n"
        else:
            # TODO: NEB?
            ...

        # output file
        in_file = os.path.join(self.directory, ASELMPCONFIG.input_fname)
        with open(in_file, "w") as fopen:
            fopen.write(content)

        return


if __name__ == "__main__":
    ...
