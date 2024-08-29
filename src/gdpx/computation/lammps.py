#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import dataclasses
import io
import itertools
import os
import pathlib
import pickle
import shutil
import tarfile
import traceback
from collections.abc import Iterable
from typing import Dict, List, Mapping, NoReturn, Optional, Tuple

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import FileIOCalculator, all_changes
from ase.calculators.lammps import Prism, unitconvert
from ase.calculators.mixing import LinearCombinationCalculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data

from .. import config
from ..builder.constraints import parse_constraint_info, convert_indices
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


def parse_thermo_data(lines) -> dict:
    """Read energy ... results from log.lammps file."""
    # - parse input lines
    found_error = False
    start_idx, end_idx = None, None
    for idx, line in enumerate(lines):
        # - get the line index at the start of the thermo infomation
        #   test with 29Oct2020 and 23Jun2022
        if line.strip().startswith("Step"):
            start_idx = idx
        # - NOTE: find line index at the end
        if line.strip().startswith("ERROR: "):
            found_error = True
            end_idx = idx
        if line.strip().startswith("Loop time"):
            end_idx = idx
        if start_idx is not None and end_idx is not None:
            break
    else:
        end_idx = idx
    config._debug(f"INITIAL LAMMPS LOG INDEX: {start_idx} {end_idx}")

    # - check valid lines
    #   sometimes the line may not be complete
    ncols = len(lines[start_idx].strip().split())
    for i in range(end_idx, start_idx, -1):
        curr_data = lines[i].strip().split()
        curr_ncols = len(curr_data)
        config._debug(f"  LAMMPS LINE: {lines[i].strip()}")
        if curr_ncols == ncols:  # The Line has the full thermo info...
            try:
                step = int(curr_data[0])
                end_idx = i + 1
            except ValueError:
                ...
            finally:
                end_info = lines[i].strip()
                config._debug(f"  LAMMPS STEP: {end_info}")
                break
        else:
            ...
    else:
        end_idx = None  # even not one single complete line
    config._debug(f"FINAL   LAMMPS LOG INDEX: {start_idx} {end_idx}")

    if start_idx is None or end_idx is None:
        raise RuntimeError(f"ERROR   LAMMPS LOG INDEX {start_idx} {end_idx}.")

    # -- parse index of PotEng
    # TODO: save timestep info?
    thermo_keywords = lines[start_idx].strip().split()
    if "PotEng" not in thermo_keywords:
        raise RuntimeError(f"Cant find PotEng in lammps output.")
    thermo_data = []
    for x in lines[start_idx + 1 : end_idx]:
        x_data = x.strip().split()
        if x_data[0].isdigit():  # There may have some extra warnings... such as restart
            thermo_data.append(x_data)
    # thermo_data = np.array([line.strip().split() for line in thermo_data], dtype=float).transpose()
    thermo_data = np.array(thermo_data, dtype=float).transpose()
    # config._debug(thermo_data)
    thermo_dict = {}
    for i, k in enumerate(thermo_keywords):
        thermo_dict[k] = thermo_data[i]

    return thermo_dict, end_info


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

        self.conv_params = dict(
            Tdamp=unitconvert.convert(Tdamp, "time", "real", self.units),
            Pdamp=unitconvert.convert(Pdamp, "time", "real", self.units),
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

    units: str = "metal"

    ensemble: str = "nve"

    controller: dict = dataclasses.field(default_factory=dict)

    #: Whether fix com to the its initial position.
    fix_com: bool = False

    use_lmpvel: bool = True

    etol: float = 0
    fmax: float = 0.05

    neighbor: str = "0.0 bin"
    neigh_modify: Optional[str] = None

    extra_fix: List[str] = dataclasses.field(default_factory=list)

    plumed: Optional[str] = None

    def __post_init__(self):
        """"""
        if self.task == "min":
            self._internals.update(
                etol=self.etol,
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

    def get_minimisation_inputs(self, random_seed, group: str = "mobile") -> List[str]:
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
    ) -> List[str]:
        """Convert parameters into lammps input lines."""
        MD_FIX_ID: str = "controller"
        _init_md_params = dict(
            fix_id=MD_FIX_ID,
            group=group,
            timestep=unitconvert.convert(self.timestep, "time", "real", self.units),
        )

        if self.ensemble == "nve":
            lines = [
                "fix {fix_id:>24s} {group} nve".format(**_init_md_params),
                f"timestep {_init_md_params['timestep']}",
            ]
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
            lines = [thermo_line, f"timestep {_init_md_params['timestep']}"]
        elif self.ensemble == "npt":
            _init_md_params.update(
                Tstart=self.temp,
                Tstop=self.tend if self.tend else self.temp,
                Pstart=self.press,
                Pstop=self.press,  # FIXME: end pressure??
            )
            if self.controller:
                baro_cls_name = self.controller["name"] + "_" + self.ensemble
                baro_cls = controllers[baro_cls_name]
            else:
                ...
            barostat = baro_cls(units=self.units, **self.controller)
            if barostat.name == "parrinello_rahman":
                _init_md_params.update(**barostat.conv_params)
                baro_line = "fix {fix_id:>24s} {group} npt temp {Tstart} {Tstop} {Tdamp} aniso {Pstart} {Pstop} {Pdamp}".format(
                    **_init_md_params
                )
            else:
                raise RuntimeError(f"Unknown barostat {barostat}.")
            lines = [baro_line, f"timestep {_init_md_params['timestep']}"]
        else:
            raise RuntimeError(f"Unknown ensemble {self.ensemble}.")

        if self.fix_com:
            com_line = "fix  fix_com {group} recenter INIT INIT INIT".format(
                **_init_md_params
            )
            lines.insert(0, com_line)

        return lines

    def get_run_params(self, *args, **kwargs):
        """"""
        # - pop out special keywords
        # convergence criteria
        ftol_ = kwargs.pop("fmax", self.fmax)
        etol_ = kwargs.pop("etol", self.etol)
        if etol_ is None:
            etol_ = 0.0
        if ftol_ is None:
            ftol_ = 0.0

        steps_ = kwargs.pop("steps", self.steps)

        run_params = dict(
            steps=steps_,
            constraint=kwargs.get("constraint", self.constraint),
            etol=etol_,
            ftol=ftol_,
        )

        # - add extra parameters
        run_params.update(**kwargs)

        return run_params


class LmpDriver(AbstractDriver):
    """Use lammps to perform dynamics.

    Minimisation and/or molecular dynamics.

    """

    name = "lammps"

    special_keywords = {}

    default_task = "min"
    supported_tasks = ["min", "md"]

    #: List of output files would be saved when restart.
    saved_fnames: List[str] = [
        ASELMPCONFIG.log_filename,
        ASELMPCONFIG.trajectory_filename,
        ASELMPCONFIG.deviation_filename,
    ]

    def __init__(self, calc, params: dict, directory="./", *args, **kwargs):
        """"""
        calc, params = self._check_plumed(calc=calc, params=params)

        super().__init__(calc, params, directory=directory, *args, **kwargs)

        params.update(units=calc.units)
        self.setting = LmpDriverSetting(**params)

        return

    def _check_plumed(self, calc, params: dict):
        """"""
        # TODO: We should better move this to potential_manager.
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
        except ImportError:
            new_calc = calc
            new_params = params

        return new_calc, new_params

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """"""
        verified = super()._verify_checkpoint(*args, **kwargs)
        if verified:
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
        try:
            run_params = self.setting.get_init_params()
            run_params.update(**self.setting.get_run_params(**kwargs))

            if ckpt_wdir is None:  # start from the scratch
                ...
            else:
                checkpoints = sorted(
                    list(ckpt_wdir.glob("restart.*")),
                    key=lambda x: int(x.name.split(".")[1]),
                )
                self._debug(f"checkpoints to restart: {checkpoints}")
                target_steps = run_params["steps"]
                run_params.update(
                    read_restart=str(checkpoints[-1].resolve()),
                    steps=target_steps - int(checkpoints[-1].name.split(".")[1]),
                )
                # shutil.move(
                #     checkpoints[-1].parent / "traj.dump", self.directory / "traj.dump"
                # )

            dynamics = self._create_dynamics(atoms, *args, **kwargs)

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

            # - run
            _ = atoms.get_forces()
        except Exception as e:
            config._debug(traceback.format_exc())

        return

    @staticmethod
    def _read_a_single_trajectory(
        wdir: pathlib.Path,
        mdir,
        units: str,
        archive_path: pathlib.Path = None,
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
        thermo_dict, end_info = parse_thermo_data(log_io.readlines())

        # NOTE: last frame would not be dumpped if timestep not equals multiple*dump_period
        #       if there were any error,
        pot_energies = [
            unitconvert.convert(p, "energy", units, "ASE")
            for p in thermo_dict["PotEng"]
        ]
        nframes_thermo = len(pot_energies)
        nframes = min([nframes_traj, nframes_thermo])
        config._debug(
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
            # config._print(data)

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
    ) -> List[Atoms]:
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
            if lines[-1].strip().startswith("Total wall time:"):
                converged = True
        else:
            ...

        return converged


class Lammps(FileIOCalculator):

    #: Calculator name.
    name: str = "Lammps"

    #: Implemented properties.
    implemented_properties: List[str] = ["energy", "forces", "stress"]

    #: LAMMPS command.
    command: str = "lmp 2>&1 > lmp.out"
    # command: str = "lmp"

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
        # --- lmp params ---
        read_restart=None,
        units="metal",
        atom_style="atomic",
        processors="* * 1",
        # boundary = "p p p",
        newton=None,
        pair_style=None,
        pair_coeff=None,
        neighbor="0.0 bin",
        neigh_modify=None,
        mass="* 1.0",
        # - extra fix
        extra_fix=[],
        # - externals
        plumed=None,
    )

    #: Symbol to integer.
    type_list: List[str] = None

    #: Cached trajectory of the previous simulation.
    cached_traj_frames: List[Atoms] = None

    def __init__(self, command=None, label=name, **kwargs):
        """"""
        FileIOCalculator.__init__(self, command=command, label=label, **kwargs)

        # check command
        # if "-in" in self.command or ">" in self.command:
        #     raise RuntimeError(f"LAMMPS command must not contain input or output files.")
        # self.command = self.command + "  -in in.lammps 2>&1 > lmp.out"

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
            force_skew=True,
            prismobj=prismobj,
            velocities=write_velocities,
            units=self.units,
            atom_style=self.atom_style,
        )

        # write input
        self._write_input(atoms)

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
            mdir=curr_wdir, wdir=curr_wdir, units=self.units
        )
        converged_frame = self.cached_traj_frames[-1]

        self.results["forces"] = converged_frame.get_forces().copy()
        self.results["energy"] = converged_frame.get_potential_energy()

        # - add deviation info
        for k, v in converged_frame.info.items():
            if "devi" in k:
                self.results[k] = v

        return

    def _write_input(self, atoms) -> None:
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
        content += "box             tilt large\n"
        if self.read_restart is None:
            content += "read_data	    %s\n" % ASELMPCONFIG.inputstructure_filename
        else:
            content += f"read_restart    {self.read_restart}\n"
            # os.remove(ASELMPCONFIG.inputstructure_filename)
        content += "change_box      all triclinic\n"

        # - particle masses
        mass_line = "".join(
            "mass %d %f\n" % (idx + 1, atomic_masses[atomic_numbers[elem]])
            for idx, elem in enumerate(self.type_list)
        )
        content += mass_line
        content += "\n"

        # - pair, MLIP specific settings
        potential = self.pair_style.strip().split()[0]
        if potential == "reax/c":
            assert self.atom_style == "charge", "reax/c should have charge atom_style"
            content += "pair_style  {}\n".format(self.pair_style)
            content += "pair_coeff {} {}\n".format(
                self.pair_coeff, " ".join(self.type_list)
            )
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
            content += "pair_coeff	{} {}\n".format(pair_coeff, " ".join(self.type_list))
        elif potential == "deepmd":
            content += "pair_style  {} out_freq {}\n".format(
                self.pair_style, self.dump_period
            )
            content += "pair_coeff	{} {}\n".format(
                self.pair_coeff, " ".join(self.type_list)
            )
        elif potential == "nequip":
            content += "pair_style  {}\n".format(
                self.pair_style
            )
            content += "pair_coeff	{} {}\n".format(
                self.pair_coeff, " ".join(self.type_list)
            )
        else:
            content += "pair_style {}\n".format(self.pair_style)
            # content += "pair_coeff {} {}\n".format(self.pair_coeff, " ".join(self.type_list))
            content += "pair_coeff {}\n".format(self.pair_coeff)
        content += "\n"

        # - neighbor
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
        content += (
            "dump		1 all custom {} {} id type element x y z fx fy fz vx vy vz\n".format(
                self.dump_period, ASELMPCONFIG.trajectory_filename
            )
        )
        content += "dump_modify 1 element {} flush yes\n".format(
            " ".join(self.type_list)
        )
        content += "\n"

        # - add extra fix
        for i, fix_info in enumerate(self.extra_fix):
            if isinstance(fix_info, str):  # fix ID command
                content += "{:<24s}  {:<24s}  {:<s}\n".format("fix", f"extra{i}", fix_info)
            else:  # fix ID group-ID command
                group_indices = create_a_group(atoms, fix_info[0])
                group_text = convert_indices(group_indices, index_convention="py") # py-index -> lmp-index text
                content += "{:<24s}  {:<24s}  id  {:<s}  \n".format("group", f"extra_group_{i}", group_text)
                content += "{:<24s}  {:<24s}  {:<s}  {:<s}\n".format("fix", f"extra{i}", f"extra_group_{i}", fix_info[1])
        content += "\n"

        # --- run type
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
                    from ..potential.managers.plumed.calculators.plumed2 import update_stride_and_file
                    plumed_inp = update_stride_and_file(
                        self.plumed, wdir=str(self.directory), stride=self.dump_period
                    )
                    with open(os.path.join(self.directory, "plumed.inp"), "w") as fopen:
                        fopen.write("".join(plumed_inp))
                    content += "fix             metad all plumed plumedfile plumed.inp outfile plumed.out\n"
                except:
                    raise RuntimeError("Plumed Bias is included but cannot be imported.")
            content += f"run             {self.steps}\n"
        else:
            # TODO: NEB?
            ...

        # - output file
        in_file = os.path.join(self.directory, ASELMPCONFIG.input_fname)
        with open(in_file, "w") as fopen:
            fopen.write(content)

        return


if __name__ == "__main__":
    ...
