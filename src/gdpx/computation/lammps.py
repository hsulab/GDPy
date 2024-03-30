#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import io
import itertools
import shutil
import warnings
import subprocess
import pathlib
import pickle
import tarfile
import dataclasses

from collections.abc import Iterable
from typing import List, Mapping, Dict, Optional, NoReturn, Tuple

import numpy as np

from ase import Atoms
from ase import units
from ase.data import atomic_numbers, atomic_masses
from ase.io import read, write
from ase.io.lammpsrun import read_lammps_dump_text
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.calculators.lammps import unitconvert, Prism
from ase.calculators.calculator import (
    CalculationFailed,
    Calculator, all_changes, PropertyNotImplementedError, FileIOCalculator
)
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.mixing import LinearCombinationCalculator

from .. import config
from ..builder.constraints import parse_constraint_info
from .driver import AbstractDriver, DriverSetting

from ..potential.managers.plumed.calculators.plumed2 import Plumed, update_stride_and_file


dataclasses.dataclass(frozen=True)
class AseLammpsSettings:

    """File names."""

    inputstructure_filename: str = "stru.data"
    trajectory_filename: str = "traj.dump"
    input_fname: str = "in.lammps"
    log_filename: str = "log.lammps"
    deviation_filename: str = "model_devi.out"
    prism_filename: str = "ase-prism.bindat"

#: Instance.
ASELMPCONFIG = AseLammpsSettings()

def parse_type_list(atoms):
    """Parse the type list based on input atoms."""
    # elements
    type_list = list(set(atoms.get_chemical_symbols()))
    type_list.sort() # by alphabet

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
    config._debug(f"Initial lammps LOG index: {start_idx} {end_idx}")
    
    # - check valid lines
    #   sometimes the line may not be complete
    ncols = len(lines[start_idx].strip().split())
    for i in range(end_idx, start_idx, -1):
        curr_data = lines[i].strip().split()
        curr_ncols = len(curr_data)
        config._debug(f"Error: {lines[i]}")
        if curr_ncols == ncols: # still log step info and no LOOP
            try:
                step = int(curr_data[0])
                end_idx = i+1
            except ValueError:
                config._debug(f"Error: {lines[i]}")
            finally:
                break
        else:
            ...
    else:
        end_idx = None # even not one single complete line
    config._debug(f"Sanitised lammps LOG index: {start_idx} {end_idx}")

    if start_idx is None or end_idx is None:
        raise RuntimeError(f"Error in lammps output with start {start_idx} end {end_idx}.")
    end_info = lines[end_idx] # either loop time or error
    config._debug(f"lammps END info: {end_info}")

    config._debug(f"lammps LOG index: {start_idx} {end_idx}")

    # -- parse index of PotEng
    # TODO: save timestep info?
    thermo_keywords = lines[start_idx].strip().split()
    if "PotEng" not in thermo_keywords:
        raise RuntimeError(f"Cant find PotEng in lammps output.")
    thermo_data = []
    for x in lines[start_idx+1:end_idx]:
        x_data = x.strip().split()
        if x_data[0].isdigit(): # There may have some extra warnings... such as restart
            thermo_data.append(x_data)
    #thermo_data = np.array([line.strip().split() for line in thermo_data], dtype=float).transpose()
    thermo_data = np.array(thermo_data, dtype=float).transpose() 
    #config._debug(thermo_data)
    thermo_dict = {}
    for i, k in enumerate(thermo_keywords):
        thermo_dict[k] = thermo_data[i]

    return thermo_dict, end_info

def read_single_simulation(
        mdir, wdir: pathlib.Path, prefix: str, units: str, add_step_info=True,
        archive_path: pathlib.Path=None
    ):
    """"""
    # - get FileIO
    if archive_path is None:
        traj_io = open(wdir/ASELMPCONFIG.trajectory_filename, "r")
        log_io = open(wdir/ASELMPCONFIG.log_filename, "r")
        prism_file = wdir/ASELMPCONFIG.prism_filename
        if prism_file.exists():
            prism_io = open(prism_file, "rb")
        else:
            prism_io = None
        devi_path = wdir / (prefix+ASELMPCONFIG.deviation_filename)
        if devi_path.exists():
            devi_io = open(devi_path, "r")
        else:
            devi_io = None
        colvar_path = wdir/"COLVAR"
        if colvar_path.exists():
            colvar_io = open(colvar_path, "r")
        else:
            colvar_io = None
    else:
        rpath = wdir.relative_to(mdir.parent)
        traj_tarname = str(rpath/ASELMPCONFIG.trajectory_filename)
        prism_tarname = str(rpath/ASELMPCONFIG.prism_filename)
        log_tarname = str(rpath/ASELMPCONFIG.log_filename)
        devi_tarname = str(rpath/ASELMPCONFIG.deviation_filename)
        colvar_tarname = str(rpath/"COLVAR")
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
            else: # TODO: if not find target traj?
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
        index=":", format="lammps-dump-text", prismobj=prismobj, units=units
    )
    nframes_traj = len(curr_traj_frames_)
    timesteps = timesteps[:nframes_traj] # avoid incomplete structure

    # - read thermo data
    thermo_dict, end_info = parse_thermo_data(log_io.readlines())

    # NOTE: last frame would not be dumpped if timestep not equals multiple*dump_period
    #       if there were any error, 
    pot_energies = [unitconvert.convert(p, "energy", units, "ASE") for p in thermo_dict["PotEng"]]
    nframes_thermo = len(pot_energies)
    nframes = min([nframes_traj, nframes_thermo])
    config._debug(f"nframes in lammps: {nframes} traj {nframes_traj} thermo {nframes_thermo}")

    # NOTE: check whether steps in thermo and traj are consistent
    #pot_energies = pot_energies[:nframes]
    #curr_traj_frames = curr_traj_frames[:nframes]
    #assert len(pot_energies) == len(curr_traj_frames), f"Number of pot energies and frames are inconsistent at {str(wdir)}."

    curr_traj_frames, curr_energies = [], []
    for i, t in enumerate(timesteps):
        if t in thermo_dict["Step"]:
            curr_atoms = curr_traj_frames_[i]
            curr_atoms.info["step"] = t
            curr_traj_frames.append(curr_atoms)
            curr_energies.append(pot_energies[thermo_dict["Step"].tolist().index(t)])

    for pot_eng, atoms in zip(curr_energies, curr_traj_frames):
        forces = atoms.get_forces()
        # NOTE: forces have already been converted in ase read, so velocities are
        sp_calc = SinglePointCalculator(atoms, energy=pot_eng, forces=forces)
        atoms.calc = sp_calc

    # - check model_devi.out
    # TODO: convert units?
    if devi_io is not None:
        lines = devi_io.readlines()
        if "#" in lines[0]: # the first file
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
        #config._print(data)

        for i, atoms in enumerate(curr_traj_frames):
            for j, k in enumerate(dkeys):
                try:
                    atoms.info[k] = data[j,i]
                except IndexError:
                    # NOTE: Some potentials donot print last frames of min
                    #       for example, lammps
                    atoms.info[k] = 0.
    else:
        ...
    
    # - check COLVAR
    if colvar_io is not None:
        # - read latest COLVAR Files
        names = colvar_io.readline().split()[2:]
        colvar_io.seek(0)
        colvars = np.loadtxt(colvar_io)
        #print("colvars: ", colvars.shape)
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

@dataclasses.dataclass
class LmpDriverSetting(DriverSetting):

    min_style: str = "fire"
    min_modify: str = "integrator verlet tmax 4"

    etol: float = 0
    fmax: float = 0.05

    neighbor: str = "0.0 bin"
    neigh_modify: str = None

    extra_fix: List[str] = dataclasses.field(default_factory=list)

    plumed: str = None

    def __post_init__(self):
        """"""
        if self.task == "min":
            self._internals.update(
                min_style = self.min_style,
                min_modify = self.min_modify,
                etol = self.etol,
                ftol = self.fmax,
                #maxstep = self.maxstep
            )
        
        if self.task == "md":
            self._internals.update(
                md_style = self.md_style,
                timestep = self.timestep,
                velocity_seed = self.velocity_seed,
                ignore_atoms_velocities = self.ignore_atoms_velocities,
                remove_rotation = self.remove_rotation,
                remove_translation = self.remove_translation,
                temp = self.temp,
                # TODO: end temperature
                Tdamp = self.Tdamp,
                press = self.press,
                Pdamp = self.Pdamp,
                # - ext
                plumed = self.plumed,
            )
        
        # - shared params
        self._internals.update(
            task = self.task,
            dump_period = self.dump_period,
            ckpt_period = self.ckpt_period
        )

        # - special params
        self._internals.update(
            neighbor = self.neighbor,
            neigh_modify = self.neigh_modify,
            extra_fix = self.extra_fix
        )

        return 
    
    def get_run_params(self, *args, **kwargs):
        """"""
        # - pop out special keywords
        # convergence criteria
        ftol_ = kwargs.pop("fmax", self.fmax)
        etol_ = kwargs.pop("etol", self.etol)
        if etol_ is None:
            etol_ = 0.
        if ftol_ is None:
            ftol_ = 0.

        steps_ = kwargs.pop("steps", self.steps)

        run_params = dict(
            constraint = kwargs.get("constraint", self.constraint),
            etol=etol_, ftol=ftol_, maxiter=steps_, maxeval=2*steps_
        )

        # - add extra parameters
        run_params.update(
            **kwargs
        )

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
    saved_fnames: List[str] = [ASELMPCONFIG.log_filename, ASELMPCONFIG.trajectory_filename, ASELMPCONFIG.deviation_filename]

    def __init__(self, calc, params: dict, directory="./", *args, **kwargs):
        """"""
        calc, params = self._check_plumed(calc=calc, params=params)

        super().__init__(calc, params, directory=directory, *args, **kwargs)
        self.setting = LmpDriverSetting(**params)

        return
    
    def _check_plumed(self, calc, params: dict):
        """"""
        new_calc, new_params = calc, params
        if isinstance(calc, LinearCombinationCalculator):
            ncalcs = len(calc.calcs)
            assert ncalcs == 2, "Number of calculators should be 2."
            if isinstance(calc.calcs[0], Lammps) and isinstance(calc.calcs[1], Plumed):
                new_calc = calc.calcs[0]
                new_params = copy.deepcopy(params)
                new_params["plumed"] = "".join(calc.calcs[1].input)

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

    def _irun(self, atoms: Atoms, ckpt_wdir=None, *args, **kwargs):
        """"""
        try:
            # - params
            run_params = self.setting.get_init_params()
            run_params.update(**self.setting.get_run_params(**kwargs))

            if ckpt_wdir is None: # start from the scratch
                ...
            else:
                checkpoints = sorted(list(ckpt_wdir.glob("restart.*")), key=lambda x: int(x.name.split(".")[1]))
                self._debug(f"checkpoints to restart: {checkpoints}")
                target_steps = run_params["maxiter"]
                run_params.update(
                    read_restart = str(checkpoints[-1].resolve()),
                    maxiter = target_steps - int(checkpoints[-1].name.split(".")[1])
                )

            # - check constraint
            self.calc.set(**run_params)
            atoms.calc = self.calc

            # - run
            _ = atoms.get_forces()
        except Exception as e:
            config._debug(e)

        return
    
    def read_trajectory(
            self, type_list=None, add_step_info=True, 
            archive_path: pathlib.Path=None, *args, **kwargs
        ) -> List[Atoms]:
        """Read trajectory in the current working directory."""
        if type_list is not None:
            self.calc.type_list = type_list
        curr_units = self.calc.units

        # - find runs...
        prev_wdirs = sorted(self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"))
        self._debug(f"prev_wdirs: {prev_wdirs}")

        traj_list = []
        for w in prev_wdirs:
            curr_frames = read_single_simulation(
                mdir=self.directory, wdir=w, prefix="", units=curr_units, 
                add_step_info=add_step_info, archive_path=archive_path
            )
            traj_list.append(curr_frames)
        
        # Even though traj file may be empty, the read can give a empty list...
        traj_list.append(
            read_single_simulation(
                mdir=self.directory, wdir=self.directory, prefix="", units=curr_units, 
                add_step_info=add_step_info, archive_path=archive_path
            )
        )

        # -- concatenate
        traj_frames, ntrajs = [], len(traj_list)
        if ntrajs > 0:
            traj_frames.extend(traj_list[0])
            for i in range(1, ntrajs):
                assert np.allclose(traj_list[i-1][-1].positions, traj_list[i][0].positions), f"Traj {i-1} and traj {i} are not consecutive in positions."
                assert np.allclose(traj_list[i-1][-1].get_potential_energy(), traj_list[i][0].get_potential_energy()), f"Traj {i-1} and traj {i} are not consecutive in energy."
                traj_frames.extend(traj_list[i][1:])
        else:
            ...

        return traj_frames
    

class Lammps(FileIOCalculator):

    #: Calculator name.
    name: str = "Lammps"

    #: Implemented properties.
    implemented_properties: List[str] = ["energy", "forces", "stress"]

    #: LAMMPS command.
    command: str = "lmp 2>&1 > lmp.out"

    #: Default calculator parameters, NOTE which have ase units.
    default_parameters: dict = dict(
        # ase params
        task = "min",
        constraint = None, # index of atoms, start from 0
        ignore_atoms_velocities = False,
        # --- lmp params ---
        read_restart = None,
        units = "metal",
        atom_style = "atomic",
        processors = "* * 1",
        #boundary = "p p p",
        newton = None,
        pair_style = None,
        pair_coeff = None,
        neighbor = "0.0 bin",
        neigh_modify = None,
        mass = "* 1.0",
        dump_period = 1,
        ckpt_period = 100,
        # - md
        md_style = "nvt",
        md_steps = 0,
        velocity_seed = None,
        timestep = 1.0, # fs
        temp = 300,
        pres = 1.0,
        Tdamp = 100, # fs
        Pdamp = 100,
        # - minimisation
        etol = 0.0,
        ftol = 0.05,
        maxiter = 0, # NOTE: this is steps for MD
        maxeval = 0,
        min_style = "fire",
        min_modify = "integrator verlet tmax 4",
        # - extra fix
        extra_fix = [],
        # - externals
        plumed = None
    )

    #: Symbol to integer.
    type_list: List[str] = None

    #: Cached trajectory of the previous simulation.
    cached_traj_frames: List[Atoms] = None

    def __init__(
        self, 
        command = None, 
        label = name, 
        **kwargs
    ):
        """"""
        FileIOCalculator.__init__(self, command=command, label=label, **kwargs)

        # - check potential
        assert self.pair_style is not None, "pair_style is not set."

        return
    
    def __getattr__(self, key):
        """Corresponding getattribute-function."""
        if key != "parameters" and key in self.parameters:
            return self.parameters[key]
        return object.__getattribute__(self, key)
    
    def calculate(self, atoms=None, properties=["energy"],
            system_changes=all_changes): 
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
        self.write_velocities = False
        if atoms.get_kinetic_energy() > 0.:
            self.write_velocities = (True and not self.ignore_atoms_velocities)

        # write structure
        prismobj = Prism(atoms.get_cell()) # TODO: nonpbc?
        prism_file = os.path.join(self.directory, ASELMPCONFIG.prism_filename)
        with open(prism_file, "wb") as fopen:
            pickle.dump(prismobj, fopen)
        stru_data = os.path.join(self.directory, ASELMPCONFIG.inputstructure_filename)
        write_lammps_data(
            stru_data, atoms, specorder=self.type_list, 
            force_skew=True, prismobj=prismobj, velocities=self.write_velocities,
            units=self.units, atom_style=self.atom_style
        )

        # write input
        self._write_input(atoms)

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
        self.cached_traj_frames = read_single_simulation(
            mdir=curr_wdir, wdir=curr_wdir, prefix="", 
            units=self.units, add_step_info=True
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
        content =  f"restart         {self.ckpt_period}  restart.*.data\n\n"
        content += "units           %s\n" %self.units
        content += "atom_style      %s\n" %self.atom_style

        # - mpi settings
        if self.processors is not None:
            content += "processors {}\n".format(self.processors) # if 2D simulation
        
        # - simulation box
        pbc = atoms.get_pbc()
        if "boundary" in self.parameters:
            content += "boundary {0} \n".format(self.parameters["boundary"])
        else:
            content += "boundary {0} {1} {2} \n".format(
                *tuple("fp"[int(x)] for x in pbc) # sometimes s failed to wrap all atoms
            )
        content += "\n"
        if self.newton:
            content += "newton {}\n".format(self.newton)
        content += "box             tilt large\n"
        if self.read_restart is None:
            content += "read_data	    %s\n" %ASELMPCONFIG.inputstructure_filename
        else:
            content += f"read_restart    {self.read_restart}\n"
            #os.remove(ASELMPCONFIG.inputstructure_filename)
        content += "change_box      all triclinic\n"

        # - particle masses
        mass_line = "".join(
            "mass %d %f\n" %(idx+1,atomic_masses[atomic_numbers[elem]]) for idx, elem in enumerate(self.type_list)
        )
        content += mass_line
        content += "\n"

        # - pair, MLIP specific settings
        # TODO: neigh settings?
        potential = self.pair_style.strip().split()[0]
        if potential == "reax/c":
            assert self.atom_style == "charge", "reax/c should have charge atom_style"
            content += "pair_style  {}\n".format(self.pair_style)
            content += "pair_coeff {} {}\n".format(self.pair_coeff, " ".join(self.type_list))
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
            content += "pair_coeff	{} {}\n".format(pair_coeff, " ".join(self.type_list))
        elif potential == "deepmd":
            content += "pair_style  {} out_freq {}\n".format(self.pair_style, self.dump_period)
            content += "pair_coeff	{} {}\n".format(self.pair_coeff, " ".join(self.type_list))
        else:
            content += "pair_style {}\n".format(self.pair_style)
            #content += "pair_coeff {} {}\n".format(self.pair_coeff, " ".join(self.type_list))
            content += "pair_coeff {}\n".format(self.pair_coeff)
        content += "\n"

        # - neighbor
        content += "neighbor        {}\n".format(self.neighbor)
        if self.neigh_modify:
            content += "neigh_modify        {}\n".format(self.neigh_modify)
        content += "\n"

        # - constraint
        mobile_text, frozen_text = parse_constraint_info(atoms, self.constraint)
        if mobile_text: # NOTE: sometimes all atoms are fixed
            content += "group mobile id %s\n" %mobile_text
            content += "\n"
        if frozen_text: # not empty string
            # content += "region bottom block INF INF INF INF 0.0 %f\n" %zmin # unit A
            content += "group frozen id %s\n" %frozen_text
            content += "fix cons frozen setforce 0.0 0.0 0.0\n"
        content += "\n"

        # - outputs
        # TODO: use more flexible notations
        if self.task == "min":
            content += "thermo_style    custom step pe ke etotal temp press vol fmax fnorm\n"
        elif self.task == "md":
            content += "compute mobileTemp mobile temp\n"
            content += "thermo_style    custom step c_mobileTemp pe ke etotal press vol lx ly lz xy xz yz\n"
        else:
            pass
        content += "thermo          {}\n".format(self.dump_period) 

        # TODO: How to dump total energy?
        content += "dump		1 all custom {} {} id type element x y z fx fy fz vx vy vz\n".format(
            self.dump_period, ASELMPCONFIG.trajectory_filename
        )
        content += "dump_modify 1 element {}\n".format(" ".join(self.type_list))
        content += "\n"

        # - add extra fix
        for i, fix_info in enumerate(self.extra_fix):
            content += "{:<24s}  {:<24s}  {:<s}\n".format("fix", f"extra{i}", fix_info)
        
        # --- run type
        if self.task == "min":
            # - minimisation
            content += "min_style       {}\n".format(self.min_style)
            content += "min_modify      {}\n".format(self.min_modify)
            content += "minimize        {:f} {:f} {:d} {:d}\n".format(
                unitconvert.convert(self.etol, "energy", "ASE", self.units),
                unitconvert.convert(self.ftol, "force", "ASE", self.units),
                self.maxiter, self.maxeval
            )
        elif self.task == "md":
            if not self.write_velocities and self.read_restart is None:
                velocity_seed = self.velocity_seed
                if velocity_seed is None:
                    velocity_seed = np.random.randint(0,10000)
                velocity_command = "velocity        mobile create {} {} dist gaussian ".format(self.temp, velocity_seed)
                if hasattr(self, "remove_translation"):
                    if self.remove_translation:
                        velocity_command += "mom yes "
                if hasattr(self, "remove_rotation"):
                    if self.remove_rotation:
                        velocity_command += "rot yes "
                velocity_command += "\n"
                content += velocity_command
        
            if self.md_style == "nvt":
                Tdamp_ = unitconvert.convert(self.Tdamp, "time", "real", self.units)
                content += "fix             thermostat mobile nvt temp {} {} {}\n".format(
                    self.temp, self.temp, Tdamp_
                )
            elif self.md_style == "npt":
                pres_ = unitconvert.convert(self.pres, "pressure", "metal", self.units)
                Tdamp_ = unitconvert.convert(self.Tdamp, "time", "real", self.units)
                Pdamp_ = unitconvert.convert(self.Pdamp, "time", "real", self.units)
                content += "fix             thermostat mobile npt temp {} {} {} aniso {} {} {}\n".format(
                    self.temp, self.temp, Tdamp_, pres_, pres_, Pdamp_
                )
            elif self.md_style == "nve":
                content += "fix             thermostat mobile nve \n"

            timestep_ = unitconvert.convert(self.timestep, "time", "real", self.units)
            content += "\n"
            content += f"timestep        {timestep_}\n"
            if self.plumed is not None:
                plumed_inp = update_stride_and_file(self.plumed, wdir=str(self.directory), stride=self.dump_period)
                with open(os.path.join(self.directory, "plumed.inp"), "w") as fopen:
                    fopen.write("".join(plumed_inp))
                content += "fix             metad all plumed plumedfile plumed.inp outfile plumed.out\n"
            content += f"run             {self.maxiter}\n"
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
