#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import shutil
import warnings
import subprocess
import pathlib
from pathlib import Path
import dataclasses

from collections.abc import Iterable
from typing import List, Mapping, Dict, Optional, NoReturn

import numpy as np

from ase import Atoms
from ase import units
from ase.data import atomic_numbers, atomic_masses
from ase.io import read, write
from ase.io.lammpsrun import read_lammps_dump_text
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.calculators.calculator import (
    CalculationFailed,
    Calculator, all_changes, PropertyNotImplementedError, FileIOCalculator
)
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lammps import unitconvert

from GDPy.computation.driver import AbstractDriver
from GDPy.utils.command import find_backups

from GDPy.builder.constraints import parse_constraint_info

dataclasses.dataclass(frozen=True)
class AseLammpsSettings:

    """File names."""

    inputstructure_filename: str = "stru.data"
    trajectory_filename: str = "traj.dump"
    input_fname: str = "in.lammps"
    log_filename: str = "log.lammps"
    deviation_filename: str = "model_devi.out"

#: Instance.
ASELMPCONFIG = AseLammpsSettings()

def parse_type_list(atoms):
    """Parse the type list based on input atoms."""
    # elements
    type_list = list(set(atoms.get_chemical_symbols()))
    type_list.sort() # by alphabet

    return type_list

def parse_thermo_data(logfile_path) -> dict:
    """Read energy ... results from log.lammps file."""
    # - read thermo data
    # better move to calculator class
    with open(logfile_path, "r") as fopen:
        lines = fopen.readlines()
    
    found_error = False
    start_idx, end_idx = None, None
    for idx, line in enumerate(lines):
        # - get the line index at the start of the thermo infomation
        if line.startswith("Step"):
            start_idx = idx
        # - NOTE: find line index at the end
        if line.startswith("ERROR: "):
            found_error = True
            end_idx = idx
        if line.startswith("Loop time"):
            end_idx = idx
        if start_idx is not None and end_idx is not None:
            break
    else:
        raise RuntimeError(f"Error in lammps output of {str(logfile_path)}.")
    end_info = lines[end_idx] # either loop time or error

    # -- parse index of PotEng
    # TODO: save timestep info?
    thermo_keywords = lines[start_idx].strip().split()
    if "PotEng" not in thermo_keywords:
        raise RuntimeError(f"Cant find PotEng in lammps output of {str(logfile_path)}.")
    thermo_data = lines[start_idx+1:end_idx]
    thermo_data = np.array([line.strip().split() for line in thermo_data], dtype=float).transpose()
    #print(thermo_data)
    thermo_dict = {}
    for i, k in enumerate(thermo_keywords):
        thermo_dict[k] = thermo_data[i]

    return thermo_dict, end_info

class LmpDriver(AbstractDriver):

    """Use lammps to perform dynamics.
    
    Minimisation and/or molecular dynamics.

    """

    name = "lammps"

    delete = []
    keyword: Optional[str] = None
    special_keywords = {}

    # - defaults
    default_task = "min"
    supported_tasks = ["min", "md"]

    default_init_params = {
        "min": {
            "min_style": "fire",
            "min_modify": "integrator verlet tmax 4"
        },
        "md": {
            "md_style": "nvt",
            "velocity_seed": None,
            "timestep": 1.0, # fs
            "temp": 300, # K
            "Tdamp": 100, # fs
            "pres": 1.0, # atm
            "Pdamp": 100
        }
    }

    default_run_params = {
        "min": {
            "etol": 0.0,
            "ftol": 0.05,
            "maxiter": 0,
            "maxeval": 0
        },
        "md": {
            "maxiter": 0
        }
    }

    param_mapping = {
        "fmax": "ftol",
        "steps": "maxiter"
    }

    #: List of output files would be saved when restart.
    saved_cards: List[str] = [ASELMPCONFIG.trajectory_filename]

    def _parse_params(self, params: dict):
        """Set several connected parameters."""
        super()._parse_params(params)

        # - update task
        self.init_params.update(task=self.task)

        # - special settings
        self.run_params = self.__set_special_params(self.run_params)

        return 
    
    def __set_special_params(self, params: dict) -> dict:
        """Set several connected parameters."""
        maxiter = params.get("maxiter", None)
        if maxiter is not None:
            params["maxeval"] = 2*maxiter

        return params
    
    def run(self, atoms_, read_exists: bool=True, extra_info: dict=None, **kwargs):
        """Run the driver.

        If the converged results were found in the current working directory,
        the simulation would not be performed again. Otherwise, the results would 
        be read.

        """
        atoms = atoms_.copy()

        # - backup old params
        # TODO: change to context message?
        calc_old = atoms.calc 
        params_old = copy.deepcopy(self.calc.parameters)

        # - set special keywords
        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)

        # - run params
        kwargs = self._map_params(kwargs)

        run_params = self.run_params.copy()
        run_params.update(kwargs)

        # - init params
        run_params.update(**self.init_params)

        run_params = self.__set_special_params(run_params)

        self.calc.set(**run_params)
        atoms.calc = self.calc

        # - run dynamics
        try:
            # NOTE: some calculation can overwrite existed data
            if read_exists:
                converged, end_info = atoms.calc._is_finished()
                if converged:
                    print(f"Found finished simulation {self.directory.name} with info {end_info}.")
                    atoms.calc.type_list = parse_type_list(atoms)
                    atoms.calc.read_results()
                else:
                    # NOTE: restart calculation!!!
                    _  = atoms.get_forces()
                    converged, end_info = atoms.calc._is_finished()
            else:
                _  = atoms.get_forces()
                converged, end_info = atoms.calc._is_finished()
        except OSError:
            converged = False
        #else:
        #    converged = False

        # NOTE: always use dynamics calc
        # TODO: should change positions and other properties for input atoms?
        assert converged and atoms.calc.cached_traj_frames is not None, "Failed to read results in lammps."
        new_atoms = atoms.calc.cached_traj_frames[-1]

        if extra_info is not None:
            new_atoms.info.update(**extra_info)

        # - reset params
        self.calc.parameters = params_old
        self.calc.reset()
        if calc_old is not None:
            atoms.calc = calc_old

        return new_atoms
    
    def read_trajectory(self, type_list=None, add_step_info=True, *args, **kwargs) -> List[Atoms]:
        """Read trajectory in the current working directory."""
        if type_list is not None:
            self.calc.type_list = type_list

        return self.calc._read_trajectory(add_step_info)


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
        constraint = None, # index of atoms, start from 0
        task = "min",
        # --- lmp params ---
        units = "metal",
        atom_style = "atomic",
        processors = "* * 1",
        boundary = "p p p",
        newton = None,
        pair_style = None,
        pair_coeff = None,
        neighbor = "0.0 bin",
        neigh_modify = None,
        mass = "* 1.0",
        dump_period = 1,
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
        min_modify = "integrator verlet tmax 4"
    )

    #: Symbol to integer.
    type_list: List[str] = None

    #: Cached trajectory of the previous simulation.
    cached_traj_frames: List[Atoms] = None

    def __init__(
        self, 
        command = None, 
        label = name, 
        #pair_style: Mapping = None, # pair_style specific parameters
        **kwargs
    ):
        """"""
        FileIOCalculator.__init__(self, command=command, label=label, **kwargs)

        self.command = command
        
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
        # check specorder
        self.type_list = parse_type_list(atoms)

        # init for creating the directory
        FileIOCalculator.calculate(self, atoms, properties, system_changes)

        return
    
    def write_input(self, atoms, properties=None, system_changes=None) -> NoReturn:
        """Write input file and input structure."""
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # - check velocities
        self.write_velocities = False
        if atoms.get_kinetic_energy() > 0.:
            self.write_velocities = True

        # write structure
        stru_data = os.path.join(self.directory, ASELMPCONFIG.inputstructure_filename)
        write_lammps_data(
            stru_data, atoms, specorder=self.type_list, 
            force_skew=True, prismobj=None, velocities=self.write_velocities,
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
        log_filepath = Path(os.path.join(self.directory, ASELMPCONFIG.log_filename))

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
        self.cached_traj_frames = self._read_trajectory()
        converged_frame = self.cached_traj_frames[-1]

        self.results["forces"] = converged_frame.get_forces().copy()
        self.results["energy"] = converged_frame.get_potential_energy()

        # - add deviation info
        for k, v in converged_frame.info.items():
            if "devi" in k:
                self.results[k] = v

        return

    def _read_trajectory(self, add_step_info: bool=True) -> List[Atoms]:
        """Read the trajectory and its corresponding thermodynamics data."""
        # NOTE: always use dynamics calc
        # - read trajectory that contains positions and forces
        _directory_path = Path(self.directory)
        # NOTE: forces would be zero if setforce 0 is set
        traj_frames = read(
            _directory_path / ASELMPCONFIG.trajectory_filename, ":", "lammps-dump-text", 
            #specorder=self.type_list, # NOTE: elements are written to dump file
            units=self.units
        )
        nframes_traj = len(traj_frames)

        # - read thermo data
        thermo_dict, end_info = parse_thermo_data(_directory_path / ASELMPCONFIG.log_filename)

        # NOTE: last frame would not be dumpped if timestep not equals multiple*dump_period
        #       if there were any error, 
        pot_energies = [unitconvert.convert(p, "energy", self.units, "ASE") for p in thermo_dict["PotEng"]]
        nframes_thermo = len(pot_energies)
        nframes = min([nframes_traj, nframes_thermo])

        # TODO: check whether steps in thermo and traj are consistent
        pot_energies = pot_energies[:nframes]
        traj_frames = traj_frames[:nframes]
        assert len(pot_energies) == len(traj_frames), f"Number of pot energies and frames are inconsistent at {str(self.directory)}."

        for pot_eng, atoms in zip(pot_energies, traj_frames):
            forces = atoms.get_forces()
            # NOTE: forces have already been converted in ase read, so velocities are
            #forces = unitconvert.convert(forces, "force", self.units, "ASE")
            sp_calc = SinglePointCalculator(atoms, energy=pot_eng, forces=forces)
            atoms.calc = sp_calc
        
        # - check model_devi.out
        # TODO: convert units?
        devi_path = _directory_path / ASELMPCONFIG.deviation_filename
        if devi_path.exists():
            with open(devi_path, "r") as fopen:
                lines = fopen.readlines()
            dkeys = ("".join([x for x in lines[0] if x != "#"])).strip().split()
            dkeys = [x.strip() for x in dkeys][1:]
            data = np.loadtxt(devi_path, dtype=float)
            ncols = data.shape[-1]
            data = data.reshape(-1,ncols)
            data = data.transpose()[1:,:len(traj_frames)]

            for i, atoms in enumerate(traj_frames):
                for j, k in enumerate(dkeys):
                    atoms.info[k] = data[j,i]
        
        # - label steps
        if add_step_info:
            for i, atoms in enumerate(traj_frames):
                atoms.info["source"] = _directory_path.name
                atoms.info["step"] = int(thermo_dict["Step"][i])

        return traj_frames
    
    def _write_input(self, atoms) -> NoReturn:
        """Write input file in.lammps"""
        # - write in.lammps
        content = ""
        content += "units           %s\n" %self.units
        content += "atom_style      %s\n" %self.atom_style

        # - mpi settings
        if self.processors is not None:
            content += "processors {}\n".format(self.processors) # if 2D simulation
        
        # - simulation box
        content += "boundary        p p p\n"
        content += "\n"
        if self.newton:
            content += "newton {}\n".format(self.newton)
        content += "box             tilt large\n"
        content += "read_data	    %s\n" %ASELMPCONFIG.inputstructure_filename
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
            content += "pair_coeff  {}\n".format(self.pair_coeff)
        else:
            content += "pair_style {}\n".format(self.pair_style)
            content += "pair_coeff {} {}\n".format(self.pair_coeff, " ".join(self.type_list))
        content += "\n"

        # - neighbor
        content += "neighbor        {}\n".format(self.neighbor)
        if self.neigh_modify:
            content += "neigh_modify        {}\n".format(self.neigh_modify)
        content += "\n"

        # - constraint
        mobile_text, frozen_text = parse_constraint_info(atoms, self.constraint)
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
            if not self.write_velocities:
                velocity_seed = self.velocity_seed
                if velocity_seed is None:
                    velocity_seed = np.random.randint(0,10000)
                content += "velocity        mobile create {} {}\n".format(self.temp, velocity_seed)
        
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
            content += f"run             {self.maxiter}\n"
        else:
            # TODO: NEB?
            pass

        # - output file
        in_file = os.path.join(self.directory, ASELMPCONFIG.input_fname)
        with open(in_file, "w") as fopen:
            fopen.write(content)

        return
 

if __name__ == "__main__":
    pass