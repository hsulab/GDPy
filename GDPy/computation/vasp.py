#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import copy
import dataclasses
import json
import warnings
import pathlib
from typing import Union, List, NoReturn
from collections import Counter

import shutil

from pathlib import Path

import numpy as np 

from ase import Atoms 
from ase.io import read, write
from ase.constraints import FixAtoms

from GDPy.data.trajectory import Trajectory
from GDPy.builder.constraints import parse_constraint_info
from GDPy.computation.utils import create_single_point_calculator
from GDPy.computation.driver import AbstractDriver, DriverSetting

"""Driver for VASP."""

def run_vasp(name, command, directory):
    """Run vasp from the command. 
    
    ASE Vasp does not treat restart of a MD simulation well. Therefore, we run 
    directly from the command if INCAR aready exists.
    
    """
    import subprocess
    from ase.calculators.calculator import EnvironmentError, CalculationFailed

    try:
        proc = subprocess.Popen(command, shell=True, cwd=directory)
    except OSError as err:
        # Actually this may never happen with shell=True, since
        # probably the shell launches successfully.  But we soon want
        # to allow calling the subprocess directly, and then this
        # distinction (failed to launch vs failed to run) is useful.
        msg = 'Failed to execute "{}"'.format(command)
        raise EnvironmentError(msg) from err

    errorcode = proc.wait()

    if errorcode:
        path = os.path.abspath(directory)
        msg = ('Calculator "{}" failed with command "{}" failed in '
               '{} with error code {}'.format(name, command,
                                              path, errorcode))
        raise CalculationFailed(msg)

    return

# vasp utils
def read_sort(directory):
    """Create the sorting and resorting list from ase-sort.dat.

    If the ase-sort.dat file does not exist, the sorting is redone.

    """
    sortfile = directory / 'ase-sort.dat'
    if os.path.isfile(sortfile):
        sort = []
        resort = []
        with open(sortfile, 'r') as fd:
            for line in fd:
                s, rs = line.split()
                sort.append(int(s))
                resort.append(int(rs))
    else:
        # warnings.warn(UserWarning, 'no ase-sort.dat')
        raise ValueError('no ase-sort.dat')

    return sort, resort

@dataclasses.dataclass
class VaspDriverSetting(DriverSetting):

    def __post_init__(self):
        """Convert parameters into driver-specific ones.

        These parameters are frozen when the driver is initialised.

        """
        # - update internals that are specific for each calculator...
        if self.task == "min":
            # minimisation
            if self.min_style == "bfgs":
                ibrion = 1
            elif self.min_style == "cg":
                ibrion = 2
            else:
                #raise ValueError(f"Unknown minimisation {self.min_style} for vasp".)
                ...

            self._internals.update(
                ibrion = ibrion,
                potim = self.maxstep
            )

        if self.task == "md":
            # some general
            potim = self.timestep
            # TODO: init vel here?
            ibrion, isif = 0, 0
            if self.md_style == "nve":
                smass, mdalgo = -3, 2
                self._internals.update(
                    ibrion=ibrion, potim=potim, isif=isif, 
                    smass=smass, mdalgo=mdalgo
                )
            elif self.md_style == "nvt":
                #assert self.init_params["smass"] > 0, "NVT needs positive SMASS."
                smass = 0.
                if self.tend is None:
                    self.tend = self.temp
                tebeg, teend = self.temp, self.tend
                self._internals.update(
                    ibrion=ibrion, potim=potim, isif=isif, 
                    smass=smass, tebeg=tebeg, teend=teend
                )
            elif self.md_style == "npt":
                mdalgo = 3 # langevin thermostat
                # Parrinello-Rahman Lagrangian
                isif, smass = 3, 0
                if self.tend is None:
                    self.tend = self.temp
                tebeg, teend = self.temp, self.tend
                if self.pend is None:
                    self.pend = self.press
                # NOTE: pressure unit 1 GPa = 10 kBar
                #                     1 kB  = 1000 bar = 10^8 Pa
                pstress = 1e-3*self.press
                langevin_gamma = self.Tdamp # array, ps^-1
                langevin_gamma_l = self.Pdamp # real, ps^-1
                pmass = 100. # a.m.u., default 1000
                self._internals.update(
                    mdalgo = mdalgo,
                    ibrion=ibrion, potim=potim, isif=isif, 
                    # thermostat
                    smass=smass, tebeg=tebeg, teend=teend,
                    # barostat
                    pstress = pstress, pmass = pmass,
                    langevin_gamma=langevin_gamma,
                    langevin_gamma_l=langevin_gamma_l,
                )
            else:
                raise NotImplementedError(f"{self.md_style} is not supported yet.")
            
        if self.task == "freq":
            # ibrion, nfree, potim
            raise NotImplementedError("")

        return

    def get_run_params(self, *args, **kwargs):
        """"""
        # convergence criteria
        fmax_ = kwargs.get("fmax", self.fmax)
        etol_ = kwargs.get("etol", self.etol)

        if fmax_ is not None:
            ediffg = -1.*fmax_
        else:
            if etol_ is not None:
                ediffg = etol_
            else:
                ediffg = -5e-2

        steps_ = kwargs.get("steps", self.steps)
        nsw = steps_

        run_params = dict(
            constraint = kwargs.get("constraint", self.constraint),
            ediffg = ediffg, nsw=nsw
        )

        return run_params

class VaspDriver(AbstractDriver):

    name = "vasp"

    # - defaults
    default_task = "min"
    supported_tasks = ["min", "md", "freq"]

    # - system depandant params
    syswise_keys: List[str] = ["system", "kpts", "kspacing"]

    # - file names would be copied when continuing a calculation
    saved_fnames = ["OSZICAR", "OUTCAR", "POSCAR", "CONTCAR", "vasprun.xml"]

    def __init__(self, calc, params: dict, directory="./", *args, **kwargs):
        """"""
        self.calc = calc
        self.calc.reset()

        self._directory = pathlib.Path(directory)

        self._org_params = copy.deepcopy(params)

        self.setting = VaspDriverSetting(**params)

        return
    
    def run(self, atoms_: Atoms, read_exists: bool=True, extra_info: dict=None, *args, **kwargs):
        """Run the simulation."""
        atoms = atoms_.copy()

        # - backup old params
        # TODO: change to context message?
        calc_old = atoms.calc 
        params_old = copy.deepcopy(self.calc.parameters)

        # - set special keywords
        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)

        # - merge params
        run_params = self.setting.get_run_params(**kwargs)

        # - init params
        run_params.update(**self.setting.get_init_params())

        # - update some system-dependant params
        if "langevin_gamma" in run_params:
            ntypes = len(set(atoms.get_chemical_symbols()))
            run_params["langevin_gamma"] = [run_params["langevin_gamma"]]*ntypes

        # - check constraint
        cons_text = run_params.pop("constraint", None)
        mobile_indices, frozen_indices = parse_constraint_info(atoms, cons_text, ret_text=False)
        if frozen_indices:
            atoms._del_constraints()
            atoms.set_constraint(FixAtoms(indices=frozen_indices))
        #print("constraints: ", atoms.constraints)
        
        run_params["system"] = self.directory.name

        self.calc.set(**run_params)

        # BUG: ase 3.22.1 no special params in param_state
        #calc_params["inputs"]["lreal"] = self.calc.special_params["lreal"] 

        atoms.calc = self.calc

        # - run dynamics
        converged = self._continue(atoms, read_exists=read_exists)
        
        # NOTE: always use dynamics calc
        # TODO: should change positions and other properties for input atoms?
        # TODO: if not converged?
        traj_frames = self.read_trajectory()
        new_atoms = traj_frames[-1]

        if extra_info is not None:
            new_atoms.info.update(**extra_info)

        # - reset params
        self.calc.parameters = params_old
        self.calc.reset()
        if calc_old is not None:
            atoms.calc = calc_old

        return new_atoms
    
    def _continue(self, atoms, read_exists=True, *args, **kwargs):
        """Check whether continue unfinished calculation.

        NOTE: vasprun.xml maybe incomplete, which leads a runtime error in 
             read_trajectory.

        """
        print(f"run {self.directory}")
        # NOTE: some calculation can overwrite existed data
        converged = self.read_convergence()
        if not converged:
            # - backup previous data
            if read_exists:
                # TODO: add a max for continued calculations? 
                #       such calcs can be labelled as a failure
                # TODO: check WAVECAR to speed restart?
                for fname in self.saved_fnames:
                    curr_fpath = self.directory/fname
                    if curr_fpath.exists():
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
                # -- continue calculation
                #    update positions and velocities
                if (self.directory/"CONTCAR").exists():
                    # vasp does not finish one step so CONTCAR is empty
                    if (self.directory/"CONTCAR").stat().st_size != 0:
                        shutil.copy(self.directory/"CONTCAR", self.directory/"POSCAR")
                # TODO: update steps if MD 
                ...
            else:
                # - start from scratch
                ...
            # -- run calculation
            try:
                if not (self.directory/"INCAR").exists():
                    # create all input files
                    _ = atoms.get_forces()
                else:
                    run_vasp("vasp", atoms.calc.command, self.directory)
                # -- check whether the restart is converged
                converged = self.read_convergence()
            except OSError:
                converged = False
        print(f"end {self.directory}")
        print("converged: ", converged)

        return converged
    
    def read_convergence(self, *args, **kwargs) -> bool:
        """"""
        converged = super().read_convergence()

        scf_converged = False
        if (self.directory/"OUTCAR").exists():
            if hasattr(self.calc, "read_convergence"):
                scf_converged = self.calc.read_convergence()
                self._debug("SCF convergence: ", scf_converged)
            else:
                raise NotImplementedError()
        else:
            ...

        return (converged and scf_converged)
    
    def read_trajectory(self, add_step_info=True, *args, **kwargs) -> List[Atoms]:
        """Read trajectory in the current working directory.

        If the calculation failed, an empty atoms with errof info would be returned.

        """
        vasprun = self.directory / "vasprun.xml"
        backup_fmt = ("gbak.{:d}."+"vasprun.xml")

        # - read structures
        try:
            traj_frames_ = []
            # -- read backups
            idx = 0
            while True:
                backup_fname = backup_fmt.format(idx)
                backup_fpath = self.directory/backup_fname
                if Path(backup_fpath).exists():
                    traj_frames_.extend(read(backup_fpath, index=":", format="vasp-xml"))
                else:
                    break
                idx += 1
            if vasprun.exists() and vasprun.stat().st_size != 0:
                traj_frames_.extend(read(vasprun, ":")) # read current

            # - sort frames
            traj_frames = []
            sort, resort = read_sort(self.directory)
            for i, sorted_atoms in enumerate(traj_frames_):
                input_atoms = create_single_point_calculator(sorted_atoms, resort, "vasp")
                #if input_atoms is None:
                #    input_atoms = Atoms()
                #    input_atoms.info["error"] = str(self.directory)
                if input_atoms is not None:
                    if add_step_info:
                        input_atoms.info["step"] = i
                    traj_frames.append(input_atoms)
        except Exception as e:
            self._debug(e)
            atoms = Atoms()
            atoms.info["error"] = str(self.directory)
            traj_frames = [atoms]

        return Trajectory(images=traj_frames, driver_config=dataclasses.asdict(self.setting))


if __name__ == "__main__": 
    ...