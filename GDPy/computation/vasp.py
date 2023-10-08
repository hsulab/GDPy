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

from GDPy.builder.constraints import parse_constraint_info
from GDPy.computation.utils import create_single_point_calculator
from GDPy.computation.driver import AbstractDriver, DriverSetting

"""Driver for VASP."""
#: str
ASE_VASP_SORT_FNAME = "ase-sort.dat"

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

def read_sort(directory: pathlib.Path):
    """Create the sorting and resorting list from ase-sort.dat.

    If the ase-sort.dat file does not exist, the sorting is redone.

    """
    sortfile = directory / ASE_VASP_SORT_FNAME
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

    etol: float = None
    fmax: float = 0.05 

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

        # -- cmin: cell minimisation
        if self.task == "cmin":
            if self.min_style == "bfgs":
                ibrion = 1
            elif self.min_style == "cg":
                ibrion = 2
            else:
                #raise ValueError(f"Unknown minimisation {self.min_style} for vasp".)
                ...

            self._internals.update(
                isif = 3,
                ibrion = ibrion,
                potim = self.maxstep
            )

        if self.task == "md":
            # NOTE: Always use Selective Dynamics and MDALAGO
            #       since it properly treats the DOF and velocities
            # some general
            if self.velocity_seed is None:
                self.velocity_seed = np.random.randint(0, 10000)
            random_seed = [self.velocity_seed, 0, 0]

            potim = self.timestep
            # TODO: init vel here?
            ibrion, isif = 0, 0
            if self.md_style == "nve":
                smass, mdalgo = -3, 2
                self._internals.update(
                    ibrion=ibrion, potim=potim, isif=isif, 
                    smass=smass, mdalgo=mdalgo,
                    random_seed=random_seed
                )
            elif self.md_style == "nvt":
                #assert self.init_params["smass"] > 0, "NVT needs positive SMASS."
                smass, mdalgo = 0., 2
                if self.tend is None:
                    self.tend = self.temp
                tebeg, teend = self.temp, self.tend
                self._internals.update(
                    mdalgo=mdalgo,
                    ibrion=ibrion, potim=potim, isif=isif, 
                    smass=smass, tebeg=tebeg, teend=teend,
                    random_seed=random_seed
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
                    random_seed=random_seed
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

        # etol is prioritised
        if etol_ is not None:
            ediffg = etol_
        else:
            if fmax_ is not None:
                ediffg = -1.*fmax_
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
    supported_tasks = ["min", "cmin", "md", "freq"]

    # - system depandant params
    syswise_keys: List[str] = ["system", "kpts", "kspacing"]

    # - file names would be copied when continuing a calculation
    saved_fnames = ["OSZICAR", "OUTCAR", "POSCAR", "CONTCAR", "vasprun.xml", "REPORT"]

    def __init__(self, calc, params: dict, directory="./", *args, **kwargs):
        """"""
        super().__init__(calc, params, directory=directory, *args, **kwargs)

        self.setting = VaspDriverSetting(**params)

        return
    
    def _irun(self, atoms: Atoms, *args, **kwargs):
        """"""
        try:
            # - merge params
            run_params = self.setting.get_run_params(**kwargs)
            run_params.update(**self.setting.get_init_params())

            # - update some system-dependant params
            if "langevin_gamma" in run_params:
                ntypes = len(set(atoms.get_chemical_symbols()))
                run_params["langevin_gamma"] = [run_params["langevin_gamma"]]*ntypes
            run_params["system"] = self.directory.name

            # - check constraint
            cons_text = run_params.pop("constraint", None)
            mobile_indices, frozen_indices = parse_constraint_info(atoms, cons_text, ret_text=False)
            if frozen_indices:
                atoms._del_constraints()
                atoms.set_constraint(FixAtoms(indices=frozen_indices))
            #print("constraints: ", atoms.constraints)
            self.calc.set(**run_params)
            atoms.calc = self.calc

            # NOTE: ASE VASP does not write velocities and thermostat to POSCAR
            #       thus we manually call the function to write input files and
            #       run the calculation
            self.calc.write_input(atoms)
            if (self.directory/"CONTCAR").exists() and (self.directory/"CONTCAR").stat().st_size != 0:
                shutil.copy(self.directory/"CONTCAR", self.directory/"POSCAR")
            run_vasp("vasp", atoms.calc.command, self.directory)

        except Exception as e:
            self._debug(e)

        return
    
    def _resume(self, atoms: Atoms, *args, **kwargs):
        """"""
        # - update atoms and driver
        traj = self.read_trajectory()
        nframes = len(traj)
        if nframes > 0:
            # --- update atoms
            resume_atoms = traj[-1]
            resume_params = {}
            # --- update run_params in settings
            dump_period = 1
            target_steps = self.setting.get_run_params(*args, **kwargs)["nsw"]
            if target_steps > 0: # not a spc 
                steps = target_steps + dump_period - nframes*dump_period
                assert steps > 0, f"Steps should be greater than 0. (steps = {steps})"
                resume_params.update(steps=steps)
        else:
            resume_atoms = atoms
            resume_params = {}

        return resume_atoms, resume_params
    
    def read_force_convergence(self, *args, **kwargs) -> bool:
        """"""
        scf_converged = False
        if (self.directory/"OUTCAR").exists():
            if hasattr(self.calc, "read_convergence"):
                scf_converged = self.calc.read_convergence()
                self._print(f"SCF convergence: {scf_converged}@{self.directory.name}")
                #self._debug(f"ignore convergence: {self.ignore_convergence}")
            else:
                raise NotImplementedError()
        else:
            ...

        return scf_converged
    
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
            nframes = len(traj_frames_)
            natoms = len(traj_frames_[0])

            # - sort frames
            traj_frames = []
            if nframes > 0:
                if (self.directory/ASE_VASP_SORT_FNAME).exists():
                    sort, resort = read_sort(self.directory)
                else: # without sort file, use default order
                    sort, resort = list(range(natoms)), list(range(natoms))
                for i, sorted_atoms in enumerate(traj_frames_):
                    input_atoms = create_single_point_calculator(sorted_atoms, resort, "vasp")
                    #if input_atoms is None:
                    #    input_atoms = Atoms()
                    #    input_atoms.info["error"] = str(self.directory)
                    if input_atoms is not None:
                        if add_step_info:
                            input_atoms.info["step"] = i
                        traj_frames.append(input_atoms)
            else:
                ...
        except Exception as e:
            self._debug(e)
            atoms = Atoms()
            atoms.info["error"] = str(self.directory)
            traj_frames = [atoms]
        
        ret = traj_frames

        if (len(ret) > 0) and (not self.read_force_convergence()):
            ret[0].info["error"] = str(self.directory)

        return ret



if __name__ == "__main__": 
    ...
