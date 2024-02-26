#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import io
import os
import re
import time
import copy
import dataclasses
import json
import warnings
import pathlib
import tarfile
import traceback
from collections import Counter
from typing import Union, List, NoReturn

import shutil

from pathlib import Path

import numpy as np 

from ase import Atoms 
from ase.io import read, write
from ase.constraints import FixAtoms

from gdpx.builder.constraints import parse_constraint_info
from gdpx.computation.utils import create_single_point_calculator
from gdpx.computation.driver import AbstractDriver, DriverSetting

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
    saved_fnames = [
        "ase-sort.dat",
        "INCAR", "POSCAR","KPOINTS", "POTCAR",
        "OSZICAR", "OUTCAR", "CONTCAR", "vasprun.xml", "REPORT"
    ]

    def __init__(self, calc, params: dict, directory="./", *args, **kwargs):
        """"""
        super().__init__(calc, params, directory=directory, *args, **kwargs)

        self.setting = VaspDriverSetting(**params)

        return

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """Check whether there is a previous calculation in the `self.directory`."""
        verified = True
        if self.directory.exists():
            vasprun = self.directory/"vasprun.xml"
            if (vasprun.exists() and vasprun.stat().st_size != 0):
                temp_frames = read(vasprun, ":")
                try: 
                    _ = temp_frames[0].get_forces()
                except: # `RuntimeError: Atoms object has no calculator.`
                    verified = False
            else:
                verified = False
        else:
            verified = False

        return verified
    
    def _irun(self, atoms: Atoms, ckpt_wdir=None, cache_traj: List[Atoms]=None, *args, **kwargs):
        """"""
        try:
            if ckpt_wdir is None: # start from the scratch
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

                self.calc.set(**run_params)
                atoms.calc = self.calc
                # NOTE: ASE VASP does not write velocities and thermostat to POSCAR
                #       thus we manually call the function to write input files and
                #       run the calculation
                self.calc.write_input(atoms)
            else:
                self.calc.read_incar(ckpt_wdir/"INCAR") # read previous incar
                if cache_traj is None:
                    traj = self.read_trajectory()
                else:
                    traj = cache_traj
                nframes = len(traj)
                assert nframes > 0, "VaspDriver restarts with a zero-frame trajectory."
                dump_period = 1 # since we read vasprun.xml, every frame is dumped
                target_steps = self.setting.get_run_params(*args, **kwargs)["nsw"]
                if target_steps > 0: # not a spc 
                    # BUG: ...
                    steps = target_steps + dump_period - nframes*dump_period
                    assert steps > 0, f"Steps should be greater than 0. (steps = {steps})"
                    self.calc.set(nsw=steps)
                # NOTE: ASE VASP does not write velocities and thermostat to POSCAR
                #       thus we manually call the function to write input files and
                #       run the calculation
                self.calc.write_input(atoms)
                # To restart, velocities are always retained 
                #if (self.directory/"CONTCAR").exists() and (self.directory/"CONTCAR").stat().st_size != 0:
                #    shutil.copy(self.directory/"CONTCAR", self.directory/"POSCAR")
                shutil.copy(ckpt_wdir/"CONTCAR", self.directory/"POSCAR")

            run_vasp("vasp", self.calc.command, self.directory)

        except Exception as e:
            self._debug(e)
            self._debug(traceback.print_exc())

        return
    
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
    
    def _read_a_single_trajectory(self, wdir, archive_path=None, *args, **kwargs):
        """"""
        if archive_path is None:
            vasprun = wdir / "vasprun.xml"
            frames = read(vasprun, ":")
        else:
            target_name = str((wdir/"vasprun.xml").relative_to(self.directory.parent))
            with tarfile.open(archive_path, "r:gz") as tar:
                for tarinfo in tar:
                    if tarinfo.name == target_name:
                        fobj = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                        frames = read(fobj, ":", format="vasp-xml")
                        fobj.close()
                        break
                else: # TODO: if not find target traj?
                    ...

        return frames
    
    def read_trajectory(
            self, add_step_info=True, archive_path=None, *args, **kwargs
        ) -> List[Atoms]:
        """Read trajectory in the current working directory.

        If the calculation failed, an empty atoms with errof info would be returned.

        """
        # - read structures
        try:
            # -- read backups
            prev_wdirs = sorted(self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"))
            self._debug(f"prev_wdirs: {prev_wdirs}")

            traj_list = []
            for w in prev_wdirs:
                curr_frames = self._read_a_single_trajectory(w, archive_path=archive_path)
                traj_list.append(curr_frames)

            # Even though vasprun file may be empty, the read can give a empty list...
            vasprun = self.directory / "vasprun.xml"
            curr_frames = self._read_a_single_trajectory(
                self.directory, archive_path=archive_path
            )
            traj_list.append(curr_frames)

            # -- concatenate
            traj_frames_, ntrajs = [], len(traj_list)
            if ntrajs > 0:
                traj_frames_.extend(traj_list[0])
                for i in range(1, ntrajs):
                    assert np.allclose(traj_list[i-1][-1].positions, traj_list[i][0].positions), f"Traj {i-1} and traj {i} are not consecutive."
                    traj_frames_.extend(traj_list[i][1:])
            else:
                ...

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
                    # NOTE: calculation with only one unfinished step does not have forces
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
