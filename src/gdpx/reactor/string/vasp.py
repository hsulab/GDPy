#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import dataclasses
import os
import pathlib
import re
import shutil
import traceback

from typing import Union, List

import numpy as np

from ase import Atoms
from ase import units
from ase.io import read, write
from ase.calculators.cp2k import parse_input, InputSection
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.neb import NEB

from .string import AbstractStringReactor, StringReactorSetting
from .. import parse_constraint_info


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


@dataclasses.dataclass
class VaspStringReactorSetting(StringReactorSetting):

    backend: str = "vasp"

    #: Number of tasks/processors/cpus for each image.
    ntasks_per_image: int = 1

    def __post_init__(self):
        """"""
        self._internals.update(
            # ---
            ibrion = 3,
            potim = 0,
            isif = 2,
            # ---
            lclimb = self.climb,
            ichain = 0,
            images = self.nimages-2,
            iopt = 1,
            spring = -5,
        )

        return
    
    def get_run_params(self, *args, **kwargs):
        """"""
        # - convergence criteria
        fmax_ = kwargs.get("fmax", self.fmax)
        steps_ = kwargs.get("steps", self.steps)

        run_params = dict(
            constraint = kwargs.get("constraint", self.constraint),
            ediffg = fmax_*-1., nsw=steps_
        )

        return run_params


class VaspStringReactor(AbstractStringReactor):

    name: str = "vasp"

    traj_name: str = "01/OUTCAR"

    def __init__(self, calc=None, params={}, ignore_convergence=False, directory="./", *args, **kwargs) -> None:
        """"""
        self.calc = calc
        if self.calc is not None:
            self.calc.reset()

        self.ignore_convergence = ignore_convergence

        self.directory = directory

        # - parse params
        self.setting = VaspStringReactorSetting(**params)
        self._debug(self.setting)

        return
    
    def _verify_checkpoint(self):
        """Check if the current directory has any valid outputs or it just created 
            the input files.

        """
        verified = super()._verify_checkpoint()
        if verified:
            checkpoints = list(self.directory.glob("*vasprun.xml"))
            self._debug(f"checkpoints: {checkpoints}")
            if not checkpoints:
                verified = False
        else:
            ...

        return verified
    
    def _irun(self, structures: List[Atoms], ckpt_wdir=None, *args, **kwargs):
        """"""
        try:
            # --
            run_params = self.setting.get_run_params(**kwargs)
            run_params.update(**self.setting.get_init_params())

            if ckpt_wdir is None: # start from the scratch
                images = self._align_structures(structures)
                write(self.directory/"images.xyz", images)
            else:
                # - update structures
                rep_dirs = sorted([x.name for x in sorted(self.directory.glob(r"[0-9][0-9]"))])

                frames_ = []
                for x in rep_dirs[1:-1]:
                    frames_.append(read(self.directory/x/"OUTCAR", ":"))
                nframes = min([len(x) for x in frames_])
                assert nframes > 0, "At least one step finished before resume..."
                intermediates_ = [x[nframes-1] for x in frames_]
                images = [structures[0]] + intermediates_ + [structures[-1]]

                run_params.update(
                    steps = self.setting.steps - nframes
                )

            # -- check constraint
            atoms = images[0] # use the initial state
            cons_text = run_params.pop("constraint", None)
            mobile_indices, frozen_indices = parse_constraint_info(
                atoms, cons_text=cons_text, ignore_ase_constraints=True, ret_text=False
            )
            if frozen_indices:
                #atoms._del_constraints()
                #atoms.set_constraint(FixAtoms(indices=frozen_indices))
                frozen_indices = sorted(frozen_indices)
                for a in images:
                    a.set_constraint(FixAtoms(indices=frozen_indices))

            # -- add replica information
            for i, a in enumerate(images):
                rep_dir = (self.directory/str(i).zfill(2))
                rep_dir.mkdir() # TODO: exists?
                write(rep_dir/"POSCAR", a)

            # - update input
            self.calc.set(**run_params)
            atoms.calc = self.calc

            # - run calculation
            self.calc.write_input(atoms)
            if (self.directory/"POSCAR").exists():
                os.remove(self.directory/"POSCAR")
            run_vasp("vasp", atoms.calc.command, self.directory)

        except Exception as e:
            self._debug(e)
            self._debug(traceback.print_exc())

        return
    
    def read_convergence(self, *args, **kwargs):
        """"""
        converged = super().read_convergence(*args, **kwargs)

        with open(self.directory/"vasp.out", "r") as fopen:
            lines = fopen.readlines()
        
        for line in lines:
            if "reached required accuracy" in line:
                converged = True
                break

        return converged
    
    def _read_a_single_trajectory(self, wdir, *args, **kwargs):
        """

        NOTE: Fixed atoms have zero forces.

        """
        self._debug(f"***** read_trajectory *****")
        self._debug(f"{str(wdir)}")

        images = read(wdir/"images.xyz", ":")
        ini_atoms, fin_atoms = images[0], images[-1]

        # TODO: energy and forces of IS and FS?
        calc = SinglePointCalculator(
            ini_atoms, energy=ini_atoms.info["energy"], forces=np.zeros((len(ini_atoms), 3))
        )
        ini_atoms.calc = calc
        calc = SinglePointCalculator(
            fin_atoms, energy=fin_atoms.info["energy"], forces=np.zeros((len(fin_atoms), 3))
        )
        fin_atoms.calc = calc

        # - read OUTCARs
        frames_ = []
        for i in range(1, self.setting.nimages-1):
            curr_frames = read(wdir/f"{str(i).zfill(2)}"/"OUTCAR", ":")
            frames_.append(curr_frames)

        # nframes may not consistent across replicas 
        # due to unfinished calculations
        nframes_list = [len(x) for x in frames_]
        nsteps = min(nframes_list) 

        frames = []
        for i in range(nsteps):
            curr_frames = [ini_atoms] + [frames_[j][i] for j in range(self.setting.nimages-2)] + [fin_atoms]
            frames.append(curr_frames)

        return frames


if __name__ == "__main__":
    ...
