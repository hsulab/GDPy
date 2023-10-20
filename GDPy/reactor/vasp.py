#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import dataclasses
import os
import pathlib
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
from ..builder.constraints import parse_constraint_info
from .utils import plot_bands, plot_mep


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

    def __init__(self, calc=None, params={}, ignore_convergence=False, directory="./", *args, **kwargs) -> None:
        """"""
        self.calc = calc
        if self.calc is not None:
            self.calc.reset()

        self.ignore_convergence = ignore_convergence

        self.directory = directory
        self.cache_nebtraj = self.directory/self.traj_name

        # - parse params
        self.setting = VaspStringReactorSetting(**params)
        self._debug(self.setting)

        return
    
    def _verify_checkpoint(self):
        """Check if the current directory has any valid outputs or it just created 
            the input files.

        """
        checkpoints = list(self.directory.glob("*vasprun.xml"))
        print(f"checkpoints: {checkpoints}")

        return checkpoints

    def run(self, structures: List[Atoms], read_cache=True, *args, **kwargs):
        """"""
        #super().run(structures=structures, *args, **kwargs)

        # - Double-Ended Methods...
        ini_atoms, fin_atoms = structures
        try:
            self._print(f"ini_atoms: {ini_atoms.get_potential_energy()}")
            self._print(f"fin_atoms: {fin_atoms.get_potential_energy()}")
        except RuntimeError:
            # RuntimeError: Atoms object has no calculator.
            self._print("Not energies attached to IS and FS.")

        # - backup old parameters
        prev_params = copy.deepcopy(self.calc.parameters)
        print(f"prev_params: {prev_params}")

        # -
        if not self._verify_checkpoint(): # is not a []
            self.directory.mkdir(parents=True, exist_ok=True)
            self._irun([ini_atoms, fin_atoms])
        else:
            # - check if converged
            converged = self.read_convergence()
            if not converged:
                if read_cache:
                    ...
                self._irun(structures, *args, **kwargs)
            else:
                ...
        
        self.calc.set(**prev_params)
        
        # - get results
        band_frames = self.read_trajectory() # (nbands, nimages)
        if band_frames:
            plot_mep(self.directory, band_frames[-1])
            #plot_bands(self.directory, images, nimages=nimages_per_band)
            write(self.directory/"nebtraj.xyz", itertools.chain(*band_frames))
            # --
            last_band = band_frames[-1]
            energies = [a.get_potential_energy() for a in last_band]
            imax = 1 + np.argsort(energies[1:-1])[-1]
            print(f"imax: {imax}")
            # NOTE: maxforce in cp2k is norm(atomic_forces)
            maxfrc = np.max(last_band[imax].get_forces(apply_constraint=True))
            print(f"maxfrc: {maxfrc}")
        else:
            last_band = []

        return last_band
    
    def _backup(self):
        """"""

        return
    
    def _irun(self, structures: List[Atoms], *args, **kwargs):
        """"""
        images = self._align_structures(structures)
        write(self.directory/"images.xyz", images)

        atoms = images[0] # use the initial state
        try:
            run_params = self.setting.get_run_params(**kwargs)
            run_params.update(**self.setting.get_init_params())

            # - update input template
            # GLOBAL section is automatically created...
            # FORCE_EVAL.(METHOD, POISSON)
            #inp = self.calc.parameters.inp # string
            #sec = parse_input(inp)
            #for (k, v) in run_params["pairs"]:
            #    sec.add_keyword(k, v)
            #for (k, v) in run_params["run_pairs"]:
            #    sec.add_keyword(k, v)

            # -- check constraint
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
    
    def read_trajectory(self, *args, **kwargs):
        """

        NOTE: Fixed atoms have zero forces.

        """
        self._debug(f"***** read_trajectory *****")
        self._debug(f"{str(self.directory)}")


        #ini_atoms = read(self.directory/"00"/"POSCAR")
        #fin_atoms = read(self.directory/f"{str(self.setting.nimages-1).zfill(2)}"/"POSCAR")
        images = read(self.directory/"images.xyz", ":")
        ini_atoms, fin_atoms = images[0], images[-1]

        frames_ = []
        for i in range(1, self.setting.nimages-1):
            curr_frames = read(self.directory/f"{str(i).zfill(2)}"/"OUTCAR", ":")
            frames_.append(curr_frames)
        
        nsteps = len(frames_[0])
        frames = []
        for i in range(nsteps):
            curr_frames = [ini_atoms] + [frames_[j][i] for j in range(self.setting.nimages-2)] + [fin_atoms]
            frames.append(curr_frames)

        return frames

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()

        for k, v in dataclasses.asdict(self.setting).items():
            if not k.startswith("_"):
                params[k] = v

        return params


if __name__ == "__main__":
    ...
