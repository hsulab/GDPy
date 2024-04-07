#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
import os
import re
import pathlib
import traceback

from typing import Union, List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp import Vasp

from .. import read_sort, resort_atoms_with_spc, run_ase_calculator
from .string import AbstractStringReactor, StringReactorSetting


#: Ase-vasp sort fname.
ASE_VASP_SORT_FNAME: str = "ase-sort.dat"


def read_vaspout(
    lines: List[str],
) -> int:
    """"""
    pattern = re.compile("[0-9]+ F=")

    steps = [0]
    for line in lines:
        m = pattern.match(line.strip())
        if m:
            step = int(m.group().split()[0])
            steps.append(step)
        else:
            ...
    # print(f"{steps =}")

    return max(steps)


@dataclasses.dataclass
class VaspStringReactorSetting(StringReactorSetting):

    backend: str = "vasp"

    #: Number of tasks/processors/cpus for each image.
    ntasks_per_image: int = 1

    def __post_init__(self):
        """"""
        self._internals.update(
            # ---
            ibrion=3,
            potim=0,
            isif=2,
            # ---
            lclimb=self.climb,
            ichain=0,
            images=self.nimages - 2,
            iopt=1,
            spring=-5,
        )

        return

    def get_run_params(self, *args, **kwargs):
        """"""
        # - convergence criteria
        fmax_ = kwargs.get("fmax", self.fmax)
        steps_ = kwargs.get("steps", self.steps)

        run_params = dict(
            constraint=kwargs.get("constraint", self.constraint),
            ediffg=fmax_ * -1.0,
            nsw=steps_,
        )

        return run_params


class VaspStringReactor(AbstractStringReactor):

    name: str = "vasp"

    traj_name: str = "01/OUTCAR"

    def __init__(
        self,
        calc: Vasp = None,
        params: dict = {},
        ignore_convergence: bool = False,
        directory: Union[str, pathlib.Path] = "./",
        *args,
        **kwargs,
    ) -> None:
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
            vasprun = self.directory / "01" / "OUTCAR"
            if vasprun.exists() and vasprun.stat().st_size != 0:
                temp_frames = read(vasprun, ":")
                try:
                    _ = temp_frames[0].get_forces()
                except:
                    verified = False
            else:
                verified = False
        else:
            verified = False

        return verified

    def _irun(self, structures: List[Atoms], ckpt_wdir=None, *args, **kwargs):
        """"""
        try:
            # --
            run_params = self.setting.get_run_params(**kwargs)
            run_params.update(**self.setting.get_init_params())

            if ckpt_wdir is None:  # start from the scratch
                images = self._align_structures(structures, run_params)
                write(self.directory / "images.xyz", images)
            else:
                # - update structures
                rep_dirs = sorted(
                    [x.name for x in sorted(self.directory.glob(r"[0-9][0-9]"))]
                )

                frames_ = []
                for x in rep_dirs[1:-1]:
                    frames_.append(read(self.directory / x / "OUTCAR", ":"))
                nframes = min([len(x) for x in frames_])
                assert nframes > 0, "At least one step finished before resume..."
                intermediates_ = [x[nframes - 1] for x in frames_]
                images = [structures[0]] + intermediates_ + [structures[-1]]

                run_params.update(steps=self.setting.steps - nframes)

            # - update input
            self.calc.set(**run_params)

            atoms = images[0]
            atoms.calc = self.calc

            # -- write input files
            self.calc.write_input(atoms)
            if (self.directory / "POSCAR").exists():
                os.remove(self.directory / "POSCAR")

            # -- add replica information
            for i, a in enumerate(images):
                rep_dir = self.directory / str(i).zfill(2)
                # It has already been created when images are written.
                # If the previous run has no outputs, we just overwrite everything.
                rep_dir.mkdir(exist_ok=True)
                write(
                    rep_dir / "POSCAR",
                    a[self.calc.sort],
                    symbol_count=self.calc.symbol_count,
                )

            # - run calculation
            run_ase_calculator("vasp", atoms.calc.command, self.directory)

        except Exception as e:
            self._debug(e)
            self._debug(traceback.print_exc())

        return

    def read_convergence(self, *args, **kwargs):
        """Check whether vasp-neb is converged.

        The convergence meets when either the required force is reached or
        the maximum steps exceed.

        """
        converged = super().read_convergence(*args, **kwargs)

        self._print(f"{self.directory =}")
        vaspout_fpath = self.directory / "vasp.out"
        if vaspout_fpath.exists():
            with open(self.directory / "vasp.out", "r") as fopen:
                lines = fopen.readlines()

            steps = read_vaspout(lines)
            if steps >= self.setting.steps:
                converged = True

            for line in lines:
                if "reached required accuracy" in line:
                    converged = True
                    break
        else:
            ...

        return converged

    def _read_a_single_trajectory(self, wdir, *args, **kwargs):
        """

        NOTE: Fixed atoms have zero forces.

        """
        self._debug(f"***** read_trajectory *****")
        self._debug(f"{str(wdir)}")

        images = read(wdir / "images.xyz", ":")
        ini_atoms, fin_atoms = images[0], images[-1]

        # TODO: energy and forces of IS and FS?
        calc = SinglePointCalculator(
            ini_atoms,
            energy=ini_atoms.info["energy"],
            forces=np.zeros((len(ini_atoms), 3)),
        )
        ini_atoms.calc = calc
        calc = SinglePointCalculator(
            fin_atoms,
            energy=fin_atoms.info["energy"],
            forces=np.zeros((len(fin_atoms), 3)),
        )
        fin_atoms.calc = calc

        # - read OUTCARs
        if (self.directory / ASE_VASP_SORT_FNAME).exists():
            sort, resort = read_sort(self.directory, ASE_VASP_SORT_FNAME)
        else:
            natoms = len(ini_atoms)
            sort, resort = list(range(natoms)), list(range(natoms))

        frames_ = []
        for i in range(1, self.setting.nimages - 1):
            curr_frames = read(wdir / f"{str(i).zfill(2)}" / "OUTCAR", ":")
            sorted_frames = []
            for a in curr_frames:
                sorted_atoms = resort_atoms_with_spc(
                    a, resort, "vasp", print_func=self._print, debug_func=self._debug
                )
                sorted_frames.append(sorted_atoms)
            frames_.append(sorted_frames)

        # nframes may not consistent across replicas
        # due to unfinished calculations
        nframes_list = [len(x) for x in frames_]
        nsteps = min(nframes_list)

        frames = []
        for i in range(nsteps):
            curr_frames = (
                [ini_atoms]
                + [frames_[j][i] for j in range(self.setting.nimages - 2)]
                + [fin_atoms]
            )
            frames.append(curr_frames)

        return frames


if __name__ == "__main__":
    ...

