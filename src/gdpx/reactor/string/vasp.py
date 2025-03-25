#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
import os
import re
import traceback

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from gdpx.utils.cmdrun import run_ase_calculator
from gdpx.utils.strucopy import read_sort, resort_atoms_with_spc

from .string import BaseStringReactor, Controller, StringReactorSetting

#: Ase-vasp sort fname.
ASE_VASP_SORT_FNAME: str = "ase-sort.dat"


def read_vaspout(
    lines: list[str],
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
class BFGSMinimiser(Controller):

    name: str = "bfgs"

    def __post_init__(self):
        """"""
        maxstep = self.params.get("maxstep", 0.2)  # Ang
        assert maxstep is not None

        self.params.update(iopt=1, maxmove=maxstep)

        return


@dataclasses.dataclass
class CGMinimiser(Controller):

    name: str = "cg"

    def __post_init__(self):
        """"""
        maxstep = self.params.get("maxstep", 0.2)

        self.params.update(iopt=2, maxmove=maxstep)

        return


@dataclasses.dataclass
class FireMinimiser(Controller):

    name: str = "fire"

    def __post_init__(self):
        """"""
        dt = self.params.get("timestep", 0.1)

        maxstep = self.params.get("maxstep", 0.2)  # Ang
        assert maxstep is not None

        self.params.update(iopt=7, maxmove=maxstep, timestep=dt)

        return


@dataclasses.dataclass
class MDMinimiser(Controller):

    name: str = "mdmin"

    def __post_init__(self):
        """"""
        dt = self.params.get("timestep", 0.1)

        maxstep = self.params.get("maxstep", 0.2)
        assert maxstep is not None

        self.params.update(iopt=3, maxmove=maxstep, timestep=dt)

        return


controllers = dict(
    bfgs=BFGSMinimiser,
    cg=CGMinimiser,
    fire=FireMinimiser,
    mdmin=MDMinimiser,
    quickmin=MDMinimiser,  # alias for mdmin
)


@dataclasses.dataclass
class VaspStringReactorSetting(StringReactorSetting):

    backend: str = "vasp"

    controller: dict = dataclasses.field(default_factory=dict)

    #: Number of tasks/processors/cpus for each image.
    ntasks_per_image: int = 1

    def __post_init__(self):
        """"""
        _init_params = {}
        _init_params.update(**self.controller)

        if self.controller:
            cont_cls_name = self.controller.get("name", "mdmin")
            if cont_cls_name in controllers:
                cont_cls = controllers[cont_cls_name]
            else:
                raise RuntimeError(f"Unknown controller {cont_cls_name}.")
        else:
            cont_cls = controllers["mdmin"]

        cont = cont_cls(**_init_params)

        self._internals.update(
            # Parameters enable VTST optimisers.
            ibrion=3,
            potim=0,
            isif=2,
            # Parameters for the constant-volume NEB calculation.
            ichain=0,
            lclimb=self.climb,
            images=self.nimages - 2,
            spring=self.kspring * -1,
        )
        self._internals.update(**cont.params)

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


class VaspStringReactor(BaseStringReactor):

    name: str = "vasp"

    traj_name: str = "01/OUTCAR"

    setting_cls: type[StringReactorSetting] = VaspStringReactorSetting

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
                    verified = True
                except:
                    verified = False
            else:
                verified = False
        else:
            verified = False

        return verified

    def _irun(self, structures: list[Atoms], ckpt_wdir=None, *args, **kwargs):
        """"""
        # get params
        run_params = self.setting.get_run_params(**kwargs)
        run_params.update(**self.setting.get_init_params())

        if ckpt_wdir is None:  # start from the scratch
            self._print("interpolate input images...")
            images = self._align_structures(structures, run_params)
            # The energies are stored in the info dict.
            ini_ene = images[0].info["energy"]
            fin_ene = images[-1].info["energy"]
        else:
            self._print("update input images...")
            # Read images from OUTCARs
            rep_dirs = sorted(ckpt_wdir.glob(r"[0-9][0-9]"), key=lambda x: int(x.name))

            frames_ = []
            for x in rep_dirs[1:-1]:
                frames_.append(read(x / "OUTCAR", ":"))
            nframes_per_image = [len(x) for x in frames_]
            nframes = min(nframes_per_image)
            assert nframes > 0, "At least one step finished before resume..."
            intermediates_ = [x[nframes - 1] for x in frames_]

            # Sort atoms in images
            if (ckpt_wdir / ASE_VASP_SORT_FNAME).exists():
                sort, resort = read_sort(ckpt_wdir, ASE_VASP_SORT_FNAME)
            else:
                natoms = len(structures[0])
                sort, resort = list(range(natoms)), list(range(natoms))

            intermediates = []
            for a in intermediates_:
                sorted_atoms = resort_atoms_with_spc(a, resort, "vasp", print_func=self._print, debug_func=self._debug)
                intermediates.append(sorted_atoms)

            images = [structures[0]] + intermediates + [structures[-1]]

            # the param keys have been proprocessed to vasp ones
            run_params.update(nsw=self.setting.steps + 1 - nframes)

        # The energies are stored in the info dict both start from scratch or restart as
        # the initial state and the final state are from the input structures.
        ini_ene = images[0].info["energy"]
        fin_ene = images[-1].info["energy"]

        # Update some system-dependent parameters
        run_params.update(images=len(images) - 2)
        run_params.update(efirst=ini_ene, elast=fin_ene)

        # From scratch, constraint should be removed as vasp calc does have it.
        # From restart, constraint info has already been in OUTCAR.
        run_params.pop("constraint")

        write(self.directory / "images.xyz", images)

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

        # run calculation
        try:
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
            # energy=ini_atoms.info["energy"],
            energy=ini_atoms.get_potential_energy(),
            forces=np.zeros((len(ini_atoms), 3)),
        )
        ini_atoms.calc = calc
        calc = SinglePointCalculator(
            fin_atoms,
            # energy=fin_atoms.info["energy"],
            energy=fin_atoms.get_potential_energy(),
            forces=np.zeros((len(fin_atoms), 3)),
        )
        fin_atoms.calc = calc

        # - read OUTCARs
        if (self.directory / ASE_VASP_SORT_FNAME).exists():
            sort, resort = read_sort(self.directory, ASE_VASP_SORT_FNAME)
        else:
            natoms = len(ini_atoms)
            sort, resort = list(range(natoms)), list(range(natoms))

        nimages_per_band = int(np.loadtxt(self.directory / "nimages"))

        frames_ = []
        for i in range(1, nimages_per_band - 1):
            curr_frames = read(wdir / f"{str(i).zfill(2)}" / "OUTCAR", ":")
            sorted_frames = []
            for a in curr_frames:
                sorted_atoms = resort_atoms_with_spc(a, resort, "vasp", print_func=self._print, debug_func=self._debug)
                sorted_frames.append(sorted_atoms)
            frames_.append(sorted_frames)

        # nframes may not consistent across replicas
        # due to unfinished calculations
        nframes_list = [len(x) for x in frames_]
        nsteps = min(nframes_list)

        frames = []
        for i in range(nsteps):
            curr_frames = [ini_atoms] + [frames_[j][i] for j in range(nimages_per_band - 2)] + [fin_atoms]
            frames.append(curr_frames)

        return frames


if __name__ == "__main__":
    ...
