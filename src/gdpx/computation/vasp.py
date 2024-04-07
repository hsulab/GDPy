#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import io
import os
import re
import dataclasses
import pathlib
import tarfile
import traceback
from typing import Union, Optional, List, Tuple

import shutil

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.vasp import Vasp

from ..data.extatoms import ScfErrAtoms
from ..utils.strucopy import read_sort, resort_atoms_with_spc
from ..utils.cmdrun import run_ase_calculator
from .driver import AbstractDriver, DriverSetting, Controller


"""Driver for VASP."""
#: Ase-vasp resort fname.
ASE_VASP_SORT_FNAME: str = "ase-sort.dat"


def read_outcar_scf(lines: List[str]) -> Tuple[int, float]:
    """"""
    nelm, ediff = None, None
    for line in lines:
        if line.strip().startswith("NELM"):
            nelm = int(line.split()[2][:-1])
        if line.strip().startswith("EDIFF"):
            ediff = float(line.split()[2])
        if nelm is not None and ediff is not None:
            break
    else:
        ...  # TODO: raise an error?

    return nelm, ediff


def read_oszicar(lines: List[str], nelm: int, ediff: float) -> List[bool]:
    """"""
    convergence = []
    content = ""
    for line in lines:
        start = line.strip().split()[0]
        if start == "N":
            content = ""
            continue
        if start.isdigit():
            scfsteps = [int(s.split()[1]) for s in content.strip().split("\n")]
            num_scfsteps = len(scfsteps)
            assert num_scfsteps == scfsteps[-1]
            enediffs = [float(s.split()[3]) for s in content.strip().split("\n")]
            is_converged = num_scfsteps < nelm or np.fabs(enediffs[-1]) <= ediff
            convergence.append(is_converged)
        content += line

    # If the last SCF is not finished,
    # there is no content to check

    return convergence


@dataclasses.dataclass
class LangevinThermostat(Controller):

    name: str = "langevin"

    def __post_init__(
        self,
    ):
        """"""
        friction = self.params.get("friction", None)  # fs^-1
        assert friction is not None

        self.conv_params = dict(langevin_gamma=friction * 1e3)  # ps^-1

        return


@dataclasses.dataclass
class NoseHooverThermostat(Controller):

    name: str = "nose_hoover"

    def __post_init__(
        self,
    ):
        """"""
        # FIXME: convert taut to smass
        smass = self.params.get("taut", 0.0)  # or smass?
        assert smass >= 0, "NoseHoover-NVT needs positive SMASS."

        self.conv_params = dict(smass=smass)

        return


@dataclasses.dataclass
class ParrinelloRahmanBarostat(Controller):

    name: str = "parrinello_rahman"

    def __post_init__(
        self,
    ):
        """"""
        # FIXME: convert taut to smass
        smass = self.params.get("taut", 0.0)  # or smass?
        assert smass >= 0, f"{self.name} needs positive SMASS."

        friction = self.params.get("friction", None)  # fs^-1
        assert friction is not None

        friction_lattice = self.params.get("friction_lattice", None)  # fs^-1
        assert friction_lattice is not None

        pmass = self.params.get("pmass", None)  # a.m.u
        assert pmass > 0.0

        self.conv_params = dict(
            smass=smass,
            langevin_gamma=friction * 1e3,  # array, ps^-1
            langevin_gamma_l=friction_lattice * 1e3,  # real, ps^-1
            pmass=1000.0,
        )

        return


controllers = dict(
    langevin_nvt=LangevinThermostat,
    nose_hoover_nvt=NoseHooverThermostat,
    parrinello_rahman_npt=ParrinelloRahmanBarostat,
)


@dataclasses.dataclass
class VaspDriverSetting(DriverSetting):

    # - md setting
    ensemble: str = "nve"

    # - driver detailed controller setting
    controller: dict = dataclasses.field(default_factory=dict)

    # TODO: move below to controller?
    # fix_cm: bool = False

    # - min setting
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
                # raise ValueError(f"Unknown minimisation {self.min_style} for vasp".)
                ...

            self._internals.update(ibrion=ibrion, potim=self.maxstep)

        # -- cmin: cell minimisation
        if self.task == "cmin":
            if self.min_style == "bfgs":
                ibrion = 1
            elif self.min_style == "cg":
                ibrion = 2
            else:
                # raise ValueError(f"Unknown minimisation {self.min_style} for vasp".)
                ...

            self._internals.update(isif=3, ibrion=ibrion, potim=self.maxstep)

        if self.task == "md":
            # NOTE: Always use Selective Dynamics and MDALAGO
            #       since it properly treats the DOF and velocities
            # some general
            # if self.velocity_seed is None:
            #     self.velocity_seed = np.random.randint(0, 10000)
            # random_seed = [self.velocity_seed, 0, 0]
            self._internals.update(
                # -- Some shared parameters, every MD needs these!!
                velocity_seed=self.velocity_seed,
                ignore_atoms_velocities=self.ignore_atoms_velocities,
                ibrion=0,
                isif=0,
                potim=self.timestep,  # fs
                random_seed=None,  # NOTE: init later in driver run
            )

            if self.ensemble == "nve":
                _init_md_params = dict()
                _init_md_params.update(
                    mdalgo=2,
                    smass=-3,
                )
            elif self.ensemble == "nvt":
                # We need keywords: TEBEG and TEEND.
                # Also, only consistent-temperature simulation is supported.
                _init_md_params = dict(tebeg=self.temp, teend=self.temp)

                if self.controller is not None:
                    thermo_cls_name = self.controller["name"] + "_" + self.ensemble
                    thermo_cls = controllers[thermo_cls_name]
                else:
                    thermo_cls = LangevinThermostat
                thermostat = thermo_cls(**self.controller)

                if thermostat.name == "langevin":
                    # MDALGO, LANGEVIN_GAMMA
                    _init_md_params.update(
                        mdalgo=3,
                    )
                    _init_md_params.update(**thermostat.conv_params)
                elif thermostat.name == "nose_hoover":
                    # MDALGO, SMASS
                    _init_md_params.update(
                        mdalgo=2,
                    )
                    _init_md_params.update(**thermostat.conv_params)
                else:
                    raise RuntimeError(f"Unknown {thermostat =}.")
            elif self.ensemble == "npt":
                # We need keywords: TEBEG, TEEND, PSTRESS
                _init_md_params = dict(
                    tebeg=self.temp,
                    teend=self.temp,
                    # pressure unit 1 GPa  = 10 kBar
                    #               1 kBar = 1000 bar = 10^8 Pa
                    pstress=1e-3 * self.press,  # vasp uses kB
                )

                if self.controller is not None:
                    baro_cls_name = self.controller["name"] + "_" + self.ensemble
                    baro_cls = controllers[baro_cls_name]
                else:
                    baro_cls = ParrinelloRahmanBarostat
                barostat = baro_cls(**self.controller)

                if barostat.name == "parrinello_rahman":
                    _init_md_params.update(mdalgo=3)
                    _init_md_params.update(**barostat.conv_params)
                else:
                    raise RuntimeError(f"Unknown {barostat =}.")
            else:
                raise NotImplementedError(f"{self.md_style} is not supported yet.")

            self._internals.update(**_init_md_params)

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
                ediffg = -1.0 * fmax_
            else:
                ediffg = -5e-2

        steps_ = kwargs.get("steps", self.steps)
        nsw = steps_

        run_params = dict(
            constraint=kwargs.get("constraint", self.constraint), ediffg=ediffg, nsw=nsw
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
        "INCAR",
        "POSCAR",
        "KPOINTS",
        "POTCAR",
        "OSZICAR",
        "OUTCAR",
        "CONTCAR",
        "vasprun.xml",
        "REPORT",
    ]

    def __init__(self, calc: Vasp, params: dict, directory="./", *args, **kwargs):
        """"""
        super().__init__(calc, params, directory=directory, *args, **kwargs)

        self.setting = VaspDriverSetting(**params)

        return

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """Check whether there is a previous calculation in the `self.directory`."""
        verified = True
        if self.directory.exists():
            vasprun = self.directory / "vasprun.xml"
            if vasprun.exists() and vasprun.stat().st_size != 0:
                temp_frames = read(vasprun, ":")
                try:
                    _ = temp_frames[0].get_forces()
                except:  # `RuntimeError: Atoms object has no calculator.`
                    verified = False
            else:
                verified = False
        else:
            verified = False

        return verified

    def _irun(
        self,
        atoms: Atoms,
        ckpt_wdir=None,
        cache_traj: Optional[List[Atoms]] = None,
        *args,
        **kwargs,
    ):
        """"""
        try:
            if ckpt_wdir is None:  # start from the scratch
                # - merge params
                run_params = self.setting.get_run_params(**kwargs)
                run_params.update(**self.setting.get_init_params())
                run_params["system"] = self.directory.name

                # FIXME: Init velocities?
                prev_ignore_atoms_velocities = run_params.pop(
                    "ignore_atoms_velocities", False
                )
                velocity_seed = run_params.pop("velocity_seed", None)

                if self.setting.task == "md":
                    vasp_random_seed = [self.random_seed, 0, 0]
                    self._print(f"MD Driver's velocity_seed: vasp-{vasp_random_seed}")
                    self._print(f"MD Driver's rng: vasp-{vasp_random_seed}")
                    run_params["random_seed"] = vasp_random_seed
                    # TODO: use external velocities?
                else:
                    ...

                # - update some system-dependant params
                if "langevin_gamma" in run_params:
                    ntypes = len(set(atoms.get_chemical_symbols()))
                    run_params["langevin_gamma"] = [
                        run_params["langevin_gamma"]
                    ] * ntypes

                # FIXME: LDA+U

                # - constraint
                self._preprocess_constraints(atoms, run_params)

                self.calc.set(**run_params)
                atoms.calc = self.calc
                # NOTE: ASE VASP does not write velocities and thermostat to POSCAR
                #       thus we manually call the function to write input files and
                #       run the calculation
                self.calc.write_input(atoms)
            else:
                self.calc.read_incar(ckpt_wdir / "INCAR")  # read previous incar
                if cache_traj is None:
                    traj = self.read_trajectory()
                else:
                    traj = cache_traj
                nframes = len(traj)
                assert nframes > 0, "VaspDriver restarts with a zero-frame trajectory."
                dump_period = 1  # since we read vasprun.xml, every frame is dumped
                target_steps = self.setting.get_run_params(*args, **kwargs)["nsw"]
                if target_steps > 0:  # not a spc
                    # BUG: ...
                    steps = target_steps + dump_period - nframes * dump_period
                    assert (
                        steps > 0
                    ), f"Steps should be greater than 0. (steps = {steps})"
                    self.calc.set(nsw=steps)
                # NOTE: ASE VASP does not write velocities and thermostat to POSCAR
                #       thus we manually call the function to write input files and
                #       run the calculation
                # FIXME: Read random_seed in REPORT!!!
                self.calc.write_input(atoms)
                # To restart, velocities are always retained
                # if (self.directory/"CONTCAR").exists() and (self.directory/"CONTCAR").stat().st_size != 0:
                #    shutil.copy(self.directory/"CONTCAR", self.directory/"POSCAR")
                shutil.copy(ckpt_wdir / "CONTCAR", self.directory / "POSCAR")

            run_ase_calculator("vasp", self.calc.command, self.directory)

        except Exception as e:
            self._debug(f"Exception of {self.__class__.__name__} is {e}.")
            self._debug(
                f"Exception of {self.__class__.__name__} is {traceback.format_exc()}."
            )

        return

    def _read_a_single_trajectory(self, wdir, archive_path=None, *args, **kwargs):
        """"""
        oszicar_lines, outcar_lines = None, None
        if archive_path is None:
            # - read trajectory
            vasprun = wdir / "vasprun.xml"
            frames = read(vasprun, ":")
            with open(wdir / "OUTCAR") as fopen:
                outcar_lines = fopen.readlines()
            with open(wdir / "OSZICAR") as fopen:
                oszicar_lines = fopen.readlines()
        else:
            flags = [False, False, False]
            vasprun_name = str(
                (wdir / "vasprun.xml").relative_to(self.directory.parent)
            )
            oszicar_name = str((wdir / "OSZICAR").relative_to(self.directory.parent))
            outcar_name = str((wdir / "OUTCAR").relative_to(self.directory.parent))
            with tarfile.open(archive_path, "r:gz") as tar:
                for tarinfo in tar:
                    if tarinfo.name == vasprun_name:
                        fobj = io.StringIO(
                            tar.extractfile(tarinfo.name).read().decode()
                        )
                        frames = read(fobj, ":", format="vasp-xml")
                        fobj.close()
                        flags[0] = True
                    if tarinfo.name == oszicar_name:
                        fobj = io.StringIO(
                            tar.extractfile(tarinfo.name).read().decode()
                        )
                        oszicar_lines = fobj.readlines()
                        fobj.close()
                        flags[1] = True
                    if tarinfo.name == outcar_name:
                        fobj = io.StringIO(
                            tar.extractfile(tarinfo.name).read().decode()
                        )
                        outcar_lines = fobj.readlines()
                        fobj.close()
                        flags[2] = True
                    if all(flags):
                        break
                else:  # TODO: if not find target traj?
                    ...

        # - read oszicar and outcar
        if outcar_lines is not None and oszicar_lines is not None:
            nelm, ediff = read_outcar_scf(outcar_lines)
            scf_convergences = read_oszicar(oszicar_lines, nelm, ediff)
            assert len(scf_convergences) == len(frames)
            for i, is_converged in enumerate(scf_convergences):
                if not is_converged:
                    frames[i] = ScfErrAtoms.from_atoms(frames[i])
                    self._print(f"ScfErrAtoms Step {i} @ {str(wdir)}")

        return frames

    def read_trajectory(
        self, add_step_info=True, archive_path=None, *args, **kwargs
    ) -> List[Atoms]:
        """Read trajectory in the current working directory.

        If the calculation failed, an empty atoms with errof info would be returned.

        """
        # - read structures
        # -- read backups
        prev_wdirs = []
        if archive_path is None:
            prev_wdirs = sorted(self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"))
        else:
            pattern = self.directory.name + "/" + r"[0-9][0-9][0-9][0-9][.]run"
            with tarfile.open(archive_path, "r:gz") as tar:
                for tarinfo in tar:
                    if tarinfo.isdir() and re.match(pattern, tarinfo.name):
                        prev_wdirs.append(tarinfo.name)
            prev_wdirs = [
                self.directory / pathlib.Path(p).name for p in sorted(prev_wdirs)
            ]
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
        # NOTE: Some spin systems may give different scf convergence on the same
        #       structure. Sometimes, the preivous failed but the next run converged,
        #       The concat below uses the previous one...
        traj_frames_, ntrajs = [], len(traj_list)
        if ntrajs > 0:
            traj_frames_.extend(traj_list[0])
            for i in range(1, ntrajs):
                assert np.allclose(
                    traj_list[i - 1][-1].positions, traj_list[i][0].positions
                ), f"Traj {i-1} and traj {i} are not consecutive."
                traj_frames_.extend(traj_list[i][1:])
        else:
            ...

        nframes = len(traj_frames_)
        natoms = len(traj_frames_[0])

        # - sort frames
        traj_frames = []
        if nframes > 0:
            if (self.directory / ASE_VASP_SORT_FNAME).exists():
                sort, resort = read_sort(self.directory)
            else:  # without sort file, use default order
                sort, resort = list(range(natoms)), list(range(natoms))
            for i, sorted_atoms in enumerate(traj_frames_):
                # NOTE: calculation with only one unfinished step does not have forces
                input_atoms = resort_atoms_with_spc(
                    sorted_atoms,
                    resort,
                    "vasp",
                    print_func=self._print,
                    debug_func=self._debug,
                )
                # if input_atoms is None:
                #    input_atoms = Atoms()
                #    input_atoms.info["error"] = str(self.directory)
                if input_atoms is not None:
                    if add_step_info:
                        input_atoms.info["step"] = i
                    traj_frames.append(input_atoms)
        else:
            ...

        return traj_frames


if __name__ == "__main__":
    ...
