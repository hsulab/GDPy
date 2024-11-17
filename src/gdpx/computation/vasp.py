#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
import io
import os
import pathlib
import re
import shutil
import tarfile
import traceback
from typing import List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import read, write
from ase.geometry import find_mic

from ..backend.vasp import read_report, read_outcar_scf, read_oszicar
from ..data.extatoms import ScfErrAtoms
from ..utils.cmdrun import run_ase_calculator
from ..utils.strucopy import read_sort, resort_atoms_with_spc
from .driver import AbstractDriver, Controller, DriverSetting


"""Driver for VASP."""
#: Ase-vasp resort fname.
ASE_VASP_SORT_FNAME: str = "ase-sort.dat"


@dataclasses.dataclass
class SinglePointController(Controller):

    name: str = "spc"

    def __post_init__(self):
        """"""

        self.conv_params = dict(nsw=0)

        return


@dataclasses.dataclass
class BFGSMinimiser(Controller):

    name: str = "bfgs"  # RMM-DIIS

    def __post_init__(self):
        """"""

        maxstep = self.params.get("maxstep", 0.1)

        self.conv_params = dict(ibrion=1, potim=maxstep)

        return


@dataclasses.dataclass
class CGMinimiser(Controller):

    name: str = "cg"

    def __post_init__(self):
        """"""

        maxstep = self.params.get("maxstep", 0.1)

        self.conv_params = dict(ibrion=2, potim=maxstep)

        return


@dataclasses.dataclass
class CellBFGSMinimiser(BFGSMinimiser):

    def __post_init__(self):
        """"""
        super().__post_init__()

        more_params = dict(isif=3)

        self.conv_params.update(**more_params)

        return


@dataclasses.dataclass
class CellCGMinimiser(CGMinimiser):

    def __post_init__(self):
        """"""
        super().__post_init__()

        more_params = dict(isif=3)

        self.conv_params.update(**more_params)

        return


@dataclasses.dataclass
class MDController(Controller):

    #: Controller name.
    name: str = "md"

    #: Timestep in fs.
    timestep: float = 1.0

    #: Temperature in Kelvin.
    temperature: float = 300.0

    #: Temperature in Kelvin.
    temperature_end: Optional[float] = None

    #: Pressure in bar.
    pressure: float = 1.0

    #: Pressure in Kelvin.
    pressure_end: Optional[float] = None

    #: Whether fix center of mass.
    fix_com: bool = True

    def __post_init__(self):
        """"""
        basic_params = dict(
            ibrion=0,
            isif=0,
            potim=self.timestep,
            random_seed=None,  # init later in driver run
            tebeg=self.temperature,
        )
        # We need keywords: TEBEG and TEEND.
        if self.temperature_end is not None:
            basic_params.update(teend=self.temperature_end)

        self.conv_params = basic_params

        return


@dataclasses.dataclass
class Verlet(MDController):

    name: str = "verlet"

    def __post_init__(self):
        """"""
        more_params = dict(
            mdalgo=2,
            smass=3,
        )

        self.conv_params.update(**more_params)

        return


@dataclasses.dataclass
class LangevinThermostat(MDController):

    name: str = "langevin"

    def __post_init__(
        self,
    ):
        """"""
        friction = self.params.get("friction", None)  # fs^-1
        friction *= 1e3  # ps^-1
        assert friction is not None

        # MDALGO, LANGEVIN_GAMMA
        more_params = dict(
            mdalgo=3,
            langevin_gamma=friction,
        )

        self.conv_params.update(**more_params)

        return


@dataclasses.dataclass
class NoseHooverThermostat(MDController):

    name: str = "nose_hoover"

    def __post_init__(self):
        """"""
        # FIXME: convert taut to smass
        smass = self.params.get("Tdamp", 0.0)  # or smass?
        assert smass >= 0, "NoseHoover-NVT needs positive SMASS."

        # MDALGO, SMASS
        more_params = dict(mdalgo=2, smass=smass)

        self.conv_params.update(**more_params)

        return


@dataclasses.dataclass
class ParrinelloRahmanBarostat(MDController):

    name: str = "parrinello_rahman"

    def __post_init__(self):
        """"""
        # FIXME: convert taut to smass
        smass = self.params.get("Tdamp", 0.0)  # or smass?
        assert smass >= 0, f"{self.name} needs positive SMASS."

        friction = self.params.get("friction", None)  # fs^-1
        friction *= 1e3  # array, ps^-1
        assert friction is not None

        # FIXME: convert taut to pmass
        pmass = self.params.get("Pdamp", 0.0)  # or smass?
        assert pmass >= 0, f"{self.name} needs positive PMASS."

        friction_lattice = self.params.get("friction_lattice", None)  # fs^-1
        friction_lattice *= 1e3  # real, ps^-1
        assert friction_lattice is not None

        pmass = self.params.get("pmass", None)  # a.m.u
        assert pmass > 0.0

        more_params = dict(
            smass=smass,
            langevin_gamma=friction,
            langevin_gamma_l=friction_lattice,
            pmass=pmass,
            # pressure unit 1 GPa  = 10 kBar
            #               1 kBar = 1000 bar = 10^8 Pa
            pstress=1e-3 * self.pressure,  # vasp uses kB
        )
        assert (
            self.pressure_end is None
        ), "VASP does not support NPT with changing pressure."

        self.conv_params.update(**more_params)

        return


controllers = dict(
    # - spc
    single_point_spc=SinglePointController,
    # - min
    bfgs_min=BFGSMinimiser,
    cg_min=CGMinimiser,
    # - cmin
    bfgs_cmin=CellBFGSMinimiser,
    cg_cmin=CellCGMinimiser,
    # - md
    langevin_nvt=LangevinThermostat,
    nose_hoover_nvt=NoseHooverThermostat,
    parrinello_rahman_npt=ParrinelloRahmanBarostat,
)

default_controllers = dict(
    spc=SinglePointController,
    min=CGMinimiser,
    cmin=CGMinimiser,
    nve=Verlet,
    nvt=LangevinThermostat,
    npt=ParrinelloRahmanBarostat,
)


@dataclasses.dataclass
class VaspDriverSetting(DriverSetting):

    #: MD ensemble.
    ensemble: str = "nve"

    #: Driver detailed controller setting.
    controller: dict = dataclasses.field(default_factory=dict)

    #: Whether fix com to the its initial position.
    fix_com: bool = False

    #: Energy tolerance in minimisation, 1e-5 [eV].
    emax: Optional[float] = None

    #: Force tolerance in minimisation, 5e-2 eV/Ang.
    fmax: Optional[float] = 0.05

    def __post_init__(self):
        """Convert parameters into driver-specific ones.

        These parameters are frozen when the driver is initialised.

        """
        # - update internals that are specific for each calculator...

        _init_params = {}
        if self.task == "min":
            suffix = self.task
        elif self.task == "cmin":
            suffix = self.task
        elif self.task == "md":
            # NOTE: Always use Selective Dynamics and MDALAGO
            #       since it properly treats the DOF and velocities
            suffix = self.ensemble
            _init_params.update(
                temperature=self.temp, temperature_end=self.temp,
                pressure=self.press, pressure_end=self.pend
            )
        elif self.task == "freq":
            # ibrion, nfree, potim
            raise NotImplementedError("")
        else:
            raise RuntimeError(f"Unknown VASP task `{self.task}`.")

        if self.controller:
            cont_cls_name = self.controller["name"] + "_" + suffix
            if cont_cls_name in controllers:
                cont_cls = controllers[cont_cls_name]
            else:
                raise RuntimeError(f"Unknown controller {cont_cls_name}.")
        else:
            cont_cls = default_controllers[suffix]

        _init_params.update(**self.controller)
        cont = cont_cls(**_init_params)

        self._internals.update(**cont.conv_params)

        return

    def get_run_params(self, *args, **kwargs):
        """"""
        # convergence criteria
        fmax_ = kwargs.get("fmax", self.fmax)
        emax_ = kwargs.get("emax", self.emax)

        # emax is prioritised
        if emax_ is not None:
            ediffg = emax_ 
        else:
            if fmax_ is not None:
                ediffg = -1.0 * fmax_
            else:
                raise RuntimeError(f"VASP fmax should not be `{fmax_}`.")

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

    #: Class for setting.
    setting_cls: type[DriverSetting] = VaspDriverSetting

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """Check whether there is a previous calculation in the `self.directory`."""
        verified = True
        if self.directory.exists():
            prev_wdirs = sorted(self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"))
            if not prev_wdirs:  # no previous checkpoints
                vasprun = self.directory / "vasprun.xml"
                if vasprun.exists() and vasprun.stat().st_size != 0:
                    try:
                        # `xml.etree.ElementTree.ParseError`
                        # if vasprun.xml does not have a complete finshed single-point-calculation
                        temp_atoms = read(vasprun, "0")
                        # `RuntimeError: Atoms object has no calculator.`
                        _ = temp_atoms.get_forces()
                    except:
                        verified = False
                else:
                    verified = False
            else:  # TODO: verify the previous checkpoint?
                verified = True
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
        if ckpt_wdir is None:  # start from the scratch
            # - merge params
            run_params = self.setting.get_run_params(**kwargs)
            run_params.update(**self.setting.get_init_params())
            run_params["system"] = self.directory.name

            # FIXME: Init velocities?
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
                run_params["langevin_gamma"] = [run_params["langevin_gamma"]] * ntypes

            # FIXME: LDA+U

            # - constraint
            self._preprocess_constraints(atoms, run_params)

            self.calc.set(**run_params)

            # check dipole correction and set dipole as COM if enabled
            use_dipole_correction = self.calc.int_params["idipol"]
            if use_dipole_correction is not None and use_dipole_correction in [
                1,
                2,
                3,
                4,
            ]:
                assert self.calc.bool_params[
                    "ldipol"
                ], f"Use dipole correction {use_dipole_correction} but LDIPOL is False."
                # TODO: Check whether the scaled COM is wrapped?
                dipole_centre = atoms.get_center_of_mass(scaled=True)
                self.calc.set(dipol=dipole_centre)

            # NOTE: ASE VASP does not write velocities and thermostat to POSCAR
            #       thus we manually call the function to write input files and
            #       run the calculation
            atoms.calc = self.calc
            self.calc.write_input(atoms)
            if self.setting.task == "cmin":  # TODO: NPT simulation
                # We need POSCAR in direct coordinates to deal with constraints
                write(
                    self.directory / "POSCAR",
                    self.calc.atoms_sorted,
                    symbol_count=self.calc.symbol_count,
                    direct=True,
                )
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
                # NOTE: vasp md is not consecutive, see read_trajectory
                if self.setting.task == "min":
                    steps = target_steps + dump_period - nframes * dump_period
                elif self.setting.task == "md":
                    steps = target_steps + dump_period - nframes * dump_period - 1
                else:
                    ...
                assert steps > 0, f"Steps should be greater than 0. (steps = {steps})"
                self.calc.set(nsw=steps)
            # NOTE: ASE VASP does not write velocities and thermostat to POSCAR
            #       thus we manually call the function to write input files and
            #       run the calculation
            if self.setting.task == "md":
                # read random_seed from REPORT
                with open(ckpt_wdir / "REPORT", "r") as fopen:
                    lines = fopen.readlines()
                    report_random_seeds = read_report(lines)
                self._print(f"{report_random_seeds.shape =}")
                # FIXME: The nframes is the number of frames of the entire trajectory.
                # assert report_random_seeds.shape[0] == nframes+1, "Inconsistent number of frames and number of random_seeds."
                self.calc.set(random_seed=report_random_seeds[-1].tolist())
            else:
                ...
            self.calc.write_input(atoms)
            # To restart, velocities are always retained
            # if (self.directory/"CONTCAR").exists() and (self.directory/"CONTCAR").stat().st_size != 0:
            #    shutil.copy(self.directory/"CONTCAR", self.directory/"POSCAR")
            shutil.copy(ckpt_wdir / "CONTCAR", self.directory / "POSCAR")

        # TODO: Make gdpx process exists when calculation failed,
        #       which maybe due to machine error and should be checked by users manually
        try:
            run_ase_calculator("vasp", self.calc.command, self.directory)
        except Exception as e:
            self._debug(f"Exception of {self.__class__.__name__} is {e}.")
            self._debug(
                f"Exception of {self.__class__.__name__} is {traceback.format_exc()}."
            )
            # TODO: Deal with different exceptions...
            # If CalculationFailed and no outputs, it may not have an appropriate
            # caculation environment...

        return

    def _read_a_single_trajectory(self, wdir, archive_path=None, *args, **kwargs):
        """"""
        oszicar_lines, outcar_lines = None, None
        if archive_path is None:
            # - read trajectory
            vasprun = wdir / "vasprun.xml"
            if vasprun.exists() and vasprun.stat().st_size != 0:
                try:  # Sometimes there are some outputs in vasprun but not a complete structure.
                    frames = read(vasprun, ":")
                    with open(wdir / "OUTCAR") as fopen:
                        outcar_lines = fopen.readlines()
                    with open(wdir / "OSZICAR") as fopen:
                        oszicar_lines = fopen.readlines()
                except:
                    frames = []
            else:
                frames = []
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
                else:
                    if not flags[0]:  # check if vasprun exists
                        frames = []
                    else:
                        self._print(f"FILE FLAG AT {str(wdir)} IS {flags}.")

        # - read oszicar and outcar
        if outcar_lines is not None and oszicar_lines is not None:
            nelm, ediff = read_outcar_scf(outcar_lines)
            scf_convergences = read_oszicar(oszicar_lines, nelm, ediff)
            num_scfconvs, num_frames = len(scf_convergences), len(frames)
            self._debug(f"{num_scfconvs =} {num_frames =}")
            if num_scfconvs == num_frames:
                # assert len(scf_convergences) == len(
                #     frames
                # ), f"Failed to read OUTCAR in {str(self.directory)}. OSZICAR {oszicar_lines}."
                ...
            elif num_scfconvs == num_frames - 1:
                # The LAST SCF failed due to some error, for example, too small distance
                # So we manually set conv to false for the last step and also
                # we set frames energy and forces to a very large value...
                # FIXME: We need check whether the last step unfinished is due to exceed wall time
                #        or some other errors... Othwise, a normal structure will be considered as an error.
                scf_convergences.append(False)
                # FIXME: Use a new CustomAtoms object to deal with this?
                from ase.calculators.singlepoint import SinglePointCalculator

                calc = SinglePointCalculator(
                    frames[-1],
                    energy=1e8,
                    free_energy=1e8,
                    forces=1e8 * np.ones(frames[-1].positions.shape),
                )
                frames[-1].calc = calc
            else:
                raise RuntimeError(
                    f"Failed to read OUTCAR in {str(self.directory)}. OSZICAR {oszicar_lines}."
                )
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

        # Try to read the latest outputs.
        # If the latest calculation is not finished/converged, the ouputs will be moved to
        # a new folder 000x.run and an empty trajectory should be return.
        # Even though vasprun file may be empty, the read can give a empty list...
        curr_frames = self._read_a_single_trajectory(
            self.directory, archive_path=archive_path
        )
        if not curr_frames:  # empty trajectory
            ...
        else:
            traj_list.append(curr_frames)

        # -- concatenate
        # NOTE: Some spin systems may give different scf convergence on the same
        #       structure. Sometimes, the preivous failed but the next run converged,
        #       The concat below uses the previous one...
        traj_frames_, ntrajs = [], len(traj_list)
        if ntrajs > 0:
            traj_frames_.extend(traj_list[0])
            if self.setting.task == "min":
                for i in range(1, ntrajs):
                    # FIXME: ase complete_cell bug?
                    prev_box = traj_list[i - 1][-1].get_cell(complete=True)
                    curr_box = traj_list[i][0].get_cell(complete=True)
                    assert np.allclose(
                        prev_box, curr_box
                    ), f"Traj {i-1} and traj {i} are not consecutive in cell."

                    prev_pos = traj_list[i - 1][-1].positions
                    curr_pos = traj_list[i][0].positions
                    pos_vec, _ = find_mic(
                        prev_pos - curr_pos, traj_list[i - 1][-1].get_cell()
                    )
                    assert np.allclose(
                        pos_vec, np.zeros(pos_vec.shape)
                    ), f"Traj {i-1} and traj {i} are not consecutive."
                    traj_frames_.extend(traj_list[i][1:])
            elif self.setting.task == "md":
                # NOTE: vasp md restart is not from the last frame
                #       as it addes the velocities in contcar to get a new frame to restart.
                for i in range(1, ntrajs):
                    traj_frames_.extend(traj_list[i][:])
            else:
                ...
        else:
            ...

        nframes = len(traj_frames_)

        # - sort frames
        traj_frames = []
        if nframes > 0:
            num_atoms = len(traj_frames_[0])
            if (self.directory / ASE_VASP_SORT_FNAME).exists():
                sort, resort = read_sort(self.directory)
            else:  # without sort file, use default order
                sort, resort = list(range(num_atoms)), list(range(num_atoms))
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
