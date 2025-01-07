#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
import pathlib
import traceback
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.calculators.cp2k import parse_input

from ..backend.cp2k import (
    Cp2kFileIO,
    read_cp2k_convergence,
    read_cp2k_outputs,
    read_cp2k_spc,
    read_cp2k_spc_convergence,
)
from ..builder.constraints import parse_constraint_info
from ..data.extatoms import ScfErrAtoms
from .driver import AbstractDriver, Controller, DriverSetting


@dataclasses.dataclass
class SinglePointController(Controller):

    #: Controller name.
    name: str = "single_point"

    #: Save checkpoint every.
    ckpt_period: int = 100

    def __post_init__(self):
        """"""
        # FIXME: We must use list here as the section may have duplicate keys
        #        due to current terrible parser.
        # NOTE: Add more parameters as tuples.
        self.conv_params = [
            ("GLOBAL", "RUN_TYPE ENERGY_FORCE"),
            ("FORCE_EVAL/PRINT/FORCES", "_SECTION_PARAMETERS_ ON"),
        ]

        return


@dataclasses.dataclass
class FrequencyController(Controller):

    #: Controller name.
    name: str = "frequency"

    #: Save checkpoint every.
    ckpt_period: int = 100

    #: The finite difference.
    maxstep: float = 0.01

    def __post_init__(self):
        """"""
        self.conv_params = [
            ("GLOBAL", "RUN_TYPE VIBRATIONAL_ANALYSIS"),
            ("VIBRATIONAL_ANALYSIS", f"DX {self.maxstep}"),
            ("VIBRATIONAL_ANALYSIS", "NPROC_REP 16"),
            # ("VIBRATIONAL_ANALYSIS/MODE_SELECTIVE", "INITIAL_GUESS ATOMIC"),
            # ("VIBRATIONAL_ANALYSIS/MODE_SELECTIVE", "EPS_NORM 1.0E-5"),
            # ("VIBRATIONAL_ANALYSIS/MODE_SELECTIVE", "EPS_MAX_VAL 1.0E-6"),
        ]

        return


@dataclasses.dataclass
class MotionController(Controller):

    #: Controller name.
    name: str = "motion"

    #: Save checkpoint every.
    ckpt_period: int = 100

    def __post_init__(self):
        """"""
        # FIXME: We must use list here as the section may have duplicate keys
        #        due to current terrible parser.
        self.conv_params = [
            ("MOTION/PRINT/CELL", "_SECTION_PARAMETERS_ ON"),
            ("MOTION/PRINT/TRAJECTORY", "_SECTION_PARAMETERS_ ON"),
            ("MOTION/PRINT/FORCES", "_SECTION_PARAMETERS_ ON"),
        ]

        return


@dataclasses.dataclass
class BFGSMinimiser(MotionController):

    name: str = "bfgs"

    def __post_init__(self):
        """"""
        super().__post_init__()

        more_params = [
            ("GLOBAL", "RUN_TYPE GEO_OPT"),
            ("MOTION/GEO_OPT", "TYPE MINIMIZATION"),
            ("MOTION/GEO_OPT", "OPTIMIZER BFGS"),
            ("MOTION/PRINT/RESTART_HISTORY/EACH", f"GEO_OPT {self.ckpt_period}"),
        ]

        self.conv_params.extend(more_params)

        return


@dataclasses.dataclass
class CGMinimiser(MotionController):

    name: str = "cg"

    def __post_init__(self):
        """"""
        super().__post_init__()

        more_params = [
            ("GLOBAL", "RUN_TYPE GEO_OPT"),
            ("MOTION/GEO_OPT", "TYPE MINIMIZATION"),
            ("MOTION/GEO_OPT", "OPTIMIZER CG"),
            ("MOTION/PRINT/RESTART_HISTORY/EACH", f"GEO_OPT {self.ckpt_period}"),
        ]

        self.conv_params.extend(more_params)

        return


@dataclasses.dataclass
class MDController(MotionController):

    #: Controller name.
    name: str = "md"

    #: Timestep in fs.
    timestep: float = 1.0

    #: Temperature in Kelvin.
    temperature: float = 300.0

    #: Pressure in bar.
    pressure: float = 1.0

    #: Whether fix center of mass.
    fix_com: bool = True

    def __post_init__(self):
        """"""
        super().__post_init__()

        more_params = [
            ("GLOBAL", "RUN_TYPE MD"),
            ("MOTION/MD", f"TIMESTEP {self.timestep}"),
            ("MOTION/MD", f"TEMPERATURE {self.temperature}"),
            ("MOTION/PRINT/RESTART_HISTORY/EACH", f"MD {self.ckpt_period}"),
        ]

        # TODO: Check remove_rotation and remove_translation,
        #       CP2K initialise velocities without net translation,
        #       and the ANGVEL_ZERO seems not to work with pbc systems.

        self.conv_params.extend(more_params)

        return


@dataclasses.dataclass
class Verlet(MDController):

    name: str = "verlet"

    def __post_init__(self):
        """"""
        super().__post_init__()

        more_params = [
            ("MOTION/MD", "ENSEMBLE NVE"),
        ]

        self.conv_params.extend(more_params)

        return


@dataclasses.dataclass
class NoseHooverThermostat(MDController):

    name: str = "nose_hoover"

    def __post_init__(self):
        """"""
        super().__post_init__()

        taut = self.params.get("Tdamp", 100.0)
        assert taut is not None

        more_params = [
            ("MOTION/MD", "ENSEMBLE NVT"),
            ("MOTION/MD/THERMOSTAT", f"TYPE NOSE"),
            ("MOTION/MD/THERMOSTAT/NOSE", f"TIMECON {taut}"),
        ]

        self.conv_params.extend(more_params)

        return


@dataclasses.dataclass
class CSVRThermostat(MDController):

    name: str = "csvr"

    def __post_init__(self):
        """"""
        super().__post_init__()

        taut = self.params.get("Tdamp", 100.0)
        assert taut is not None

        more_params = [
            ("MOTION/MD", "ENSEMBLE NVT"),
            ("MOTION/MD/THERMOSTAT", f"TYPE CSVR"),
            ("MOTION/MD/THERMOSTAT/CSVR", f"TIMECON {taut}"),
        ]

        self.conv_params.extend(more_params)

        return


@dataclasses.dataclass
class MartynaBarostat(MDController):

    def __post_init__(self):
        """"""
        super().__post_init__()

        taut = self.params.get("Tdamp", 100.0)
        assert taut is not None

        taup = self.params.get("Pdamp", 100.0)
        assert taup is not None

        isotropic = self.params.get("isotropic", True)
        assert isotropic is not None

        more_params = [
            ("MOTION/MD", "ENSEMBLE NPT_I" if isotropic else "ENSEMBLE NPT_F"),
            ("MOTION/MD/THERMOSTAT", f"TYPE NOSE"),
            ("MOTION/MD/THERMOSTAT/NOSE", f"TIMECON {taut}"),
            ("MOTION/MD/BAROSTAT", f"PRESSURE {self.pressure}"),
            ("MOTION/MD/BAROSTAT", f"TIMECON {taup}"),
        ]

        self.conv_params.extend(more_params)

        return


controllers = dict(
    # - spc,
    single_point_spc=SinglePointController,
    # - min
    cg_min=CGMinimiser,
    bfgs_min=BFGSMinimiser,
    # - md
    verlet_nve=Verlet,
    csvr_nvt=CSVRThermostat,
    nose_hoover_nvt=NoseHooverThermostat,
    martyna_npt=MartynaBarostat,
    # - freq
    finite_difference_freq=FrequencyController,
)

default_controllers = dict(
    spc=SinglePointController,
    min=BFGSMinimiser,
    nve=Verlet,
    nvt=CSVRThermostat,
    npt=MartynaBarostat,
    # TODO: Make this a submodule!
    freq=FrequencyController,
)


@dataclasses.dataclass
class Cp2kDriverSetting(DriverSetting):

    #: Simulation task.
    task: str = "spc"

    #: MD Ensemble.
    ensemble: str = "nve"

    #: Dynamics controller.
    controller: dict = dataclasses.field(default_factory=dict)

    #: Force tolerance.
    fmax: Optional[float] = 4.5e-4 * (units.Hartree / units.Bohr)

    def __post_init__(self):
        """"""
        pairs = []  # key-value pairs that avoid conflicts by same keys

        _init_params = {}
        _init_params.update(ckpt_period=self.ckpt_period)
        if self.task == "min":
            # fmax and steps can be set on-the-fly, see get_run_params
            suffix = self.task
        elif self.task == "md":
            # steps can be set on-the-fly, see get_run_params
            suffix = self.ensemble
            _init_params.update(
                timestep=self.timestep,
                temperature=self.temp,
                pressure=self.press,
            )
        elif self.task == "freq":
            suffix = "freq"
        else:
            suffix = self.task

        _init_params.update(**self.controller)

        if self.controller:
            cont_cls_name = self.controller["name"] + "_" + suffix
            if cont_cls_name in controllers:
                cont_cls = controllers[cont_cls_name]
            else:
                raise RuntimeError(f"Unknown controller {cont_cls_name}.")
        else:
            cont_cls = default_controllers[suffix]

        cont = cont_cls(**_init_params)
        pairs.extend(cont.conv_params)  # type: ignore

        # We store cp2k parameters as pairs
        self._internals["pairs"] = pairs

        return

    def get_run_params(self, *args, **kwargs):
        """"""
        # - convergence criteria
        fmax_ = kwargs.get("fmax", self.fmax)
        steps_ = kwargs.get("steps", self.steps)

        run_pairs = []
        if self.task == "min":
            run_pairs.append(
                ("MOTION/GEO_OPT", f"MAX_ITER {steps_}"),
            )
            if fmax_ is not None:
                run_pairs.append(
                    ("MOTION/GEO_OPT", f"MAX_FORCE {fmax_/(units.Hartree/units.Bohr)}")
                )
        if self.task == "md":
            run_pairs.append(
                ("MOTION/MD", f"STEPS {steps_}"),
            )

        # - add constraint
        run_params = dict(
            constraint=kwargs.get("constraint", self.constraint), run_pairs=run_pairs
        )

        return run_params


class Cp2kDriver(AbstractDriver):

    name = "cp2k"

    default_task = "spc"
    supported_tasks = ["spc", "min", "md", "freq"]

    #: Class for setting.
    setting_cls: type[DriverSetting] = Cp2kDriverSetting

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """"""
        verified = super()._verify_checkpoint(*args, **kwargs)
        if verified:
            if self.setting.task == "spc":
                verified = read_cp2k_spc_convergence(self.directory / "cp2k.out")
            else:
                checkpoints = list(self.directory.glob("*.restart"))
                self._debug(f"checkpoints: {checkpoints}")
                if not checkpoints:
                    verified = False
        else:
            ...

        return verified

    def _irun(self, atoms: Atoms, ckpt_wdir=None, *args, **kwargs):
        """"""
        assert isinstance(self.calc, Cp2kFileIO), "Cp2kDriver must use Cp2kFileIO."
        if ckpt_wdir is None:  # start from the scratch
            # Check if there is `cp2k.out` from a previous failed calculation.
            # If there are outputs from multiple calculations, the parser for
            # spc will fail as it needs read `- Atoms:`.
            if (self.directory / "cp2k.out").exists():
                (self.directory / "cp2k.out").unlink()
                self._print("The previous `cp2k.out` is removed.")
            # Get all parameters
            run_params = self.setting.get_run_params(**kwargs)
            run_params.update(**self.setting.get_init_params())

            # Update input template
            # GLOBAL section is automatically created...
            # FORCE_EVAL.(METHOD, POISSON)
            inp = self.calc.parameters.inp  # string
            sec = parse_input(inp)
            for k, v in run_params["pairs"]:
                sec.add_keyword(k, v)
            for k, v in run_params["run_pairs"]:
                sec.add_keyword(k, v)

            # Ceck constraint
            cons_text = run_params.pop("constraint", None)
            mobile_indices, frozen_indices = parse_constraint_info(
                atoms, cons_text, ret_text=False
            )
            # if self.setting.task == "freq" and mobile_indices:
            #     mobile_indices = sorted(mobile_indices)
            #     sec.add_keyword(
            #         "VIBRATIONAL_ANALYSIS/MODE_SELECTIVE",
            #         f"ATOMS {' '.join([str(i+1) for i in mobile_indices])}",
            #     )
            #     sec.add_keyword(
            #         "VIBRATIONAL_ANALYSIS/MODE_SELECTIVE/INVOLVED_ATOMS",
            #         f"INVOLVED_ATOMS {' '.join([str(i+1) for i in mobile_indices])}",
            #     )
            if frozen_indices:
                # atoms._del_constraints()
                # atoms.set_constraint(FixAtoms(indices=frozen_indices))
                frozen_indices = sorted(frozen_indices)
                sec.add_keyword(
                    "MOTION/CONSTRAINT/FIXED_ATOMS",
                    "LIST {}".format(" ".join([str(i + 1) for i in frozen_indices])),
                )
        else:
            with open(ckpt_wdir / "cp2k.inp", "r") as fopen:
                inp = "".join(fopen.readlines())
            sec = parse_input(inp)

            def remove_keyword(section, keywords):
                """"""
                for kw in keywords:
                    parts = kw.upper().split("/")
                    subsection = section.get_subsection("/".join(parts[0:-1]))
                    new_subkeywords = []
                    for subkw in subsection.keywords:
                        if parts[-1] not in subkw:
                            new_subkeywords.append(subkw)
                    subsection.keywords = new_subkeywords

                return

            remove_keyword(
                sec,
                keywords=[  # avoid conflicts
                    "GLOBAL/PROJECT",
                    "GLOBAL/PRINT_LEVEL",
                    "FORCE_EVAL/METHOD",
                    "FORCE_EVAL/DFT/BASIS_SET_FILE_NAME",
                    "FORCE_EVAL/DFT/POTENTIAL_FILE_NAME",
                    "FORCE_EVAL/SUBSYS/CELL/PERIODIC",
                    "FORCE_EVAL/SUBSYS/CELL/A",
                    "FORCE_EVAL/SUBSYS/CELL/B",
                    "FORCE_EVAL/SUBSYS/CELL/C",
                ],
            )

            sec.add_keyword(
                "EXT_RESTART/RESTART_FILE_NAME", str(ckpt_wdir / "cp2k-1.restart")
            )
            sec.add_keyword("FORCE_EVAL/DFT/SCF/SCF_GUESS", "RESTART")

            # - copy wavefunctions...
            restart_wfns = sorted(list(ckpt_wdir.glob("*.wfn")))
            for wfn in restart_wfns:
                (self.directory / wfn.name).symlink_to(wfn, target_is_directory=False)

        self.calc.parameters.inp = "\n".join(sec.write())

        try:
            atoms.calc = self.calc
            _ = atoms.get_forces()
        except Exception as e:
            self._debug(e)
            self._debug(traceback.format_exc())

        return

    def _read_a_single_trajectory(self, wdir, *args, **kwargs):
        """"""
        frames = read_cp2k_outputs(wdir, prefix=self.name)

        return frames

    def read_trajectory(self, *args, **kwargs) -> list[Atoms]:
        """"""
        if self.setting.task in ["spc"]:
            atoms = read_cp2k_spc(self.directory, prefix="cp2k")
            scf_convergence = read_cp2k_convergence(
                pathlib.Path(self.directory) / "cp2k.out"
            )
            if not scf_convergence:
                atoms = ScfErrAtoms.from_atoms(atoms)
                self._print(f"ScfErrAtoms Step {0} @ {str(self.directory)}")
            traj_frames = [atoms]
        elif self.setting.task in ["min", "cmin", "md"]:
            prev_wdirs = sorted(self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"))
            self._debug(f"prev_wdirs: {prev_wdirs}")

            traj_list = []
            for w in prev_wdirs:
                curr_frames = self._read_a_single_trajectory(w)
                traj_list.append(curr_frames)

            cp2ktraj = self.directory / "cp2k-pos-1.xyz"
            if cp2ktraj.exists() and cp2ktraj.stat().st_size != 0:
                traj_list.append(read_cp2k_outputs(self.directory, prefix=self.name))

            # -- concatenate
            traj_frames, ntrajs = [], len(traj_list)
            if ntrajs > 0:
                traj_frames.extend(traj_list[0])
                for i in range(1, ntrajs):
                    assert np.allclose(
                        traj_list[i - 1][-1].positions, traj_list[i][0].positions
                    ), f"Traj {i-1} and traj {i} are not consecutive."
                    traj_frames.extend(traj_list[i][1:])
            else:
                ...
        else:
            raise RuntimeError(f"Unknown task `{self.setting.task}`.")

        return traj_frames


if __name__ == "__main__":
    ...
