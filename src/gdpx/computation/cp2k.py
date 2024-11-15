#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
import pathlib
import traceback
import warnings
from typing import List

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.cp2k import InputSection, parse_input

from ..backend.cp2k import read_cp2k_energy_force, read_cp2k_outputs, read_cp2k_spc
from ..builder.constraints import parse_constraint_info
from .driver import AbstractDriver, Controller, DriverSetting

UNCONVERGED_SCF_FLAG: str = "*** WARNING in qs_scf.F:598 :: SCF run NOT converged ***"
ABORT_FLAG: str = "ABORT"


@dataclasses.dataclass
class MotionController(Controller):

    #: Controller name.
    name: str = "motion"

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

    ckpt_period: int = 100

    def __post_init__(self):
        """"""
        super().__post_init__()

        more_params = {
            "GLOBAL": "RUN_TYPE GEO_OPT",
            "MOTION/GEO_OPT": "TYPE MINIMIZATION",
            "MOTION/GEO_OPT": "OPTIMIZER BFGS",
            "MOTION/PRINT/RESTART_HISTORY/EACH": f"GEO_OPT {self.ckpt_period}",
        }

        self.conv_params.update(**more_params)

        return


@dataclasses.dataclass
class CGMinimiser(MotionController):

    name: str = "cg"

    ckpt_period: int = 100

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

    #: Save checkpoint every.
    ckpt_period: int = 100

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
            ("MOTION/PRINT/RESTART_HISTORY/EACH", f"MD {self.ckpt_period}"),
        ]

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
            ("MOTION/MD/THERMOSTAT/NOSE", f"TIMECON {taut}"),
            ("MOTION/MD/BAROSTAT", f"PRESSURE {self.pressure}"),
            ("MOTION/MD/BAROSTAT", f"TIMECON {taup}"),
        ]

        self.conv_params.extend(more_params)

        return


controllers = dict(
    # - min
    cg_min=CGMinimiser,
    bfgs_min=BFGSMinimiser,
    # - md
    verlet_nve=Verlet,
    csvr_nvt=CSVRThermostat,
    nose_hoover_nvt=NoseHooverThermostat,
    martyna_npt=MartynaBarostat,
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
    fmax: float = 4.5e-4 * (units.Hartree / units.Bohr)

    def __post_init__(self):
        """"""
        pairs = []  # key-value pairs that avoid conflicts by same keys
        if self.task == "spc":
            pairs.extend(
                [
                    ("GLOBAL", "RUN_TYPE ENERGY_FORCE"),
                    ("FORCE_EVAL/PRINT/FORCES", "_SECTION_PARAMETERS_ ON"),
                ]
            )

        default_controllers = dict(
            min=BFGSMinimiser,
            nve=Verlet,
            nvt=CSVRThermostat,
            npt=MartynaBarostat,
            # TODO: Make this a submodule!
            freq=Controller,
        )

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
            # TODO: Make this a submodule!
            pairs.extend(
                [
                    ("GLOBAL", "RUN_TYPE VIBRATIONAL_ANALYSIS"),
                    ("VIBRATIONAL_ANALYSIS", "DX 0.01"),
                    ("VIBRATIONAL_ANALYSIS", "NPROC_REP 16"),
                    # ("VIBRATIONAL_ANALYSIS/MODE_SELECTIVE", "INITIAL_GUESS ATOMIC"),
                    # ("VIBRATIONAL_ANALYSIS/MODE_SELECTIVE", "EPS_NORM 1.0E-5"),
                    # ("VIBRATIONAL_ANALYSIS/MODE_SELECTIVE", "EPS_MAX_VAL 1.0E-6"),
                ]
            )
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

    # saved_fnames = [
    #    "cp2k.inp", "cp2k.out", "cp2k-1.cell", "cp2k-pos-1.xyz", "cp2k-frc-1.xyz",
    #    "cp2k-BFGS.Hessian"
    # ]

    #: Class for setting.
    setting_cls: type[DriverSetting] = Cp2kDriverSetting

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """"""
        verified = super()._verify_checkpoint(*args, **kwargs)
        if verified:
            checkpoints = list(self.directory.glob("*.restart"))
            self._debug(f"checkpoints: {checkpoints}")
            if not checkpoints:
                verified = False
        else:
            ...

        return verified

    def _irun(self, atoms: Atoms, ckpt_wdir=None, *args, **kwargs):
        """"""
        if ckpt_wdir is None:  # start from the scratch
            run_params = self.setting.get_run_params(**kwargs)
            run_params.update(**self.setting.get_init_params())

            # - update input template
            # GLOBAL section is automatically created...
            # FORCE_EVAL.(METHOD, POISSON)
            inp = self.calc.parameters.inp  # string
            sec = parse_input(inp)
            for k, v in run_params["pairs"]:
                sec.add_keyword(k, v)
            for k, v in run_params["run_pairs"]:
                sec.add_keyword(k, v)

            # -- check constraint
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

    def read_trajectory(self, *args, **kwargs) -> List[Atoms]:
        """"""
        super().read_trajectory(*args, **kwargs)

        # read backups
        if self.setting.task in ["spc"]:
            atoms = read_cp2k_spc(self.directory, prefix="cp2k")
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


class Cp2kFileIO(FileIOCalculator):

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    default_parameters = dict(
        auto_write=False,
        basis_set="DZVP-MOLOPT-SR-GTH",
        basis_set_file="BASIS_MOLOPT",
        pseudo_potential="GTH-PBE",
        potential_file="POTENTIAL",
        inp="",
        force_eval_method="Quickstep",
        charge=0,
        uks=False,
        stress_tensor=False,
        poisson_solver="auto",
        xc="PBE",
        max_scf=50,
        cutoff=400 * units.Rydberg,
        print_level="MEDIUM",
    )

    """This calculator is consistent with v9.1 and v2022.1.
    """

    def __init__(
        self, restart=None, label="cp2k", atoms=None, command="cp2k.psmp", **kwargs
    ):
        """Construct CP2K-calculator object"""
        super().__init__(
            restart=restart, label=label, atoms=atoms, command=command, **kwargs
        )

        # complete command
        command_ = self.profile.command
        if "-i" in command_:
            ...
        else:
            label_name = pathlib.Path(self.label).name
            command_ += f" -i {label_name}.inp -o {label_name}.out"
        self.profile.command = command_

        return

    def read_results(self):
        """"""
        super().read_results()

        label_name = pathlib.Path(self.label).name

        # TODO: stress
        run_type = self.get_run_type().upper()
        if run_type in ["GEO_OPT", "CELL_OPT", "MD"]:
            trajectory = read_cp2k_outputs(self.directory, prefix=label_name)
            atoms = trajectory[-1]
            self.results["energy"] = atoms.get_potential_energy()
            self.results["free_energy"] = atoms.get_potential_energy(
                force_consistent=True
            )
            self.results["forces"] = atoms.get_forces()
        elif run_type in ["ENERGY_FORCE"]:
            atoms = self.atoms
            self.results = read_cp2k_energy_force(self.directory, prefix=label_name)
            assert self.results["energy"] is not None, f"{self.results['energy'] =}"
            assert self.results["forces"].shape[0] == len(
                atoms
            ), f"{self.results['forces'] =}"
        else:
            raise RuntimeError()

        scf_convergence = self.read_convergence()
        atoms.info["scf_convergence"] = scf_convergence
        if not scf_convergence:
            atoms.info["error"] = f"Unconverged SCF at {self.directory}."

        return

    def write_input(self, atoms, properties=None, system_changes=None):
        """"""
        super().write_input(atoms, properties, system_changes)

        # Support mixed basis_set
        prev_basis_set = self.parameters.basis_set
        if isinstance(prev_basis_set, str):
            curr_basis_set = {
                k: prev_basis_set for k in list(set(atoms.get_chemical_symbols()))
            }
        elif isinstance(prev_basis_set, dict):
            for k in list(set(atoms.get_chemical_symbols())):
                if k not in prev_basis_set:
                    raise RuntimeError(f"No basis_set for {k}.")
            curr_basis_set = prev_basis_set
        else:
            raise RuntimeError(f"Unknown basis_set {prev_basis_set}.")
        self.parameters.basis_set = curr_basis_set

        label_name = pathlib.Path(self.label).name
        wdir = pathlib.Path(self.directory)
        with open(wdir / f"{label_name}.inp", "w") as fopen:
            fopen.write(self._generate_input())

        self.parameters.basis_set = prev_basis_set

        return

    def get_run_type(self) -> str:
        """"""
        root = parse_input(self.parameters.inp)

        run_type = ""
        for subsection in root.subsections:
            if subsection.name == "GLOBAL":
                for keyword in subsection.keywords:
                    kw = keyword.strip().split()
                    if kw[0] == "RUN_TYPE":
                        run_type = kw[1]
                        break
                if run_type:
                    break
        else:
            ...

        assert run_type in [
            "ENERGY_FORCE",
            "GEO_OPT",
            "CELL_OPT",
            "MD",
        ], f"Unknown run_type `{run_type}`."

        return run_type

    def _generate_input(self):
        """Generates a CP2K input file"""
        p = self.parameters
        root = parse_input(p.inp)
        label_name = pathlib.Path(self.label).name
        root.add_keyword("GLOBAL", "PROJECT " + label_name)
        if p.print_level:
            root.add_keyword("GLOBAL", "PRINT_LEVEL " + p.print_level)
        # root.add_keyword("GLOBAL", "RUN_TYPE " + "CELL_OPT")
        if p.force_eval_method:
            root.add_keyword("FORCE_EVAL", "METHOD " + p.force_eval_method)
        if p.stress_tensor:
            root.add_keyword("FORCE_EVAL", "STRESS_TENSOR ANALYTICAL")
            root.add_keyword(
                "FORCE_EVAL/PRINT/STRESS_TENSOR", "_SECTION_PARAMETERS_ ON"
            )
        if p.basis_set_file:
            root.add_keyword(
                "FORCE_EVAL/DFT", "BASIS_SET_FILE_NAME " + p.basis_set_file
            )
        if p.potential_file:
            root.add_keyword(
                "FORCE_EVAL/DFT", "POTENTIAL_FILE_NAME " + p.potential_file
            )
        if p.cutoff:
            root.add_keyword("FORCE_EVAL/DFT/MGRID", "CUTOFF [eV] %.18e" % p.cutoff)
        if p.max_scf:
            root.add_keyword("FORCE_EVAL/DFT/SCF", "MAX_SCF %d" % p.max_scf)
            root.add_keyword("FORCE_EVAL/DFT/LS_SCF", "MAX_SCF %d" % p.max_scf)

        if p.xc:
            legacy_libxc = ""
            for functional in p.xc.split():
                functional = functional.replace("LDA", "PADE")  # resolve alias
                xc_sec = root.get_subsection("FORCE_EVAL/DFT/XC/XC_FUNCTIONAL")
                # libxc input section changed over time
                if functional.startswith("XC_") and self._shell.version < 3.0:
                    legacy_libxc += " " + functional  # handled later
                elif functional.startswith("XC_") and self._shell.version < 5.0:
                    s = InputSection(name="LIBXC")
                    s.keywords.append("FUNCTIONAL " + functional)
                    xc_sec.subsections.append(s)
                elif functional.startswith("XC_"):
                    s = InputSection(name=functional[3:])
                    xc_sec.subsections.append(s)
                else:
                    s = InputSection(name=functional.upper())
                    xc_sec.subsections.append(s)
            if legacy_libxc:
                root.add_keyword(
                    "FORCE_EVAL/DFT/XC/XC_FUNCTIONAL/LIBXC",
                    "FUNCTIONAL " + legacy_libxc,
                )

        if p.uks:
            root.add_keyword("FORCE_EVAL/DFT", "UNRESTRICTED_KOHN_SHAM ON")

        if p.charge and p.charge != 0:
            root.add_keyword("FORCE_EVAL/DFT", "CHARGE %d" % p.charge)

        # add Poisson solver if needed
        if p.poisson_solver == "auto" and not any(self.atoms.get_pbc()):
            root.add_keyword("FORCE_EVAL/DFT/POISSON", "PERIODIC NONE")
            root.add_keyword("FORCE_EVAL/DFT/POISSON", "PSOLVER  MT")

        # write coords
        syms = self.atoms.get_chemical_symbols()
        atoms = self.atoms.get_positions()
        for elm, pos in zip(syms, atoms):
            line = "%s %.18e %.18e %.18e" % (elm, pos[0], pos[1], pos[2])
            root.add_keyword("FORCE_EVAL/SUBSYS/COORD", line, unique=False)

        # write cell
        pbc = "".join([a for a, b in zip("XYZ", self.atoms.get_pbc()) if b])
        if len(pbc) == 0:
            pbc = "NONE"
        root.add_keyword("FORCE_EVAL/SUBSYS/CELL", "PERIODIC " + pbc)
        c = self.atoms.get_cell()
        for i, a in enumerate("ABC"):
            line = "%s %.18e %.18e %.18e" % (a, c[i, 0], c[i, 1], c[i, 2])
            root.add_keyword("FORCE_EVAL/SUBSYS/CELL", line)

        # determine pseudo-potential
        potential = p.pseudo_potential
        if p.pseudo_potential == "auto":
            if p.xc and p.xc.upper() in (
                "LDA",
                "PADE",
                "BP",
                "BLYP",
                "PBE",
            ):
                potential = "GTH-" + p.xc.upper()
            else:
                msg = "No matching pseudo potential found, using GTH-PBE"
                warnings.warn(msg, RuntimeWarning)
                potential = "GTH-PBE"  # fall back

        # write atomic kinds
        subsys = root.get_subsection("FORCE_EVAL/SUBSYS").subsections
        kinds = dict([(s.params, s) for s in subsys if s.name == "KIND"])
        for elem in set(self.atoms.get_chemical_symbols()):
            if elem not in kinds.keys():
                s = InputSection(name="KIND", params=elem)
                subsys.append(s)
                kinds[elem] = s
            if p.basis_set:
                kinds[elem].keywords.append("BASIS_SET " + p.basis_set[elem])
            if potential:
                kinds[elem].keywords.append("POTENTIAL " + potential)

        output_lines = ["!!! Generated by ASE !!!"] + root.write()
        return "\n".join(output_lines)

    def read_convergence(self):
        """Read SCF convergence."""
        label_name = pathlib.Path(self.label).name
        cp2kout = pathlib.Path(self.directory) / f"{label_name}.out"

        converged = True
        with open(cp2kout, "r") as fopen:
            while True:
                line = fopen.readline()
                if not line:
                    break
                if line.strip() == UNCONVERGED_SCF_FLAG:
                    converged = False
                    break
                if ABORT_FLAG in line:
                    converged = False
                    break

        return converged


if __name__ == "__main__":
    ...
