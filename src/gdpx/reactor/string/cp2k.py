#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
import os
import pathlib
import traceback

import numpy as np
from ase import Atoms, units
from ase.calculators.cp2k import InputSection, parse_input
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from gdpx.backend.cp2k import read_cp2k_output_from_band
from gdpx.group import evaluate_constraint_expression

from .string import BaseStringReactor, StringReactorSetting


def run_cp2k(name, command, directory):
    """Run vasp from the command.

    ASE Vasp does not treat restart of a MD simulation well. Therefore, we run
    directly from the command if INCAR aready exists.

    """
    import subprocess

    from ase.calculators.calculator import CalculationFailed, EnvironmentError

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
        msg = 'Calculator "{}" failed with command "{}" failed in ' "{} with error code {}".format(
            name, command, path, errorcode
        )
        raise CalculationFailed(msg)

    return


@dataclasses.dataclass
class Cp2kStringReactorSetting(StringReactorSetting):

    backend: str = "cp2k"

    #: Number of tasks/processors/cpus for each image.
    ntasks_per_image: int = 1

    def __post_init__(self):
        """"""
        pairs = []

        method_section = [("GLOBAL", "RUN_TYPE BAND")]
        if not self.climb:
            method_section.append(("MOTION/BAND", "BAND_TYPE IT-NEB"))
        else:
            method_section.append(("MOTION/BAND", "BAND_TYPE CI-NEB"))
            # Number of IT-NEB steps before CI-NEB
            method_section.append(("MOTION/BAND/CI_NEB", "NSTEPS_IT 2"))
        pairs.extend(method_section)

        pairs.extend(
            [
                ("MOTION/BAND", f"NPROC_REP {self.ntasks_per_image}"),
                ("MOTION/BAND", f"NUMBER_OF_REPLICA {self.nimages}"),
                (
                    "MOTION/BAND",
                    f"K_SPRING {self.kspring/(units.Hartree/units.Bohr**2)}",
                ),
                ("MOTION/BAND", "ROTATE_FRAMES F"),
                ("MOTION/BAND", "ALIGN_FRAMES F"),
                ("MOTION/BAND/OPTIMIZE_BAND", "OPT_TYPE DIIS"),
                ("MOTION/BAND/OPTIMIZE_BAND/DIIS", "NO_LS T"),
                ("MOTION/BAND/OPTIMIZE_BAND/DIIS", "N_DIIS 3"),
                (
                    "MOTION/PRINT/RESTART_HISTORY/EACH",
                    f"BAND {self.ckpt_period}",
                ),
            ]
        )

        pairs.extend(
            [
                ("MOTION/PRINT/CELL", "_SECTION_PARAMETERS_ ON"),
                ("MOTION/PRINT/TRAJECTORY", "_SECTION_PARAMETERS_ ON"),
                ("MOTION/PRINT/FORCES", "_SECTION_PARAMETERS_ ON"),
            ]
        )
        self._internals["pairs"] = pairs

        return

    def get_run_params(self, *args, **kwargs):
        """"""
        # - convergence criteria
        rmax_ = kwargs.get("rmax", self.rmax)
        rrms_ = kwargs.get("rrms", self.rrms)
        fmax_ = kwargs.get("fmax", self.fmax)
        frms_ = kwargs.get("frms", self.frms)
        steps_ = kwargs.get("steps", self.steps)

        run_pairs = []
        run_pairs.append(("MOTION/BAND/OPTIMIZE_BAND/DIIS", f"MAX_STEPS {steps_}"))
        if fmax_ is not None:
            run_pairs.extend(
                [
                    (
                        "MOTION/BAND/CONVERGENCE_CONTROL",
                        f"MAX_FORCE {fmax_/(units.Hartree/units.Bohr)}",
                    ),
                    (
                        "MOTION/BAND/CONVERGENCE_CONTROL",
                        f"MAX_DR {rmax_/(units.Bohr)}",
                    ),
                    (
                        "MOTION/BAND/CONVERGENCE_CONTROL",
                        f"RMS_FORCE {frms_/(units.Hartree/units.Bohr)}",
                    ),
                    (
                        "MOTION/BAND/CONVERGENCE_CONTROL",
                        f"RMS_DR {rrms_/(units.Bohr)}",
                    ),
                ]
            )

        # - add constraint
        run_params = dict(
            constraint=kwargs.get("constraint", self.constraint),
            run_pairs=run_pairs,
        )

        return run_params


class Cp2kStringReactor(BaseStringReactor):

    name: str = "cp2k"

    traj_name: str = "cp2k.out"

    setting_cls: type[StringReactorSetting] = Cp2kStringReactorSetting

    def _verify_checkpoint(self):
        """Check if the current directory has any valid outputs or
        it just created the input files.

        """
        verified = super()._verify_checkpoint()
        if verified:
            checkpoints = list(self.directory.glob("*.restart"))
            self._debug(f"checkpoints: {checkpoints}")
            if not checkpoints:
                verified = False
        else:
            ...

        return verified

    def _irun(self, structures: list[Atoms], ckpt_wdir=None, *args, **kwargs):
        """"""
        run_params = self.setting.get_run_params(**kwargs)
        run_params.update(**self.setting.get_init_params())

        if ckpt_wdir is None:  # start from the scratch
            images = self._align_structures(structures, run_params)
            write(self.directory / "images.xyz", images)
            atoms = images[0]  # use the initial state

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
            cons_expr = run_params.pop("constraint", None)
            _, frozen_indices = evaluate_constraint_expression(
                atoms,
                cons_expr,
            )
            if frozen_indices:
                # atoms._del_constraints()
                # atoms.set_constraint(FixAtoms(indices=frozen_indices))
                frozen_indices = sorted(frozen_indices)
                sec.add_keyword(
                    "MOTION/CONSTRAINT/FIXED_ATOMS",
                    "LIST {}".format(" ".join([str(i + 1) for i in frozen_indices])),
                )

            # -- add replica information
            band_section = sec.get_subsection("MOTION/BAND")
            for replica in images:
                cur_rep = InputSection(name="REPLICA")
                for pos in replica.positions:
                    cur_rep.add_keyword("COORD", ("{:.18e} " * 3).format(*pos), unique=False)
                band_section.subsections.append(cur_rep)
        else:  # start from a checkpoint
            atoms = read(ckpt_wdir / "images.xyz", "0")
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
                    "FORCE_EVAL/DFT/SCF/SCF_GUESS",
                    "FORCE_EVAL/DFT/BASIS_SET_FILE_NAME",
                    "FORCE_EVAL/DFT/POTENTIAL_FILE_NAME",
                    "EXT_RESTART/RESTART_FILE_NAME",
                ],
            )

            # Make cp2k takes CELL, POS, VEL, THERMOSTAT and ... from the restart file.
            sec.add_keyword(
                "EXT_RESTART",
                "RESTART_FILE_NAME " + str(ckpt_wdir / "cp2k-1.restart"),
            )

            # Make cp2k initialise the wavefunction from the restart file.
            sec.add_keyword("FORCE_EVAL/DFT/SCF", "SCF_GUESS RESTART")

            # Copy wavefunctions...
            restart_wfns = sorted(list(ckpt_wdir.glob("*.wfn")))
            for wfn in restart_wfns:
                (self.directory / wfn.name).symlink_to(wfn, target_is_directory=False)

        # update input
        self.calc.parameters.inp = "\n".join(sec.write())
        atoms.calc = self.calc

        # run calculation
        try:
            self.calc.atoms = atoms
            self.calc.write_input(atoms)
            run_cp2k("cp2k", self.calc.command, self.directory)
            self.calc.atoms = None

        except Exception as e:
            self._debug(e)
            self._debug(traceback.print_exc())

        return

    def read_convergence(self, *args, **kwargs):
        """"""
        converged = super().read_convergence(*args, **kwargs)

        with open(self.directory / "cp2k.out", "r") as fopen:
            lines = fopen.readlines()

        for line in lines:
            if "PROGRAM ENDED AT" in line:
                converged = True
                break

        return converged

    def _read_a_single_trajectory(self, wdir: pathlib.Path) -> list[list[Atoms]]:
        """Read a single trajectory from the output file.

        Fixed atoms have zero forces.

        """
        frames = read_cp2k_output_from_band(wdir, prefix="cp2k", print_func=self._print, debug_func=self._debug)

        return frames

    def concatenate_trajectories(self, traj_list: list[list[Atoms]]) -> list[list[Atoms]]:
        """Concatenate a list of trajectories.

        CP2K starts the new trajectory from the next image of the previous trajectory, therefore, we do not check
        continuity in positions here.

        In case of some calculations, the energetic continuity is not guaranteed, for example, the spin-polarised
        calculation can give slightly different energies for the same structure due to the random initialisation of the
        wavefunction.

        Args:
            traj_list: A list of trajectories, each trajectory is a list of Atoms objects.

        Returns:
            A list of Atoms objects that are concatenated from the input list of trajectories.

        """
        traj_frames, ntrajs = [], len(traj_list)
        if ntrajs > 0:
            traj_frames.extend(traj_list[0])
            for i in range(1, ntrajs):
                # TODO: We should check step info if we have.
                traj_frames.extend(traj_list[i])
        else:
            ...

        return traj_frames


if __name__ == "__main__":
    ...
