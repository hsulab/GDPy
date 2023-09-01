#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
import pathlib
import traceback

from typing import Union, List

from ase import Atoms
from ase import units
from ase.io import read, write
from ase.calculators.cp2k import parse_input, InputSection

from .string import AbstractStringReactor, StringReactorSetting
from ..builder.constraints import parse_constraint_info

@dataclasses.dataclass
class Cp2kStringReactorSetting(StringReactorSetting):

    #: Number of tasks/processors/cpus for each image.
    ntasks_per_image: int = 1

    def __post_init__(self):
        """"""
        pairs = []

        pairs.extend(
            [
                ("GLOBAL", "RUN_TYPE BAND"),
                ("MOTION/BAND", "BAND_TYPE CI-NEB"),
                ("MOTION/BAND", f"NPROC_REP {self.ntasks_per_image}"),
                ("MOTION/BAND", f"NUMBER_OF_REPLICA {self.nimages}"),
                ("MOTION/BAND", f"K_SPRING {self.k}"),
                ("MOTION/BAND", "ROTATE_FRAMES F"),
                ("MOTION/BAND", "ALIGN_FRAMES F"),
                ("MOTION/BAND/CI_NEB", "NSTEPS_IT 2"),
                ("MOTION/BAND/OPTIMIZE_BAND", "OPT_TYPE DIIS"),
                ("MOTION/PRINT/RESTART_HISTORY/EACH", f"BAND {self.restart_period}"),
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
        fmax_ = kwargs.get("fmax", self.fmax)
        steps_ = kwargs.get("steps", self.steps)

        run_pairs = []
        run_pairs.append(
            ("MOTION/BAND/OPTIMIZE_BAND/DIIS", f"MAX_STEPS {self.steps}")
        )
        if fmax_ is not None:
            run_pairs.append(
                ("MOTION/BAND/CONVERGENCE_CONTROL", f"MAX_FORCE {fmax_/(units.Hartree/units.Bohr)}")
            )

        # - add constraint
        run_params = dict(
            constraint = kwargs.get("constraint", self.constraint),
            run_pairs = run_pairs
        )

        return run_params


class Cp2kStringReactor(AbstractStringReactor):

    name: str = "cp2k"

    def __init__(self, calc=None, params={}, ignore_convergence=False, directory="./", *args, **kwargs) -> None:
        """"""
        self.calc = calc
        if self.calc is not None:
            self.calc.reset()

        self.ignore_convergence = ignore_convergence

        self.directory = directory
        self.cache_nebtraj = self.directory/self.traj_name

        # - parse params
        self.setting = Cp2kStringReactorSetting(**params)
        self._debug(self.setting)

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
            inp = self.calc.parameters.inp # string
            sec = parse_input(inp)
            for (k, v) in run_params["pairs"]:
                sec.add_keyword(k, v)
            for (k, v) in run_params["run_pairs"]:
                sec.add_keyword(k, v)

            # -- check constraint
            cons_text = run_params.pop("constraint", None)
            mobile_indices, frozen_indices = parse_constraint_info(
                atoms, cons_text=cons_text, ignore_ase_constraints=True, ret_text=False
            )
            if frozen_indices:
                #atoms._del_constraints()
                #atoms.set_constraint(FixAtoms(indices=frozen_indices))
                frozen_indices = sorted(frozen_indices)
                sec.add_keyword(
                    "MOTION/CONSTRAINT/FIXED_ATOMS", 
                    "LIST {}".format(" ".join([str(i+1) for i in frozen_indices]))
                )
            
            # -- add replica information
            band_section = sec.get_subsection("MOTION/BAND")
            for replica in images:
                cur_rep = InputSection(name="REPLICA")
                for pos in replica.positions:
                    cur_rep.add_keyword("COORD", ("{:.18e} "*3).format(*pos), unique=False)
                band_section.subsections.append(cur_rep)

            # - update input
            self.calc.parameters.inp = "\n".join(sec.write())
            atoms.calc = self.calc

            # - run calculation
            _ = atoms.get_forces()

        except Exception as e:
            self._debug(e)
            self._debug(traceback.print_exc())

        return
    
    def read_convergence(self, *args, **kwargs):
        """"""
        super().read_convergence(*args, **kwargs)

        return
    
    def read_trajectory(self, *args, **kwargs):
        """"""

        return


if __name__ == "__main__":
    ...