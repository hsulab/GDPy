#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pathlib
from typing import NoReturn

from . import AbstractPotentialManager


class VaspManager(AbstractPotentialManager):

    name = "vasp"

    implemented_backends = ["vasp", "vasp_interactive"]
    valid_combinations = (
        # calculator, dynamics
        ("vasp", "vasp"), 
        ("vasp_interactive", "ase")
    )

    def __init__(self):

        return

    def _set_environs(self, pp_path, vdw_path) -> None:
        """Set files need for calculation.

        NOTE: pp_path and vdw_path may not exist since we would like to create a
              dummy calculator.

        """
        # ===== environs TODO: from inputs 
        # - ASE_VASP_COMMAND
        # pseudo 
        if "VASP_PP_PATH" in os.environ.keys():
            os.environ.pop("VASP_PP_PATH", "")
        os.environ["VASP_PP_PATH"] = pp_path

        # - vdw 
        vdw_envname = "ASE_VASP_VDW"
        if vdw_envname in os.environ.keys():
            _ = os.environ.pop(vdw_envname, "")
        os.environ[vdw_envname] = vdw_path

        return

    def register_calculator(self, calc_params):
        """"""
        # - check backends
        super().register_calculator(calc_params)

        # - some extra params
        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", pathlib.Path.cwd())

        # NOTE: whether check pp and vdw existence
        #       since sometimes we'd like a dummy calculator
        #       -- convert paths to absolute ones
        incar = calc_params.pop("incar", None)
        pp_path = calc_params.pop("pp_path", "")
        vdw_path = calc_params.pop("vdw_path", "")

        incar = str(pathlib.Path(incar).absolute())
        self.calc_params.update(incar=incar)
        pp_path = str(pathlib.Path(pp_path).absolute())
        self.calc_params.update(pp_path=pp_path)
        vdw_path = str(pathlib.Path(vdw_path).absolute())
        self.calc_params.update(vdw_path=vdw_path)

        if self.calc_backend == "vasp":
            # return ase calculator
            from ase.calculators.vasp import Vasp
            calc = Vasp(directory=directory, command=command)

            # - set some default electronic parameters
            calc.set_xc_params("PBE") # NOTE: since incar may not set GGA
            calc.set(lorbit=10)
            calc.set(gamma=True)
            calc.set(lreal="Auto")
            if incar is not None:
                calc.read_incar(incar)
            self._set_environs(pp_path, vdw_path)
            # - update residual params
            calc.set(**calc_params)
        elif self.calc_backend == "vasp_interactive":
            from vasp_interactive import VaspInteractive
            calc = VaspInteractive(directory=directory, command=command)
            # - set some default electronic parameters
            calc.set_xc_params("PBE") # NOTE: since incar may not set GGA
            calc.set(lorbit=10)
            calc.set(gamma=True)
            calc.set(lreal="Auto")
            if incar is not None:
                calc.read_incar(incar)
            # - set some vasp_interactive parameters
            calc.set(potim=0.0)
            calc.set(ibrion=-1)
            calc.set(ediffg=0)
            #calc.set(isif=3) # TODO: Does not support stress for now...
            self._set_environs(pp_path, vdw_path)
            # - update residual params
            calc.set(**calc_params)
        else:
            raise NotImplementedError(f"Unimplemented backend {self.calc_backend} for vasp.")
        
        self.calc = calc

        return


if __name__ == "__main__":
    pass
