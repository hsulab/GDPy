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

        # check whether the calculation will be sent to remote
        # If so, incar will be not read until actually simulation is performed.
        # The stored potter parameters should not contain remote.
        # Maybe it is better to move incar-related thing to VaspDriver...
        self.calc_params.pop("remote", None)
        is_remote = calc_params.pop("remote", False)

        # - some extra params
        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", pathlib.Path.cwd())

        # NOTE: whether check pp and vdw existence
        #       since sometimes we'd like a dummy calculator
        #       -- convert paths to absolute ones

        inp_fdict = dict(
            incar = calc_params.pop("incar", None),
            pp_path = calc_params.pop("pp_path", ""),
            vdw_path = calc_params.pop("vdw_path", "")
        )

        if not is_remote:
            for fname in inp_fdict.keys():
                inp_fdict[fname] = str(pathlib.Path(inp_fdict[fname]).resolve())
            self.calc_params.update(**inp_fdict)
        else:
            for fname, fpath in inp_fdict.items():
                if not pathlib.Path(fpath).is_absolute():
                    raise RuntimeError(f"{fname} for remote must be an absolute path.")

        if self.calc_backend == "vasp":
            # return ase calculator
            from ase.calculators.vasp import Vasp
            calc = Vasp(directory=directory, command=command)

            # - set some default electronic parameters
            calc.set_xc_params("PBE") # NOTE: since incar may not set GGA
            calc.set(lorbit=10)
            calc.set(gamma=True)
            calc.set(lreal="Auto")
            if not is_remote and inp_fdict["incar"] is not None:
                calc.read_incar(inp_fdict["incar"])
            self._set_environs(inp_fdict["pp_path"], inp_fdict["vdw_path"])
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
            if not is_remote and inp_fdict["incar"] is not None:
                calc.read_incar(inp_fdict["incar"])
            # - set some vasp_interactive parameters
            calc.set(potim=0.0)
            calc.set(ibrion=-1)
            calc.set(ediffg=0)
            #calc.set(isif=3) # TODO: Does not support stress for now...
            self._set_environs(inp_fdict["pp_path"], inp_fdict["vdw_path"])
            # - update residual params
            calc.set(**calc_params)
        else:
            raise NotImplementedError(f"Unimplemented backend {self.calc_backend} for vasp.")
        
        self.calc = calc

        return


if __name__ == "__main__":
    pass
