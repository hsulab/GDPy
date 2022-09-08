#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
from pathlib import Path

from GDPy.potential.potential import AbstractPotential

class VaspManager(AbstractPotential):

    name = "vasp"

    implemented_backends = ["vasp"]
    valid_combinations = [
        ["vasp", "vasp"] # calculator, dynamics
    ]

    def __init__(self):

        return

    def _set_environs(self, pp_path, vdw_path):
        # ===== environs TODO: from inputs 
        # - ASE_VASP_COMMAND
        # pseudo 
        if "VASP_PP_PATH" in os.environ.keys():
            os.environ.pop("VASP_PP_PATH")
        os.environ["VASP_PP_PATH"] = pp_path

        # vdw 
        vdw_envname = "ASE_VASP_VDW"
        if vdw_envname in os.environ.keys():
            _ = os.environ.pop(vdw_envname)
        os.environ[vdw_envname] = vdw_path

        return

    def register_calculator(self, calc_params):
        """"""
        # - check backends
        super().register_calculator(calc_params)

        # - some extra params
        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())

        # TODO: check pp and vdw
        incar = calc_params.pop("incar", None)
        pp_path = calc_params.pop("pp_path", None)
        vdw_path = calc_params.pop("vdw_path", None)

        if self.calc_backend == "vasp":
            # return ase calculator
            from ase.calculators.vasp import Vasp
            calc = Vasp(directory=directory, command=command)

            # - set some default params
            calc.set_xc_params("PBE") # NOTE: since incar may not set GGA
            calc.set(lorbit=10)
            calc.set(gamma=True)
            calc.set(lreal="Auto")
            if incar is not None:
                calc.read_incar(incar)
            self._set_environs(pp_path, vdw_path)
            #print("fmax: ", calc.asdict()["inputs"])
            # - update residual params
            calc.set(**calc_params)
        else:
            pass
        
        self.calc = calc

        return


if __name__ == "__main__":
    pass