#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pathlib
from typing import NoReturn

from . import AbstractPotentialManager


class Cp2kManager(AbstractPotentialManager):

    name = "cp2k"

    implemented_backends = ["cp2k", "cp2k_shell"]
    valid_combinations = (
        # calculator, dynamics
        ("cp2k", "ase"),
        ("cp2k", "cp2k"), 
        ("cp2k_shell", "ase")
    )

    def __init__(self, *args, **kwargs):
        """"""

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
        conflict_keys =[]
        template = calc_params.pop("template", None)
        if template is not None:
            template = str(pathlib.Path(template).absolute())
            self.calc_params.update(template=template)
            # -- check conflicts
            conflict_keys =["cutoff", "max_scf", "xc"]
            from ase.calculators.cp2k import parse_input
            with open(template, "r") as fopen:
                inp = "".join(fopen.readlines())
            root_section = parse_input(inp)
            # TODO: check if run_type, default is ENERGY_FORCE
            # TODO: cutoff exists?
            _ = root_section.get_subsection("FORCE_EVAL/DFT/MGRID")
            print("FORCE_EVAL/DFT/MGRID: ", _)
            # TODO: max_scf exists?
            # TODO: xc exists?
        else:
            inp = ""

        # - check basis_set and pseudo_potential
        path_keywords = ["basis_set_file", "potential_file"]
        for k in path_keywords:
            fpath = calc_params.pop(k, None)
            if fpath is not None:
                calc_params[k] = str(pathlib.Path(fpath).resolve())
                self.calc_params[k] = str(pathlib.Path(fpath).resolve())

        if self.calc_backend == "cp2k":
            from gdpx.computation.cp2k import Cp2kFileIO as CP2K
        elif self.calc_backend == "cp2k_shell":
            from ase.calculators.cp2k import CP2K as CP2K
        else:
            raise NotImplementedError(f"Unimplemented backend {self.calc_backend} for vasp.")

        # - set some parameters
        calc = CP2K(
            directory=directory, command=command,
        )
        calc.set(inp=inp)
        if "cutoff" in conflict_keys:
            calc.set(cutoff=None) # set in inp
        if "max_scf" in conflict_keys:
            calc.set(max_scf=None)# set in inp
        if "xc" in conflict_keys:
            calc.set(xc=None) # set in inp
        calc.set(**calc_params)
        
        self.calc = calc

        return

if __name__ == "__main__":
    ...