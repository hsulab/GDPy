#!/usr/bin/env python3
# -*- coding: utf-8 -*


import pathlib


from . import AbstractPotentialManager, DummyCalculator


class EspressoManager(AbstractPotentialManager):

    name = "espresso"

    implemented_backends = ["espresso"]
    valid_combinations = (
        ("espresso", "ase")
    )

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__()

        return
    
    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        # - check backends
        super().register_calculator(calc_params, *agrs, **kwargs)

        # - parse params
        command = calc_params.get("command", "pw.x -in PREFIX.pwi > PREFIX.pwo")
        self.calc_params.update(command = command)

        pp_path = pathlib.Path(calc_params.pop("pp_path", "./")).resolve()
        if not pp_path.exists():
            raise FileNotFoundError("Pseudopotentials for espresso does not exist.")
        self.calc_params.update(pp_path=str(pp_path))

        pp_name = calc_params.get("pp_name", None)
        if pp_name is None:
            raise RuntimeError("Must set name for pseudopotentials.")

        template = calc_params.pop("template", "./espresso.pwi")
        template = pathlib.Path(template)
        if not template.exists():
            raise FileNotFoundError("Template espresso input file does not exist.")
        self.calc_params.update(template=str(template))

        from gdpx.computation.espresso import EspressoParser
        ep = EspressoParser(template=template)

        # -- kpoints that maybe read from the file
        kpts = calc_params.pop("kpts", None)
        kspacing = calc_params.pop("kspacing", None)
        koffset = calc_params.pop("koffset", 0)

        assert kpts is None or kspacing is None, "Cannot set kpts and kspacing at the same time."

        # - NOTE: check
        calc = DummyCalculator()

        if self.calc_backend == "espresso":
            from gdpx.computation.espresso import Espresso
            calc = Espresso(
                command = command,
                input_data = ep.parameters,
                pseudopotentials = pp_name,
                kspacing = kspacing,
                kpts = kpts,
                koffset = koffset
            )
        else:
            ...
        
        self.calc = calc

        return


if __name__ == "__main__":
    ...