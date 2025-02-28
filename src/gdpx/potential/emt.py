#!/usr/bin/env python3
# -*- coding: utf-8 -*


from gdpx.backend.ase import DummyCalculator
from .manager import BasePotentialManager


class EmtManager(BasePotentialManager):

    name = "emt"

    implemented_backends = ("ase",)
    valid_combinations = (
        ("ase", "ase"),
    )

    """See ASE documentation for calculator parameters.
    """

    def register_calculator(self, calc_params: dict, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        # The emt backend is just an alias of ase backend, they are the same.
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            from ase.calculators.emt import EMT

            calc = EMT(**calc_params)
        else:
            ...  # The backend has already been checked.

        self.calc = calc

        return


if __name__ == "__main__":
    ...
