#!/usr/bin/env python3
# -*- coding: utf-8 -*


from ase.calculators.calculator import Calculator, all_properties, all_changes


class DummyCalculator(Calculator):

    name = "dummy"

    def __init__(
        self, restart=None, label="dummy", atoms=None, directory=".", **kwargs
    ):
        super().__init__(
            restart, label=label, atoms=atoms, directory=directory, **kwargs
        )

        return

    def calculate(
        self, atoms=None, properties=all_properties, system_changes=all_changes
    ):
        """"""
        raise NotImplementedError("DummyCalculator is unable to calculate.")


if __name__ == "__main__":
    ...
