#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ..core.register import registers
from ..core.variable import Variable
from ..potential.managers.mixer import MixerManager


@registers.variable.register
class PotterVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        name = kwargs.get("name", None)
        potter = registers.create(
            "manager",
            name,
            convert_name=False,
            # **kwargs.get("params", {})
        )
        potter.register_calculator(kwargs.get("params", {}))

        super().__init__(initial_value=potter, directory=directory)

        return


def create_mixer(basic_params, *args, **kwargs):
    """"""
    potters = [basic_params]
    for x in args:
        potters.append(x)
    calc_params = dict(backend="ase", potters=potters)

    mixer = MixerManager()
    mixer.register_calculator(calc_params=calc_params)

    return mixer


if __name__ == "__main__":
    ...
