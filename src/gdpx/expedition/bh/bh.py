#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy 
from typing import List

from .. import registers, Variable, SingleWorker

from ..expedition import AbstractExpedition
from ..mc.mc import MonteCarlo, MonteCarloVariable


class BasinHoppingVariable(MonteCarloVariable):

    ...



class BasinHopping(MonteCarlo):

    def __init__(
        self, builder: dict, operators: List[dict], 
        convergence: dict, random_seed=None, dump_period: int=1, ckpt_period: int=100,
        restart: bool = False, directory="./", *args, **kwargs
    ) -> None:
        """"""
        super().__init__(
            builder, operators, convergence, random_seed, dump_period, ckpt_period,
            restart, directory, *args, **kwargs
        )

        return


if __name__ == "__main__":
    ...