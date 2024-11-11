#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy 
from typing import List

from .monte_carlo import MonteCarlo


class BasinHopping(MonteCarlo):

    def __init__(
        self, builder: dict, operators: List[dict], convergence: dict, *args, **kwargs
    ) -> None:
        """"""
        super().__init__(
            builder=builder, operators=operators, convergence=convergence, *args, **kwargs
        )

        return
    
    def as_dict(self) -> dict:
        """"""
        engine_params = super().as_dict()
        engine_params["method"] = "basin_hopping"

        return engine_params


if __name__ == "__main__":
    ...
