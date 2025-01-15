#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from gdpx.core.variable import Variable
from gdpx.core.register import registers

from .scheduler import AbstractScheduler


@registers.variable.register
class SchedulerVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        scheduler_params = copy.deepcopy(kwargs)
        backend = scheduler_params.pop("backend", "local")
        scheduler = registers.create(
            "scheduler", backend, convert_name=True, **scheduler_params
        )
        super().__init__(initial_value=scheduler, directory=directory)

        return
    
    @property
    def value(self) -> AbstractScheduler:
        """"""

        return self._value  # type: ignore


if __name__ == "__main__":
    ...
