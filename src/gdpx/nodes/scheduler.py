#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from gdpx.session.registry import workflow_registers as registers
from gdpx.factory.scheduler import canonicalise_scheduler
from gdpx.scheduler.scheduler import BaseScheduler
from gdpx.session.variable import Variable


@registers.variable.register
class SchedulerVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        scheduler = canonicalise_scheduler(kwargs)
        super().__init__(initial_value=scheduler, directory=directory)

        return

    @property
    def value(self) -> BaseScheduler:
        """"""

        return self._value  # type: ignore


if __name__ == "__main__":
    ...
