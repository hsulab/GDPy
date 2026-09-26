#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.execution.schedulers.factory import canonicalise_scheduler
from gdpx.execution.schedulers.scheduler import BaseScheduler
from gdpx.workflow.session.variable import Variable


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
