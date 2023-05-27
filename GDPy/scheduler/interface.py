#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GDPy.core.variable import Variable
from GDPy.core.register import registers
from GDPy.scheduler import create_scheduler


@registers.variable.register
class SchedulerVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        scheduler = create_scheduler(kwargs)
        super().__init__(initial_value=scheduler, directory=directory)

        return


if __name__ == "__main__":
    ...