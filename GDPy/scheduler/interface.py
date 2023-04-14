#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GDPy.core.variable import Variable
from GDPy.scheduler import create_scheduler

class SchedulerNode(Variable):

    def __init__(self, **kwargs):
        """"""
        scheduler = create_scheduler(kwargs)
        super().__init__(scheduler)

        return


if __name__ == "__main__":
    ...