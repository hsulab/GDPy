#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

from ..core.register import registers


""" This submodule is for exploring, sampling, 
    and performing (chemical) reactions with
    various advanced algorithms.
"""

from .pathway import MEPFinder
registers.reactor.register("ase")(MEPFinder)


if __name__ == "__main__":
    ...