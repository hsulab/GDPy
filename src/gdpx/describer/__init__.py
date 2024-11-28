#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module aims to offer a unified interface to the computation of descriptors."""

from ..core.register import registers

from .interface import DescriberVariable, describe
registers.variable.register(DescriberVariable)
registers.operation.register(describe)

from .spc import SpcDescriber
registers.describer.register("spc")(SpcDescriber)

from .coordinate import CoordinateDescriber
registers.describer.register("coordinate")(CoordinateDescriber)

from .coordination import CoordinationDescriber
registers.describer.register("coordination")(CoordinationDescriber)

from .connectivity import ConnectivityDescriber
registers.describer.register("connectivity")(ConnectivityDescriber)


if __name__ == "__main__":
    ...
