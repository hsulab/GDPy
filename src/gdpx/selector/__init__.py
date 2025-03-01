#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx import config
from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("selector")

from .compare import CompareSelector

REGISTER.register("compare")(CompareSelector)

from .interval import IntervalSelector

REGISTER.register("interval")(IntervalSelector)

from .invariant import InvariantSelector

REGISTER.register("invariant")(InvariantSelector)

from .locate import LocateSelector

REGISTER.register("locate")(LocateSelector)

from .property import PropertySelector

REGISTER.register("property")(PropertySelector)

from .random import RandomSelector

REGISTER.register("random")(RandomSelector)

from .scf import ScfSelector

REGISTER.register("scf")(ScfSelector)

try:
    # This selector depends on an external package dscribe.
    from .descriptor import DescriptorSelector

    REGISTER.register("descriptor")(DescriptorSelector)
except ImportError as e:
    config._print(f"  {'Selector':<16s} {'`descriptor`':<16s} -> require `{e.name}`.")


if __name__ == "__main__":
    ...
