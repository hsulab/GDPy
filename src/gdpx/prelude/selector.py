#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx import config
from gdpx.core.register import registers

# from .basin import BasinSelector
# registers.selector.register(BasinSelector)

from gdpx.selector.compare import CompareSelector

registers.selector.register("compare")(CompareSelector)

from gdpx.selector.interval import IntervalSelector

registers.selector.register("interval")(IntervalSelector)

from gdpx.selector.invariant import InvariantSelector

registers.selector.register("invariant")(InvariantSelector)

from gdpx.selector.locate import LocateSelector

registers.selector.register("locate")(LocateSelector)

from gdpx.selector.property import PropertySelector

registers.selector.register("property")(PropertySelector)

from gdpx.selector.random import RandomSelector

registers.selector.register("random")(RandomSelector)

from gdpx.selector.scf import ScfSelector

registers.selector.register("scf")(ScfSelector)

try:
    # This selector depends on an external package dscribe.
    from gdpx.selector.descriptor import DescriptorSelector

    registers.selector.register("descriptor")(DescriptorSelector)
except ImportError as e:
    config._print(f"  {'Selector':<16s} {'`descriptor`':<16s} -> require `{e.name}`.")


if __name__ == "__main__":
    ...
  
