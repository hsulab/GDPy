#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings

from ..core.register import registers

# from .basin import BasinSelector
# registers.selector.register(BasinSelector)

from .compare import CompareSelector

registers.selector.register("compare")(CompareSelector)

from .interval import IntervalSelector

registers.selector.register("interval")(IntervalSelector)

from .invariant import InvariantSelector

registers.selector.register("invariant")(InvariantSelector)

from .locate import LocateSelector

registers.selector.register("locate")(LocateSelector)

from .property import PropertySelector

registers.selector.register("property")(PropertySelector)

from .random import RandomSelector

registers.selector.register("random")(RandomSelector)

from .scf import ScfSelector

registers.selector.register("scf")(ScfSelector)

try:
    # TODO: This selector depends on an external package dscribe.
    from .descriptor import DescriptorSelector

    registers.selector.register("descriptor")(DescriptorSelector)
except ImportError as e:
    warnings.warn(f"Module DescriptorSelector import failed: {e}", UserWarning)


if __name__ == "__main__":
    ...
