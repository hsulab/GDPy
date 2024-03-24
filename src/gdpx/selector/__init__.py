#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings


from ..core.register import registers

from .basin import BasinSelector

registers.selector.register(BasinSelector)

from .compare import CompareSelector

registers.selector.register(CompareSelector)

from .interval import IntervalSelector

registers.selector.register(IntervalSelector)

from .invariant import InvariantSelector

registers.selector.register(InvariantSelector)

from .locate import LocateSelector

registers.selector.register(LocateSelector)

from .property import PropertySelector

registers.selector.register(PropertySelector)

from .random import RandomSelector

registers.selector.register(RandomSelector)

from .scf import ScfSelector

registers.selector.register(ScfSelector)

try:
    # TODO: This selector depends on an external package dscribe.
    from .descriptor import DescriptorSelector

    registers.selector.register(DescriptorSelector)
except ImportError as e:
    warnings.warn(f"Module DescriptorSelector import failed: {e}", UserWarning)


if __name__ == "__main__":
    ...
