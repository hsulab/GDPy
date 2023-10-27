#!/usr/bin/env python3
# -*- coding: utf-8 -*


import warnings

from .. import registers

from .grid import GridManager
registers.manager.register(GridManager)

from .mixer import MixerManager
registers.manager.register(MixerManager)

try:
    from .dftd3 import Dftd3Manager
    registers.manager.register(Dftd3Manager)
except ImportError as e:
    warnings.warn("Module {} import failed: {}".format("dftd3", e), UserWarning)

try:
    from .bias import BiasManager
    registers.manager.register(BiasManager)
except ImportError as e:
    warnings.warn("Module {} import failed: {}".format("bias", e), UserWarning)

from .plumed import PlumedManager
registers.manager.register(PlumedManager)

from .mace import MaceManager
registers.manager.register(MaceManager)


if __name__ == "__main__":
    ...