#!/usr/bin/env python3
# -*- coding: utf-8 -*


import warnings

from GDPy.core.register import registers

from .mixer import MixerManager
registers.manager.register(MixerManager)

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