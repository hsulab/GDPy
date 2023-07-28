#!/usr/bin/env python3
# -*- coding: utf-8 -*

from GDPy.core.register import registers

from .mixer import MixerManager
registers.manager.register(MixerManager)

from .plumed import PlumedManager
registers.manager.register(PlumedManager)

from .mace import MaceManager
registers.manager.register(MaceManager)


if __name__ == "__main__":
    ...