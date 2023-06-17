#!/usr/bin/env python3
# -*- coding: utf-8 -*

from GDPy.core.register import registers

from .mixer import MixerManager
from .plumed import PlumedManager

registers.manager.register(MixerManager)
registers.manager.register(PlumedManager)


if __name__ == "__main__":
    ...