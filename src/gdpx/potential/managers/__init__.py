#!/usr/bin/env python3
# -*- coding: utf-8 -*


import warnings

from .. import registers

# - potentials
from .deepmd import DeepmdManager, DeepmdTrainer
registers.manager.register(DeepmdManager)
registers.trainer.register(DeepmdTrainer)

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


# - trainers
from .gp.fgp import FGPTrainer
registers.trainer.register("FgpTrainer")(FGPTrainer)

from .gp.sgp import SGPTrainer
registers.trainer.register("SgpTrainer")(SGPTrainer)



if __name__ == "__main__":
    ...