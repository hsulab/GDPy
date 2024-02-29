#!/usr/bin/env python3
# -*- coding: utf-8 -*


import warnings

from .. import registers
from ..manager import AbstractPotentialManager, DummyCalculator
from ..trainer import AbstractTrainer

# - basic potentials
# -- MLIP
from .deepmd import DeepmdManager, DeepmdTrainer, DeepmdDataloader
registers.manager.register(DeepmdManager)
registers.trainer.register(DeepmdTrainer)
registers.dataloader.register(DeepmdDataloader)

from .eann import EannManager, EannTrainer
registers.manager.register(EannManager)
registers.trainer.register(EannTrainer)

from .lasp import LaspManager
registers.manager.register(LaspManager)

from .mace import MaceManager, MaceTrainer
registers.manager.register(MaceManager)
registers.trainer.register(MaceTrainer)

from .nequip import NequipManager, NequipTrainer
registers.manager.register(NequipManager)
registers.trainer.register(NequipTrainer)

#try:
#    from .schnet import SchnetManager
#    registers.manager.register(SchnetManager)
#except ImportError as e:
#    warnings.warn("Module {} import failed: {}".format("schnet", e), UserWarning)

# -- reference potentials
# --- DFT
from .cp2k import Cp2kManager
registers.manager.register(Cp2kManager)

from .espresso import EspressoManager
registers.manager.register(EspressoManager)

from .vasp import VaspManager
registers.manager.register(VaspManager)

# --- FFs
from .eam import EamManager
registers.manager.register(EamManager)

from .emt import EmtManager
registers.manager.register(EmtManager)

from .reax import ReaxManager
registers.manager.register(ReaxManager)

# - advanced potentials
from .grid import GridManager
registers.manager.register(GridManager)

from .mixer import MixerManager
registers.manager.register(MixerManager)

# - optional potentials
try:
    from .xtb import XtbManager
    registers.manager.register(XtbManager)
except ImportError as e:
    warnings.warn("Module {} import failed: {}".format("xtb", e), UserWarning)

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


# - trainers
from .gp.fgp import FGPTrainer
registers.trainer.register("FgpTrainer")(FGPTrainer)

from .gp.sgp import SGPTrainer
registers.trainer.register("SgpTrainer")(SGPTrainer)



if __name__ == "__main__":
    ...