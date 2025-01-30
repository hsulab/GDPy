#!/usr/bin/env python3
# -*- coding: utf-8 -*


from .. import config
from .. import registers
from ..manager import AbstractPotentialManager, BasePotentialManager
from ..trainer import AbstractTrainer

from ..calculators.dummy import DummyCalculator
from ..calculators.mixer import CommitteeCalculator

from .. import remove_extra_stream_handlers


# - basic potentials
# -- MLIP
from .deepmd import DeepmdManager, DeepmdTrainer, DeepmdDataloader
registers.manager.register("deepmd")(DeepmdManager)
registers.trainer.register(DeepmdTrainer)
registers.dataloader.register(DeepmdDataloader)

try:
    from .deepmd import DeepmdJaxManager, DeepmdJaxTrainer
    registers.manager.register("deepmd-jax")(DeepmdJaxManager)
    registers.trainer.register(DeepmdJaxTrainer)
except ImportError as e:
    config._print(f"Potential `deepmd_jax` import failed: {e}")

from .reann.beann import BeannManager, BeannTrainer
registers.manager.register("beann")(BeannManager)
registers.trainer.register(BeannTrainer)

from .reann.reann import ReannManager, ReannTrainer, ReannDataloader
registers.manager.register("reann")(ReannManager)
registers.trainer.register(ReannTrainer)
registers.dataloader.register(ReannDataloader)

from .lasp import LaspManager
registers.manager.register("lasp")(LaspManager)

from .mace import MaceManager, MaceTrainer, MaceDataloader
registers.manager.register("mace")(MaceManager)
registers.trainer.register(MaceTrainer)
registers.dataloader.register(MaceDataloader)

from .nequip import NequipManager, NequipTrainer
registers.manager.register("nequip")(NequipManager)
registers.trainer.register(NequipTrainer)

#try:
#    from .schnet import SchnetManager
#    registers.manager.register(SchnetManager)
#except ImportError as e:
#    warnings.warn("Module {} import failed: {}".format("schnet", e), UserWarning)

from .mattersim import MatterSimManager
registers.manager.register("mattersim")(MatterSimManager)

# -- reference potentials
# --- DFT
from .cp2k import Cp2kManager
registers.manager.register("cp2k")(Cp2kManager)

from .espresso import EspressoManager
registers.manager.register("espresso")(EspressoManager)

from .vasp import VaspManager
registers.manager.register("vasp")(VaspManager)

# --- FFs
from .asepot import AsePotManager
registers.manager.register("ase")(AsePotManager)

from .classic import ClassicManager
registers.manager.register("classic")(ClassicManager)

from .eam import EamManager
registers.manager.register("eam")(EamManager)

from .emt import EmtManager
registers.manager.register("emt")(EmtManager)

from .reax import ReaxManager
registers.manager.register("reax")(ReaxManager)

# - advanced potentials
from .grid import GridManager
registers.manager.register("grid")(GridManager)

from .mixer import MixerManager
registers.manager.register("mixer")(MixerManager)

# - optional potentials
try:
    from .abacus import AbacusManager
    registers.manager.register("abacus")(AbacusManager)
except ImportError as e:
    config._print(f"Potential {'abacus'} import failed: {e}")

try:
    from .xtb import XtbManager
    registers.manager.register("xtb")(XtbManager)
except ImportError as e:
    warnings.warn("Module {} import failed: {}".format("xtb", e), UserWarning)

try:
    from .dftd3 import Dftd3Manager
    registers.manager.register("dftd3")(Dftd3Manager)
except ImportError as e:
    config._print(f"Potential {'dftd3'} import failed: {e}")

try:
    from .bias import BiasManager
    registers.manager.register("bias")(BiasManager)
except ImportError as e:
    config._print(f"Potential {'bias'} import failed: {e}")

try:
    from .plumed.plumed import PlumedManager
    registers.manager.register("plumed")(PlumedManager)
except ImportError as e:
    config._print(f"Potential {'plumed'} import failed: {e}")

# - trainers
from .gp.fgp import FGPTrainer
registers.trainer.register("FgpTrainer")(FGPTrainer)

from .gp.sgp import SGPTrainer
registers.trainer.register("SgpTrainer")(SGPTrainer)



if __name__ == "__main__":
    ...
