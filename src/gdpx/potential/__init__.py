#!/usr/bin/env python3
# -*- coding: utf-8 -*


from gdpx import config
from gdpx.core.register import registers


# Basic potentials
# MLIP
from .deepmd import DeepmdManager, DeepmdTrainer, DeepmdDataloader
registers.manager.register("deepmd")(DeepmdManager)
registers.trainer.register(DeepmdTrainer)
registers.dataloader.register(DeepmdDataloader)

try:
    from .deepmd import DeepmdJaxManager, DeepmdJaxTrainer
    registers.manager.register("deepmd-jax")(DeepmdJaxManager)
    registers.trainer.register(DeepmdJaxTrainer)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`deepmd_jax`':<16s} -> require `{e.name}`.")

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

from .mattersim import MatterSimManager
registers.manager.register("mattersim")(MatterSimManager)

# DFTs
from .cp2k import Cp2kManager
registers.manager.register("cp2k")(Cp2kManager)

from .espresso import EspressoManager
registers.manager.register("espresso")(EspressoManager)

from .vasp import VaspManager
registers.manager.register("vasp")(VaspManager)

# FFs
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

# Advanced potentials
from .grid import GridManager
registers.manager.register("grid")(GridManager)

from .mixer import MixerManager
registers.manager.register("mixer")(MixerManager)

# Optional potentials
try:
    from .abacus import AbacusManager
    registers.manager.register("abacus")(AbacusManager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`abacus`':<16s} -> require `{e.name}`.")

try:
    from .xtb import XtbManager
    registers.manager.register("xtb")(XtbManager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`xtb`':<16s} -> require `{e.name}`.")

try:
    from .dftd3 import Dftd3Manager
    registers.manager.register("dftd3")(Dftd3Manager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`dftd3`':<16s} -> require `{e.name}`.")

try:
    from .bias import BiasManager
    registers.manager.register("bias")(BiasManager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`bias`':<16s} -> require `{e.name}`.")

try:
    from .plumed.plumed import PlumedManager
    registers.manager.register("plumed")(PlumedManager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`plumed`':<16s} -> require `{e.name}`.")


if __name__ == "__main__":
    ...
