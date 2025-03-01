#!/usr/bin/env python3
# -*- coding: utf-8 -*


from gdpx import config
from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("manager")

# Basic potentials
# MLIP
from .deepmd import DeepmdManager

REGISTER.register("deepmd")(DeepmdManager)

try:
    from .deepmd import DeepmdJaxManager

    REGISTER.register("deepmd-jax")(DeepmdJaxManager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`deepmd_jax`':<16s} -> require `{e.name}`.")

from .reann.beann import BeannManager

REGISTER.register("beann")(BeannManager)

from .reann.reann import ReannManager

REGISTER.register("reann")(ReannManager)

from .lasp import LaspManager

REGISTER.register("lasp")(LaspManager)

from .mace import MaceManager

REGISTER.register("mace")(MaceManager)

from .nequip import NequipManager

REGISTER.register("nequip")(NequipManager)

from .mattersim import MatterSimManager

REGISTER.register("mattersim")(MatterSimManager)

# DFTs
from .cp2k import Cp2kManager

REGISTER.register("cp2k")(Cp2kManager)

from .espresso import EspressoManager

REGISTER.register("espresso")(EspressoManager)

from .vasp import VaspManager

REGISTER.register("vasp")(VaspManager)

# FFs
from .asepot import AsePotManager

REGISTER.register("ase")(AsePotManager)

from .classic import ClassicManager

REGISTER.register("classic")(ClassicManager)

from .eam import EamManager

REGISTER.register("eam")(EamManager)

from .emt import EmtManager

REGISTER.register("emt")(EmtManager)

from .reax import ReaxManager

REGISTER.register("reax")(ReaxManager)

# Advanced potentials
from .grid import GridManager

REGISTER.register("grid")(GridManager)

from .mixer import MixerManager

REGISTER.register("mixer")(MixerManager)

# Optional potentials
try:
    from .abacus import AbacusManager

    REGISTER.register("abacus")(AbacusManager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`abacus`':<16s} -> require `{e.name}`.")

try:
    from .xtb import XtbManager

    REGISTER.register("xtb")(XtbManager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`xtb`':<16s} -> require `{e.name}`.")

try:
    from .dftd3 import Dftd3Manager

    REGISTER.register("dftd3")(Dftd3Manager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`dftd3`':<16s} -> require `{e.name}`.")

try:
    from .bias import BiasManager

    REGISTER.register("bias")(BiasManager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`bias`':<16s} -> require `{e.name}`.")

try:
    from .plumed.plumed import PlumedManager

    REGISTER.register("plumed")(PlumedManager)
except ImportError as e:
    config._print(f"  {'Potential':<16s} {'`plumed`':<16s} -> require `{e.name}`.")


if __name__ == "__main__":
    ...
