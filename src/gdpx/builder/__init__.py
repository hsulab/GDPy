#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx import config
from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("builder")

# Basic builders and modifiers
from .direct import DirectBuilder, ReadStruBuilder

REGISTER.register("direct")(DirectBuilder)
REGISTER.register("read_stru")(ReadStruBuilder)

from .dimer import DimerBuilder

REGISTER.register("dimer")(DimerBuilder)

from .trimer import TrimerBuilder

REGISTER.register("trimer")(TrimerBuilder)

from .species import MoleculeBuilder

REGISTER.register("molecule")(MoleculeBuilder)

from .wulff import WulffConstructionBuilder

REGISTER.register("wulff_construction")(WulffConstructionBuilder)

from .perturbator import PerturbatorBuilder

REGISTER.register("perturb")(PerturbatorBuilder)

from .packer import PackerBuilder

REGISTER.register("pack")(PackerBuilder)

from .graph import GraphExchangeModifier, GraphInsertModifier, GraphRemoveModifier

REGISTER.register("graph_insert")(GraphInsertModifier)
REGISTER.register("graph_remove")(GraphRemoveModifier)
REGISTER.register("graph_exchange")(GraphExchangeModifier)

from .random_bulk import RandomBulkBuilder, RandomClusterBuilder, RandomSurfaceBuilder

REGISTER.register("random_bulk")(RandomBulkBuilder)
REGISTER.register("random_cluster")(RandomClusterBuilder)
REGISTER.register("random_surface")(RandomSurfaceBuilder)

from .random_structure import RandomStructureImprovedModifier

REGISTER.register("random_structure_improved")(RandomStructureImprovedModifier)

from .adsorb import AdsorbateInsertionModifier

REGISTER.register("adsorbate_insertion")(AdsorbateInsertionModifier)

from .cleave_surface import AddVacuumModifier, CleaveSurfaceModifier

REGISTER.register("cleave_surface")(CleaveSurfaceModifier)
REGISTER.register("add_vacuum")(AddVacuumModifier)

from .repeat import RepeatModifier

REGISTER.register("repeat")(RepeatModifier)

# Extra modifiers
from .deform import DeformModifier

REGISTER.register("deform")(DeformModifier)

from .scale import ScaleModifier

REGISTER.register("scale")(ScaleModifier)

from .roulette import RouletteBuilder

REGISTER.register("roulette")(RouletteBuilder)

from .change_element import RemoveElementModifier, ReplaceElementModifier

REGISTER.register("replace_element")(ReplaceElementModifier)
REGISTER.register("remove_element")(RemoveElementModifier)

from .composed import ComposedModifier

REGISTER.register("composed")(ComposedModifier)

try:
    from .scan.angle import ScanAngleModifier

    REGISTER.register("scan_angle")(ScanAngleModifier)

    from .scan.hypercube import HypercubeBuilder

    REGISTER.register("hypercube")(HypercubeBuilder)

except ImportError as e:
    config._print(f"  {'Builder':<16s} {'`hypercube`':<16s} -> require `{e.name}`.")


if __name__ == "__main__":
    ...
