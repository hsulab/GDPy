#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx import config
from gdpx.core.register import registers

# Basic builders and modifiers
from gdpx.builder.direct import DirectBuilder, ReadStruBuilder
registers.builder.register("direct")(DirectBuilder)
registers.builder.register("read_stru")(ReadStruBuilder)

from gdpx.builder.dimer import DimerBuilder
registers.builder.register("dimer")(DimerBuilder)

from gdpx.builder.trimer import TrimerBuilder
registers.builder.register("trimer")(TrimerBuilder)

from gdpx.builder.species import MoleculeBuilder
registers.builder.register("molecule")(MoleculeBuilder)

from gdpx.builder.wulff import WulffConstructionBuilder
registers.builder.register("wulff_construction")(WulffConstructionBuilder)

from gdpx.builder.perturbator import PerturbatorBuilder
registers.builder.register("perturb")(PerturbatorBuilder)

from gdpx.builder.packer import PackerBuilder
registers.builder.register("pack")(PackerBuilder)

from gdpx.builder.graph import GraphExchangeModifier, GraphInsertModifier, GraphRemoveModifier
registers.builder.register("graph_insert")(GraphInsertModifier)
registers.builder.register("graph_remove")(GraphRemoveModifier)
registers.builder.register("graph_exchange")(GraphExchangeModifier)

from gdpx.builder.random_bulk import RandomBulkBuilder, RandomClusterBuilder, RandomSurfaceBuilder
registers.builder.register("random_bulk")(RandomBulkBuilder)
registers.builder.register("random_cluster")(RandomClusterBuilder)
registers.builder.register("random_surface")(RandomSurfaceBuilder)

from gdpx.builder.random_structure import RandomStructureImprovedModifier
registers.builder.register("random_structure_improved")(RandomStructureImprovedModifier)

from gdpx.builder.cleave_surface import CleaveSurfaceModifier, AddVacuumModifier
registers.builder.register("cleave_surface")(CleaveSurfaceModifier)
registers.builder.register("add_vacuum")(AddVacuumModifier)

from gdpx.builder.repeat import RepeatModifier
registers.builder.register("repeat")(RepeatModifier)

# Extra modifiers
from gdpx.builder.deform import DeformModifier
registers.builder.register("deform")(DeformModifier)

from gdpx.builder.scale import ScaleModifier
registers.builder.register("scale")(ScaleModifier)

from gdpx.builder.roulette import RouletteBuilder
registers.builder.register("roulette")(RouletteBuilder)

from gdpx.builder.change_element import ReplaceElementModifier, RemoveElementModifier
registers.builder.register("replace_element")(ReplaceElementModifier)
registers.builder.register("remove_element")(RemoveElementModifier)

from gdpx.builder.composed import ComposedModifier
registers.builder.register("composed")(ComposedModifier)

try:
    from gdpx.builder.scan.angle import ScanAngleModifier
    registers.builder.register("scan_angle")(ScanAngleModifier)

    from gdpx.builder.scan.hypercube import HypercubeBuilder
    registers.builder.register("hypercube")(HypercubeBuilder)

except ImportError as e:
    config._print(f"  {'Builder':<16s} {'`hypercube`':<16s} -> require `{e.name}`.")

# Extra utilities
from gdpx.builder.utils import remove_vacuum, reset_cell
registers.operation.register(remove_vacuum)
registers.operation.register(reset_cell)


if __name__ == "__main__":
    ...
  
