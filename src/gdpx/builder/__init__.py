#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings

from .. import config
from ..core.register import registers
from ..utils.command import CustomTimer
from ..utils.strconv import str2array
from ..utils.atoms_tags import get_tags_per_species



# - basic builders and modifiers
from .direct import DirectBuilder, ReadBuilder

registers.builder.register("direct")(DirectBuilder)
registers.builder.register("reader")(ReadBuilder)

from .dimer import DimerBuilder

registers.builder.register("dimer")(DimerBuilder)

from .trimer import TrimerBuilder

registers.builder.register("trimer")(TrimerBuilder)

from .species import MoleculeBuilder

registers.builder.register("molecule")(MoleculeBuilder)

from .wulff import WulffConstructionBuilder

registers.builder.register("wulff_construction")(WulffConstructionBuilder)

from .perturbator import PerturbatorBuilder

registers.builder.register("perturb")(PerturbatorBuilder)

from .packer import PackerBuilder

registers.builder.register("pack")(PackerBuilder)

from .insert import InsertModifier

registers.builder.register("insert")(InsertModifier)

from ..graph.comparison import (
    get_unique_environments_based_on_bonds,
    paragroup_unique_chem_envs,
)
from ..graph.creator import StruGraphCreator, extract_chem_envs

# --
from ..graph.sites import SiteFinder
from ..graph.utils import unpack_node_name
from .graph import GraphExchangeModifier, GraphInsertModifier, GraphRemoveModifier

registers.builder.register("graph_insert")(GraphInsertModifier)
registers.builder.register("graph_remove")(GraphRemoveModifier)
registers.builder.register("graph_exchange")(GraphExchangeModifier)

# --
from .randomBuilder import BulkBuilder, ClusterBuilder, SurfaceBuilder

registers.builder.register("random_bulk")(BulkBuilder)
registers.builder.register("random_cluster")(ClusterBuilder)
registers.builder.register("random_surface")(SurfaceBuilder)

from .random_surface_variable import RandomSurfaceVariableModifier

registers.builder.register("random_surface_variable")(RandomSurfaceVariableModifier)

from .cleave_surface import CleaveSurfaceModifier, AddVacuumModifier

registers.builder.register("cleave_surface")(CleaveSurfaceModifier)
registers.builder.register("add_vacuum")(AddVacuumModifier)

from .repeat import RepeatModifier

registers.builder.register("repeat")(RepeatModifier)

# - extra modifiers
from .zoom import ZoomModifier

registers.builder.register("zoom")(ZoomModifier)

from .scale import ScaleModifier

registers.builder.register("scale")(ScaleModifier)

from .roulette import RouletteBuilder

registers.builder.register("roulette")(RouletteBuilder)

from .change_element import ReplaceElementModifier, RemoveElementModifier

registers.builder.register("replace_element")(ReplaceElementModifier)
registers.builder.register("remove_element")(RemoveElementModifier)

from .composed import ComposedModifier

registers.builder.register("composed")(ComposedModifier)


# - optional
try:
    from .scan.angle import ScanAngleModifier

    registers.builder.register("scan_angle")(ScanAngleModifier)
    from .scan.hypercube import HypercubeBuilder

    registers.builder.register("hypercube")(HypercubeBuilder)
except ImportError as e:
    config._print(f"Builder {'hypercube'} import failed: {e}")

# - extra utilities
from .utils import remove_vacuum, reset_cell

registers.operation.register(remove_vacuum)
registers.operation.register(reset_cell)


if __name__ == "__main__":
    ...
