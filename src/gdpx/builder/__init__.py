#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings

from .. import config
from ..utils.command import CustomTimer

from .interface import create_builder

from ..core.register import registers
from .direct import DirectBuilder, ReadBuilder
from .species import MoleculeBuilder
from .dimer import DimerBuilder


# - regions
from .region import (
    AutoRegion, CubeRegion, SphereRegion, CylinderRegion, LatticeRegion,
    SurfaceLatticeRegion, SurfaceRegion,
    IntersectRegion
)
registers.region.register(AutoRegion)
registers.region.register(CubeRegion)
registers.region.register(SphereRegion)
registers.region.register(CylinderRegion)
registers.region.register(LatticeRegion)
registers.region.register(SurfaceLatticeRegion) # BUG?
registers.region.register(SurfaceRegion)        # BUG?
registers.region.register(IntersectRegion)


# - basic builders and modifiers
registers.builder.register("direct")(DirectBuilder)
registers.builder.register("reader")(ReadBuilder)
registers.builder.register("dimer")(DimerBuilder)
registers.builder.register("molecule")(MoleculeBuilder)

from .perturbator import PerturbatorBuilder
registers.builder.register("perturb")(PerturbatorBuilder)

from .packer import PackerBuilder
registers.builder.register("pack")(PackerBuilder)

from .insert import InsertModifier
registers.builder.register("insert")(InsertModifier)

# --
from ..graph.sites import SiteFinder
from ..graph.creator import StruGraphCreator, extract_chem_envs
from ..graph.comparison import get_unique_environments_based_on_bonds, paragroup_unique_chem_envs
from ..graph.utils import unpack_node_name

from .graph import GraphInsertModifier, GraphRemoveModifier, GraphExchangeModifier
registers.builder.register("graph_insert")(GraphInsertModifier)
registers.builder.register("graph_remove")(GraphRemoveModifier)
registers.builder.register("graph_exchange")(GraphExchangeModifier)

# --
from .randomBuilder import BulkBuilder, ClusterBuilder, SurfaceBuilder
registers.builder.register("random_bulk")(BulkBuilder)
registers.builder.register("random_cluster")(ClusterBuilder)
registers.builder.register("random_surface")(SurfaceBuilder)

from .cleave_surface import CleaveSurfaceModifier
registers.builder.register("cleave_surface")(CleaveSurfaceModifier)

from .repeat import RepeatModifier
registers.builder.register("repeat")(RepeatModifier)

# - extra modifiers
from .zoom import ZoomModifier
registers.builder.register("zoom")(ZoomModifier)


# - optional
try:
    from .hypercube import HypercubeBuilder
    registers.builder.register("hypercube")(HypercubeBuilder)
except ImportError as e:
    warnings.warn("Module {} import failed: {}".format("hypercube", e), UserWarning)

# - extra utilities
from .utils import remove_vacuum, reset_cell
registers.operation.register(remove_vacuum)
registers.operation.register(reset_cell)


if __name__ == "__main__":
    ...
