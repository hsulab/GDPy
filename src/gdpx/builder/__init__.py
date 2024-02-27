#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings

from .interface import create_builder

from ..core.register import registers
from .direct import DirectBuilder, ReadBuilder
from .species import MoleculeBuilder
from .dimer import DimerBuilder
from .perturbator import PerturbatorBuilder
from .packer import PackerBuilder
from .graph import GraphInsertModifier, GraphRemoveModifier, GraphExchangeModifier
from .randomBuilder import BulkBuilder, ClusterBuilder, SurfaceBuilder


# - basic builders and modifiers
registers.builder.register("direct")(DirectBuilder)
registers.builder.register("reader")(ReadBuilder)
registers.builder.register("dimer")(DimerBuilder)
registers.builder.register("molecule")(MoleculeBuilder)
registers.builder.register("perturb")(PerturbatorBuilder)
registers.builder.register("pack")(PackerBuilder)
registers.builder.register("graph_insert")(GraphInsertModifier)
registers.builder.register("graph_remove")(GraphRemoveModifier)
registers.builder.register("graph_exchange")(GraphExchangeModifier)
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
