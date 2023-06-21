#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import create_builder

from ..core.register import registers
from .direct import DirectBuilder
from .species import MoleculeBuilder
from .dimer import DimerBuilder
from .perturbator import PerturbatorBuilder
from .packer import PackerBuilder
from .graph import GraphInsertModifier, GraphRemoveModifier, GraphExchangeModifier
from .randomBuilder import BulkBuilder, ClusterBuilder, SurfaceBuilder


# - basic builders and modifiers
registers.builder.register("direct")(DirectBuilder)
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


# - extra modifiers
from .zoom import ZoomModifier
registers.builder.register("zoom")(ZoomModifier)


# - optional
try:
    from .hypercube import HypercubeBuilder
    registers.builder.register("hypercube")(HypercubeBuilder)
except ImportError as e:
    ...


if __name__ == "__main__":
    ...