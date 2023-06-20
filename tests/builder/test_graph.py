#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tempfile

from ase.io import read, write

from GDPy.builder.species import MoleculeBuilder
from GDPy.builder.graph.insert import GraphInsertModifier
from GDPy.builder.graph.remove import GraphRemoveModifier
from GDPy.builder.graph.exchange import GraphExchangeModifier


MODIFIER_INSERT_PARAMS = dict(
    species = "CO",
    adsorbate_elements = ["C", "O"], # specific species
    sites = [
        dict(
            cn = 1,
            group = [
                "symbol Cu",
                "region 0. 0. 0. -100. -100. 6. 100. 100. 8."
            ],
            radius = 3,
            ads = [
                dict(
                    mode = "atop",
                    distance = 2.0
                )
            ]
        ),
        dict(
            cn = 2,
            group = [
                "symbol Cu",
                "region 0. 0. 0. -100. -100. 6. 100. 100. 8."
            ],
            radius = 3,
            ads = [
                dict(
                    mode = "atop",
                    distance = 2.0
                )
            ]
        ),
        dict(
            cn = 3,
            group = [
                "symbol Cu",
                "region 0. 0. 0. -100. -100. 6. 100. 100. 8."
            ],
            radius = 3,
            ads = [
                dict(
                    mode = "atop",
                    distance = 2.0
                )
            ]
        )
    ],
    graph = dict(
        pbc_grid = [2, 2, 0],
        graph_radius = 2,
        neigh_params = dict(
            covalent_ratio = 1.1,
            skin = 0.25
        )
    )
)

MODIFIER_REMOVE_PARAMS = dict(
    species = "O",
    graph = dict(
        pbc_grid = [2, 2, 0],
        graph_radius = 2,
        neigh_params = dict(
            covalent_ratio = 1.1,
            skin = 0.25
        )
    ),
    adsorbate_elements = ["O"],
    target_indices = [ # py convention
        42, 43, 44, 45, 46, 47, # top surface
        36, 37, 38, 39, 40, 41, # sub surface
    ] 
)

MODIFIER_EXCHANGE_PARAMS = dict(
    species = "Zn",
    target = "Cr",
    graph = dict(
        pbc_grid = [2, 2, 0],
        graph_radius = 2,
        neigh_params = dict(
            # AssertionError: Single atoms group into one adsorbate. 
            # Try reducing the covalent radii. if it sets 1.1.
            covalent_ratio = 1.0,
            skin = 0.25
        )
    ),
    adsorbate_elements = ["Zn", "Cr"],
    target_indices = [ # py convention
        90, 91, 92, 93, 94, 95, # top surface
        84, 85, 86, 87, 88, 89, # sub surface
    ] 
)


def test_insert():
    """"""
    with tempfile.TemporaryDirectory() as tmpdir:
        modifier = GraphInsertModifier(
            **MODIFIER_INSERT_PARAMS
        )
        modifier.directory = tmpdir

        substrates = read("../assets/Cu-fcc-s111p22.xyz", ":")

        structures = modifier.run(substrates=substrates)
        n_structures = len(structures)

    assert n_structures == 4

def test_remove():
    """"""
    with tempfile.TemporaryDirectory() as tmpdir:
        #tmpdir = "./xxx"
        modifier = GraphRemoveModifier(
            **MODIFIER_REMOVE_PARAMS
        )
        modifier.directory = tmpdir

        substrates = read("./ZnO.xyz", ":")

        structures = modifier.run(substrates=substrates)
        n_structures = len(structures)
    
    assert n_structures == 2

def test_exchange():
    """"""
    with tempfile.TemporaryDirectory() as tmpdir:
        #tmpdir = "./xxx"
        modifier = GraphExchangeModifier(
            **MODIFIER_EXCHANGE_PARAMS
        )
        modifier.directory = tmpdir

        substrates = read("./ZnO.xyz", ":")

        structures = modifier.run(substrates=substrates)
        n_structures = len(structures)
    
    assert n_structures == 2
    


if __name__ == "__main__":
    ...