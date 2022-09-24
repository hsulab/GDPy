#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from GDPy.builder.direct import DirectGenerator
from GDPy.builder.species import FormulaBasedGenerator
from GDPy.builder.builder import StructureGenerator
from GDPy.builder.randomBuilder import RandomGenerator
from GDPy.builder.adsorbate import AdsorbateGraphGenerator

supported_filetypes = [".xyz", ".arc"]

def create_generator(params) -> StructureGenerator:
    """"""
    # - parse string
    if isinstance(params, str):
        suffix = params[-4:]
        if suffix in supported_filetypes:
            from ase.io import read, write
            params = dict(
                method = "direct",
                frames = read(params, ":")
            )
        else:
            params = dict(
                method = "formula",
                chemical_formula = params
            )

    # - params dict
    method = params.pop("method", "random")
    if method == "direct":
        generator = DirectGenerator(**params)
    elif method == "formula":
        generator = FormulaBasedGenerator(**params)
    elif method == "random":
        generator = RandomGenerator(params)
    elif method == "adsorbate":
        generator = AdsorbateGraphGenerator(**params)
    else:
        raise RuntimeError("Unknown generator params...")

    return generator
    

if __name__ == "__main__":
    pass