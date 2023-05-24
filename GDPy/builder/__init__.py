#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union, List

from GDPy.builder.direct import DirectBuilder
from GDPy.builder.species import FormulaBasedGenerator
from GDPy.builder.dimer import DimerBuilder
from GDPy.builder.builder import StructureGenerator
from GDPy.builder.adsorbate import AdsorbateGraphGenerator

supported_filetypes = [".xyz", ".xsd", ".arc"]
supproted_configtypes = ["json", "yaml"]

def create_generator(params: Union[str, dict]) -> StructureGenerator:
    """"""
    # - parse string
    if isinstance(params, (str,Path)):
        params = str(params)
        suffix = params[-4:]
        if suffix in supported_filetypes:
            params = dict(
                method = "direct",
                frames = params
            )
        elif suffix in supproted_configtypes:
            from GDPy.utils.command import parse_input_file
            params = parse_input_file(params)
        else:
            params = dict(
                method = "formula",
                chemical_formula = params
            )

    # - params dict
    method = params.pop("method", "random")
    if method == "direct":
        generator = DirectBuilder(**params)
    elif method == "formula":
        generator = FormulaBasedGenerator(**params)
    elif method == "dimer":
        generator = DimerBuilder(params)
    elif method == "adsorbate":
        generator = AdsorbateGraphGenerator(params)
    else:
        raise RuntimeError("Unknown generator params...")

    return generator

def create_generators(params: Union[str, dict]) -> List[StructureGenerator]:
    """"""
    # - parse string
    if isinstance(params, (str,Path)):
        params = str(params)
        suffix = params[-4:]
        if suffix in supported_filetypes:
            params = dict(
                method = "direct",
                frames = params
            )
        elif suffix in supproted_configtypes:
            from GDPy.utils.command import parse_input_file
            params = parse_input_file(params)
        else:
            params = dict(
                method = "formula",
                chemical_formula = params
            )

    # - params dict
    method = params.pop("method", "random")
    repeat = params.pop("repeat", 1) # generators would run in tandem

    generators = []
    for i in range(repeat):
        if method == "direct":
            generator = DirectBuilder(**params)
        elif method == "formula":
            generator = FormulaBasedGenerator(**params)
        elif method == "dimer":
            generator = DimerBuilder(params)
        elif method == "adsorbate":
            generator = AdsorbateGraphGenerator(params)
        else:
            raise RuntimeError("Unknown generator params...")
        generators.append(generator)

    return generators
    

if __name__ == "__main__":
    pass