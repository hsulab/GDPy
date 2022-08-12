#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GDPy.builder.builder import StructureGenerator
from GDPy.builder.randomBuilder import RandomGenerator
from GDPy.builder.adsorbate import AdsorbateGraphGenerator


def create_generator(params) -> StructureGenerator:
    """"""
    method = params.pop("method", "random")
    if method == "random":
        generator = RandomGenerator
    elif method == "adsorbate":
        generator = AdsorbateGraphGenerator

    return generator
    

if __name__ == "__main__":
    pass