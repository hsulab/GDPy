#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse

from GDPy.ga.init_ga import GeneticAlgorithemEngine

parser = argparse.ArgumentParser()
parser.add_argument(
    'INPUT', default='ga.json', help='genetic algorithem inputs'
)

args = parser.parse_args()

with open(args.INPUT, 'r') as fopen:
    ga_dict = json.load(fopen)

gae = GeneticAlgorithemEngine(ga_dict)
gae.run()

if __name__ == "__main__":
    pass