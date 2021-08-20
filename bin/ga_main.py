#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse

from GDPy.ga.engine import GeneticAlgorithemEngine

parser = argparse.ArgumentParser()
parser.add_argument(
    'INPUT', default='ga.json', help='genetic algorithem inputs'
)
parser.add_argument(
    '-r', '--run', action='store_true',
    help='run GA procedure'
)
parser.add_argument(
    '-c', '--check', action='store_true',
    help='check status'
)
parser.add_argument(
    '--report', action='store_true',
    help='generate report procedure'
)

parser.add_argument(
    '--make', type=int, default=0,
    help='make the most promising structures for accurate calculation'
)

args = parser.parse_args()

with open(args.INPUT, 'r') as fopen:
    ga_dict = json.load(fopen)

gae = GeneticAlgorithemEngine(ga_dict)
print('initialise GA engine...')

if args.check:
    gae.check_status()

if args.run:
    gae.run()

if args.report:
    gae.report()

if args.make != 0:
    gae.make(args.make)

if __name__ == "__main__":
    pass