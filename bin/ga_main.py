#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pathlib
from typing import overload
import yaml
import argparse
import datetime

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
    "--seed", default=None,
    help="add seed structures to the database"
)

# postprocess
parser.add_argument(
    '--report', action='store_true',
    help='generate report procedure'
)

parser.add_argument(
    "--refine", type=int, default=0,
    help='refine the most promising structures for accurate calculation'
)

# subcommands in the entire workflow 
subparsers = parser.add_subparsers(
    title="available subcommands", 
    dest="subcommand", 
    help="sub-command help"
)

# --- statistics the dataset 
parser_data = subparsers.add_parser(
    "data",
    help="database operations"
)
parser_data.add_argument(
    "-rq", "--remove_queued", default = None, 
    nargs="*", type=int,
    help="remove data in queue from database"
)

args = parser.parse_args()

print("\n\n===== Modified ASE-based Genetic Algorithem Structure Search =====\n\n")

input_file = pathlib.Path(args.INPUT)
# print(input_file.suffix)
if input_file.suffix == ".json":
    with open(input_file, "r") as fopen:
        ga_dict = json.load(fopen)
elif input_file.suffix == ".yaml":
    with open(input_file, "r") as fopen:
        ga_dict = yaml.safe_load(fopen)
else:
    raise ValueError("wrong input file format...")
with open("params.json", "w") as fopen:
    json.dump(ga_dict, fopen, indent=4)
print("See params.json for values of all parameters...")

gae = GeneticAlgorithemEngine(ga_dict)
print("initialise GA engine at ", datetime.datetime.now())

if args.subcommand == "data":
    gae.operate_database(args.remove_queued)

if args.check:
    gae.check_status()

if args.run:
    gae.run(args.seed)

if args.report:
    gae.report()

if args.refine != 0:
    gae.refine(args.refine)

if __name__ == "__main__":
    pass