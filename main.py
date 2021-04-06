#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse
from pathlib import Path

import numpy as np


def _main():
    # arguments 
    parser = argparse.ArgumentParser(
        prog='gdp', 
        description='GDPy: Generating DeepMD Potential with Python'
    )
    
    # the workflow tracker
    parser.add_argument(
        '-s', '--status', 
        help='pickle file with info on the current workflow'
    )
    
    # subcommands in the entire workflow 
    subparsers = parser.add_subparsers(
        title='available subcommands', 
        dest='subcommand', 
        help='sub-command help'
    )
    
    # rss
    parser_rss = subparsers.add_parser(
        'train', help='automatic training with an iterative process'
    )
    parser_rss.add_argument(
        'INPUTS',
        help='a directory with input json files'
    )

    # others
 
    
    # === execute 
    args = parser.parse_args()

    # always check the current workflow before continuing to subcommands 
    # also, the global logger will be initialised 
    # TODO: track the workflow 
    # tracker = track_workflow(args.status)

    # use subcommands
    if args.subcommand == 'train':
        iterative_train(args.INPUTS)
    else:
        pass

from ase.io import read, write
from ase.build import make_supercell

from .pertubater import pertubate_stucture

from .trainer.train_potential import read_dptrain_json
def main():
    read_dptrain_json(num_models=4)
    pass


if __name__ == '__main__':
    main()
