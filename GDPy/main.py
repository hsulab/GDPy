#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse
from pathlib import Path

import numpy as np


def main():
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
    
    # automatic training
    parser_train = subparsers.add_parser(
        'train', help='automatic training with an iterative process'
    )
    parser_train.add_argument(
        'INPUTS',
        help='a directory with input json files'
    )

    # sample
    parser_sample = subparsers.add_parser(
        'sample', help='prepare/execute/analyse training samples'
    )
    parser_train.add_argument(
        'INPUTS',
        help='a directory with input json files'
    )
    
    # === execute 
    args = parser.parse_args()

    # always check the current workflow before continuing to subcommands 
    # also, the global logger will be initialised 
    # TODO: track the workflow 
    # tracker = track_workflow(args.status)

    # use subcommands
    if args.subcommand == 'train':
        iterative_train(args.INPUTS)
    elif args.subcommand == 'sample':
        iterative_train(args.INPUTS)
    else:
        pass


if __name__ == '__main__':
    main()
