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

    # explore
    parser_explore = subparsers.add_parser(
        'explore', help='create input files for exploration'
    )
    parser_explore.add_argument(
        '-e', '--exploration', required=True,
        help='input json files with exploration parameters'
    )
    parser_explore.add_argument(
        '-p', '--potential', required=True,
        help='potential-related input json'
    )
    parser_explore.add_argument(
        '-s', '--step', required=True,
        help='exploration steps (create/collect/select)'
    )

    # ase calculator interface
    parser_ase = subparsers.add_parser(
        'ase', help='use ase calculator interface in command line'
    )
    parser_ase.add_argument(
        'INPUTS',
        help='input json file with calculation parameters'
    )

    # cur from dprss
    parser_cur = subparsers.add_parser(
        'cur',
        help='calculate features and selected configuration by CUR decomposition'
    )
    parser_cur.add_argument(
        '-i', '--input', required=True,
        help='(random/trajectory) filename (in xyz format)'
    )
    parser_cur.add_argument(
        '-d', '--descriptor', required=True,
        help='descriptor hyperparameter in json file (only support SOAP now)'
    )
    parser_cur.add_argument(
        '-n', '--number', default=100, type=int,
        help='number of structures selected'
    )
    parser_cur.add_argument(
        '-nj', '--njobs', default=16, type=int,
        help='number of threads for computing features'
    )
    parser_cur.add_argument(
        '-o', '--output', default='selected_structures.xyz',
        help='filename (in xyz format)'
    )
    parser_cur.add_argument(
        '-fe', '--feature', action="store_true",
        help='feature existence'
    )

    # validation
    parser_validation = subparsers.add_parser(
        'valid', help='validate properties with trained model'
    )
    parser_validation.add_argument(
        'INPUTS',
        help='input json file with calculation parameters'
    )
    parser_validation.add_argument(
        '-p', '--potential', nargs=2, required=True,
        help='potential-related parameters'
    )

    # utilities
    parser_utility = subparsers.add_parser(
        'util', help='use ase calculator interface in command line'
    )
    parser_utility.add_argument(
        'INPUTS',
        help='input json file with calculation parameters'
    )

    # semi-auto training
    parser_semi = subparsers.add_parser(
        'semi', help='perform each training step manually'
    )
    parser_semi.add_argument(
        'INPUTS',
        help='a directory with input json files'
    )
    parser_semi.add_argument(
        '-i', '--iter',
        help='iteration'
    )
    parser_semi.add_argument(
        '-s', '--stage',
        help='stage'
    )
    
    # === execute 
    args = parser.parse_args()

    # always check the current workflow before continuing to subcommands 
    # also, the global logger will be initialised 
    # TODO: track the workflow 
    # tracker = track_workflow(args.status)

    # use subcommands
    if args.subcommand == 'train':
        from .trainer.iterative_train import iterative_train
        iterative_train(args.INPUTS)
    elif args.subcommand == 'explore':
        from .sampler.sample_main import run_exploration
        run_exploration(args.potential, args.exploration, args.step)
    elif args.subcommand == 'semi':
        from .trainer.manual_train import manual_train
        manual_train(args.INPUTS, args.iter, args.stage)
    elif args.subcommand == 'ase':
        from .calculator.ase_interface import run_ase_calculator
        run_ase_calculator(args.INPUTS)
    elif args.subcommand == 'valid':
        from .validator.validation import run_validation
        run_validation(args.INPUTS, args.potential)
    elif args.subcommand == 'cur':
        from .selector.structure_selection import select_structures
        select_structures(args.input, args.descriptor, args.number, args.njobs, args.feature, args.output)
    else:
        pass


if __name__ == '__main__':
    main()
