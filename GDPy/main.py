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
        prog="gdp", 
        description="GDPy: Generating Deep Potential with Python"
    )
    
    # the workflow tracker
    parser.add_argument(
        '-s', '--status', 
        help='pickle file with info on the current workflow'
    )

    parser.add_argument(
        "-pot", "--potential", default = None,
        help = "potential related configuration"
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

    # automatic training
    parser_model = subparsers.add_parser(
        "model", help="model operations"
    )
    parser_model.add_argument(
        "INPUTS",
        help="a directory with input json files"
    )
    parser_model.add_argument(
        "-m", "--mode",
        help="[create/freeze] models"
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
        "-s", "--step", required=True,
        choices = ["create", "collect", "select", "calc", "harvest"],
        help="exploration steps (create/collect/select)"
    )
    parser_explore.add_argument(
        "-op", "--opt_params", nargs="*",
        help="global parameters for exploration"
    )

    # ----- data analysis -----
    parser_data = subparsers.add_parser(
        "data", help="data analysis subcommand"
    )

    parser_data.add_argument(
        "MODE", choices = ["stat", "calc", "compress"],
        help = "choose data analysis mode"
    )
    
    # general options for reading structures
    parser_data.add_argument(
        "-d", "--main_dir", # TODO: main dir for dataset
        default = "/users/40247882/scratch2/PtOx-dataset",
        help = "main directory that contains systemwise xyz files"
    )
    parser_data.add_argument(
        "-n", "--name", default = "ALL",
        help = "system name"
    )
    parser_data.add_argument(
        "-p", "--pattern", default = "*.xyz",
        help = "xyz search pattern"
    )

    parser_data.add_argument(
        "-m", "--mode", default=None,
        help = "data analysis mode"
    )

    parser_data.add_argument(
        "-num", "--number", 
        default = -1, type=int,
        help = "number of selection"
    )
    parser_data.add_argument(
        "-etol", "--energy_tolerance", 
        default = 0.020, type = float,
        help = "energy tolerance per atom"
    )
    parser_data.add_argument(
        "-es", "--energy_shift", 
        default = 0.0, type = float,
        help = "add energy correction for each structure"
    )


    # ase calculator interface
    parser_ase = subparsers.add_parser(
        'ase', help='use ase calculator interface in command line'
    )
    parser_ase.add_argument(
        'INPUTS',
        help='input json file with calculation parameters'
    )
    parser_ase.add_argument(
        '-p', '--potential', required=True,
        help='potential-related input json'
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
        '-p', '--potential', required=True,
        help='potential-related input json'
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
    if args.subcommand == "train":
        from .trainer.iterative_train import iterative_train
        iterative_train(args.INPUTS)
    elif args.subcommand == "model":
        from .potential.manager import create_manager
        pm = create_manager(args.INPUTS)
        if args.mode == "freeze":
            pm.freeze_ensemble()
        elif args.mode == "create":
            pm.create_ensemble()
    elif args.subcommand == 'explore':
        from .expedition.sample_main import run_exploration
        run_exploration(args.potential, args.exploration, args.step, args.opt_params)
    elif args.subcommand == "data":
        from GDPy.data.main import data_main
        data_main(
            args.potential, args.MODE, args.mode,
            args.main_dir, args.name, args.pattern,
            args.number, args.energy_tolerance, args.energy_shift
        )
    elif args.subcommand == 'semi':
        from .trainer.manual_train import manual_train
        manual_train(args.INPUTS, args.iter, args.stage)
    elif args.subcommand == 'ase':
        from .calculator.ase_interface import run_ase_calculator
        run_ase_calculator(args.INPUTS, args.potential)
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
