#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse
from pathlib import Path

import numpy as np

# global settings
from GDPy import config


def main():
    # arguments 
    parser = argparse.ArgumentParser(
        prog="gdp", 
        description="GDPy: Generating Deep Potential with Python"
    )
    
    # the workflow tracker
    parser.add_argument(
        "-p", "--potential", default=None,
        help = "target potential related configuration (json/yaml)"
    )

    parser.add_argument(
        "-r", "--reference", default=None,
        help = "reference potential related configuration (json/yaml)"
    )

    parser.add_argument(
        "-nj", "--n_jobs", default = 1, type=int,
        help = "number of processors"
    )
    
    # subcommands in the entire workflow 
    subparsers = parser.add_subparsers(
        title='available subcommands', 
        dest='subcommand', 
        help='sub-command help'
    )

    # single calculation creator (VASP for now)
    parser_vasp = subparsers.add_parser(
        "vasp", help="utils to create and analyse vasp calculation"
    )
    parser_vasp.add_argument(
        "CHOICE", choices=["create", "freq", "data", "work"],
        help = "choices performed with vasp"
    )
    parser_vasp.add_argument(
        "STRUCTURE",
        help="structure file in any format (better xsd)"
    )
    parser_vasp.add_argument(
        "-in", "--input", default=None,
        help="template incar file or a json/yaml configuration file"
    )
    parser_vasp.add_argument(
        "-fi", "--indices", default=None,
        help="frame indices to read for each vasp directory"
    )
    parser_vasp.add_argument(
        # "-c", "--copt", action='store_true',
        "-ai", "--aindices", type=int, nargs="*",
        help="atom indices for constrain or freq, python convention"
    )
    parser_vasp.add_argument(
        "-ns", "--nosort", action="store_false",
        help="sort atoms by elemental numbers and z-positions"
    )
    parser_vasp.add_argument(
        "--sub", action="store_true",
        help="submit the job after creating input files"
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
        "explore", help="exploration configuration file (json/yaml)"
    )
    parser_explore.add_argument(
        "EXPEDITION", 
        help="expedition configuration file (json/yaml)"
    )
    parser_explore.add_argument(
        "--run", default=None,
        help="running option"
    )

    # ----- data analysis -----
    parser_data = subparsers.add_parser(
        "data", help="data analysis subcommand"
    )

    parser_data.add_argument(
        "DATA", help = "general data setting file"
    )

    parser_data.add_argument(
        "-c", "--choice", default="dryrun",
        choices = ["dryrun", "stat", "calc", "compress"],
        help = "choose data analysis mode"
    )
    
    # general options for reading structures
    parser_data.add_argument(
        "-s", "--system", default = None,
        help = "system info file"
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

    # --- driver interface
    parser_driver = subparsers.add_parser(
        "driver", help="run a driver"
    )
    parser_driver.add_argument(
        "params",
        help="json/yaml file that stores parameters for a driver"
    )
    parser_driver.add_argument(
        "-s", "--structure",
        help="a structure file that stores one or more structures"
    )
    parser_driver.add_argument(
        "-d", "--directory", default=Path.cwd(),
        help="working directory"
    )

    # --- worker interface
    parser_worker = subparsers.add_parser(
        "worker", help="run a worker"
    )
    parser_worker.add_argument(
        "params",
        help="json/yaml file that stores parameters for a worker"
    )
    parser_worker.add_argument(
        "-s", "--structure",
        help="a structure file that stores one or more structures"
    )

    # --- task interface
    parser_task = subparsers.add_parser(
        "task", help="run a task (e.g. GA and MC)"
    )
    parser_task.add_argument(
        "params",
        help="json/yaml file that stores parameters for a task"
    )
    parser_task.add_argument(
        "--run", default=1, type=int,
        help="running options"
    )

    # selection
    parser_select = subparsers.add_parser(
        "select",
        help="apply various selection operations"
    )
    parser_select.add_argument(
        "CONFIG", help="selection configuration file"
    )
    parser_select.add_argument(
        "-f", "--structure_file", required=True, 
        nargs="*", action="extend",
        help="structure filepath (in xyz format)"
    )
    parser_select.add_argument(
        "-m", "--mode", required=True, nargs="*",
        help="selection mode"
    )
    parser_select.add_argument(
        '-n', '--number', default=100, type=int,
        help='number of structures selected'
    )

    # graph utils
    parser_graph = subparsers.add_parser(
        "graph",
        help="graph utils"
    )
    parser_graph.add_argument(
        "CONFIG", help="graph configuration file"
    )
    parser_graph.add_argument(
        "-f", "--structure_file", required=True,
        help="structure filepath (in xyz format)"
    )
    parser_graph.add_argument(
        "-i", "--indices", default=":",
        help="structure indices"
    )
    parser_graph.add_argument(
        "-m", "--mode", required=True,
        choices = ["diff", "add"],
        help="structure filepath (in xyz format)"
    )

    # validation
    parser_validation = subparsers.add_parser(
        'valid', help='validate properties with trained model'
    )
    parser_validation.add_argument(
        'INPUTS',
        help='input json file with calculation parameters'
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
    
    # update njobs
    config.NJOBS = args.n_jobs
    if config.NJOBS != 1:
        print(f"Run parallel jobs {config.NJOBS}")

    # always check the current workflow before continuing to subcommands 
    # also, the global logger will be initialised 
    # TODO: track the workflow 
    # tracker = track_workflow(args.status)

    # - potential
    from GDPy.potential.register import create_potter
    potter = None
    if args.potential:
        pot_config = args.potential # configuration file of potential
        potter = create_potter(pot_config) # register calculator, and scheduler if exists
    
    referee = None
    if args.reference:
        ref_config = args.reference # configuration file of potential
        referee = create_potter(ref_config) # register calculator, and scheduler if exists

    # - use subcommands
    if args.subcommand == "vasp":
        from GDPy.utils.vasp.main import vasp_main
        vasp_main(args.STRUCTURE, args.CHOICE, args.indices, args.input, args.aindices, args.nosort, args.sub)
    elif args.subcommand == "train":
        from .trainer.iterative_train import iterative_train
        iterative_train(args.INPUTS)
    elif args.subcommand == "model":
        from .potential.manager import create_manager
        pm = create_manager(args.INPUTS)
        if args.mode == "freeze":
            pm.freeze_ensemble()
        elif args.mode == "create":
            pm.create_ensemble()
    elif args.subcommand == "explore":
        from GDPy.expedition import run_expedition
        run_expedition(potter, referee, args.EXPEDITION)
    elif args.subcommand == "data":
        from GDPy.data.main import data_main
        data_main(
            args.DATA,
            potter, args.choice, args.mode,
            args.system,
            args.name, args.pattern,
            args.number, args.energy_tolerance, args.energy_shift
        )
    elif args.subcommand == 'semi':
        from .trainer.manual_train import manual_train
        manual_train(args.INPUTS, args.iter, args.stage)
    elif args.subcommand == "driver":
        from GDPy.computation.driver import run_driver
        run_driver(args.params, args.structure, args.directory, potter)
    elif args.subcommand == "worker":
        from GDPy.computation.worker import run_worker
        run_worker(args.params, args.structure, potter)
    elif args.subcommand == "task":
        from GDPy.task.task import run_task
        run_task(args.params, potter, args.run)
    elif args.subcommand == "valid":
        from .validator.validation import run_validation
        run_validation(args.INPUTS, potter)
    elif args.subcommand == "select":
        from GDPy.selector.main import selection_main
        selection_main(args.mode, args.structure_file, args.CONFIG, pot_config, args.n_jobs)
    elif args.subcommand == "graph":
        from GDPy.graph.graph_main import graph_main
        graph_main(args.n_jobs, args.CONFIG, args.structure_file, args.indices, args.mode)
    else:
        pass


if __name__ == "__main__":
    main()
