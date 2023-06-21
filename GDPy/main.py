#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse
from pathlib import Path

import numpy as np

# global settings
from GDPy import config
from GDPy.core.register import registers, import_all_modules_for_register


def main():
    # - register
    import_all_modules_for_register()

    description = "GDPy: Generating Deep Potential with Python\n"

    # - arguments 
    parser = argparse.ArgumentParser(
        prog="gdp", 
        description=description
    )

    parser.add_argument(
        "-d", "--directory", default=Path.cwd(),
        help="working directory"
    )
    
    # the workflow tracker
    parser.add_argument(
        "-p", "--potential", default=None,
        help = "target potential related configuration (json/yaml)"
    )

    parser.add_argument(
        "-nj", "--n_jobs", default = 1, type=int,
        help = "number of processors"
    )
    
    # subcommands in the entire workflow 
    subparsers = parser.add_subparsers(
        title="available subcommands", 
        dest="subcommand", 
        help="sub-command help"
    )

    # - run session
    parser_session = subparsers.add_parser(
        "session", help="run gdpy session", 
        description=str(registers.variable)+"\n"+str(registers.operation), 
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser_session.add_argument(
        "SESSION", help="session configuration file (json/yaml)"
    )
    parser_session.add_argument(
        "-f", "--feed", default=None, nargs="+", 
        help="session placeholders"
    )
    
    # - build structures
    parser_build = subparsers.add_parser(
        "build", help="build structures",
        description=str(registers.builder), 
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser_build.add_argument(
        "CONFIG", help="builder configuration file (json/yaml)"
    )
    parser_build.add_argument(
        "-n", "--number", default=1, type=int,
        help="number of structures to build"
    )
    
    # - automatic training
    parser_train = subparsers.add_parser(
        "train", help="automatic training utilities",
        description=str(registers.trainer), 
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser_train.add_argument(
        "CONFIG", help="training configuration file (json/yaml)"
    )

    # --- worker interface
    parser_worker = subparsers.add_parser(
        "worker", help="run a worker"
    )
    parser_worker.add_argument(
        "STRUCTURE",
        help="a structure file that stores one or more structures"
    )
    parser_worker.add_argument(
        "-b", "--batch", default=None, type=int,
        help="run selected batch number (useful when queue run)"
    )
    parser_worker.add_argument(
        "-o", "--output", default="last", choices=["last","traj"],
        help="retrieve last frame or entire trajectory"
    )

    # --- routine interface
    parser_routine = subparsers.add_parser(
        "routine", help="run a routine (e.g. GA and MC)"
    )
    parser_routine.add_argument(
        "CONFIG",
        help="json/yaml file that stores parameters for a task"
    )
    parser_routine.add_argument(
        "--wait", default=None, type=float,
        help="wait time after each run"
    )

    # selection
    parser_select = subparsers.add_parser(
        "select",
        help="apply various selection operations",
        description=str(registers.selector), 
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser_select.add_argument(
        "CONFIG", help="selection configuration file"
    )
    parser_select.add_argument(
        "-s", "--structure", required=True, 
        help="structure generator"
    )

    # --- validation
    parser_validation = subparsers.add_parser(
        "valid", help="validate properties with trained models"
    )
    parser_validation.add_argument(
        "INPUTS",
        help="input json/yaml file with calculation parameters"
    )
    
    # === execute 
    args = parser.parse_args()
    
    # - update njobs
    config.NJOBS = args.n_jobs
    if config.NJOBS != 1:
        print(f"Run parallel jobs {config.NJOBS}")

    # - potential
    from GDPy.utils.command import parse_input_file
    from GDPy.worker.interface import ComputerVariable
    potter = None
    if args.potential:
        params = parse_input_file(input_fpath=args.potential)
        potter = ComputerVariable(
            params["potential"], params.get("driver", {}), params.get("scheduler", {}),
            batchsize=params.get("batchsize", 1), use_single=params.get("use_single", False), 
        ).value[0]

    # - use subcommands
    if args.subcommand == "session":
        from GDPy.core.session import run_session
        run_session(args.SESSION, args.feed, args.directory)
    elif args.subcommand == "train":
        from GDPy.trainer import run_newtrainer
        run_newtrainer(args.CONFIG, args.directory)
    elif args.subcommand == "build":
        build_config = parse_input_file(args.CONFIG)
        from .builder.interface import build_structures
        build_structures(build_config, args.number, args.directory)
    elif args.subcommand == "select":
        from GDPy.selector.interface import run_selection
        run_selection(args.CONFIG, args.structure, args.directory, potter)
    elif args.subcommand == "worker":
        from GDPy.worker.interface import run_worker
        run_worker(args.STRUCTURE, args.directory, potter, args.output, args.batch)
    elif args.subcommand == "routine":
        from .routine.interface import run_routine
        params = parse_input_file(args.CONFIG)
        run_routine(params, args.wait, args.directory)
    elif args.subcommand == "valid":
        from GDPy.validator import run_validation
        run_validation(args.directory, args.INPUTS, potter)
    else:
        ...


if __name__ == "__main__":
    main()
