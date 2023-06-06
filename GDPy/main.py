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

    # - arguments 
    parser = argparse.ArgumentParser(
        prog="gdp", 
        description="GDPy: Generating Deep Potential with Python"
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
        "-r", "--reference", default=None,
        help = "reference potential related configuration (json/yaml)"
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
        "session", help="run gdpy session"
    )
    parser_session.add_argument(
        "SESSION", help="session configuration file (json/yaml)"
    )
    parser_session.add_argument(
        "-f", "--feed", default=None, nargs="+", 
        help="session placeholders"
    )
    
    # - automatic training
    parser_train = subparsers.add_parser(
        "train", help="automatic training utilities"
    )

    parser_newtrain = subparsers.add_parser(
        "newtrain", help="automatic training utilities"
    )
    parser_newtrain.add_argument(
        "CONFIG", help="training configuration file (json/yaml)"
    )

    # - explore
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
        "data", help="data analysis"
    )
    parser_data.add_argument(
        "DATA", help = "data configuration file (json/yaml)"
    )
    parser_data.add_argument(
        "-r", "--run", default=None,
        help = "configuration for specific operation (json/yaml)"
    )
    parser_data.add_argument(
        "-c", "--choice", default="dryrun",
        choices = ["dryrun", "stat", "calc", "compress"],
        help = "choose data analysis mode"
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
    parser_task.add_argument(
        "--report", action="store_true", # TODO: analysis config file
        help="report options"
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
        "-s", "--structure", required=True, 
        help="structure generator"
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
    
    # update njobs
    config.NJOBS = args.n_jobs
    if config.NJOBS != 1:
        print(f"Run parallel jobs {config.NJOBS}")

    # - potential
    #from GDPy.potential.register import create_potter
    #potter = None
    #if args.potential:
    #    pot_config = args.potential # configuration file of potential
    #    potter = create_potter(pot_config) # register calculator, and scheduler if exists
    
    #referee = None
    #if args.reference:
    #    ref_config = args.reference # configuration file of potential
    #    referee = create_potter(ref_config) # register calculator, and scheduler if exists

    from GDPy.utils.command import parse_input_file
    from GDPy.computation.worker.interface import WorkerVariable
    potter = None
    if args.potential:
        params = parse_input_file(input_fpath=args.potential)
        potter = WorkerVariable(
            params["potential"], params.get("driver", {}), params.get("scheduler", {}),
            params.get("batchsize", 1)
        ).value

    # - use subcommands
    if args.subcommand == "train":
        from GDPy.trainer import run_trainer
        run_trainer(potter, args.directory)
    elif args.subcommand == "newtrain":
        from GDPy.trainer import run_newtrainer
        run_newtrainer(args.CONFIG, args.directory)
    elif args.subcommand == "session":
        from GDPy.core.session import run_session
        run_session(args.SESSION, args.feed, args.directory)
    elif args.subcommand == "select":
        from GDPy.selector.interface import run_selection
        run_selection(args.CONFIG, args.structure, args.directory, potter)
    elif args.subcommand == "explore":
        from GDPy.expedition import run_expedition
        run_expedition(potter, referee, args.EXPEDITION)
    elif args.subcommand == "data":
        from GDPy.data import data_main
        data_main(
            args.DATA,
            potter, referee,
            args.run,
            #
            args.choice, args.mode,
            args.name, args.pattern,
            args.number, args.energy_tolerance, args.energy_shift
        )
    elif args.subcommand == "worker":
        from GDPy.computation.worker.interface import run_worker
        run_worker(args.STRUCTURE, args.directory, potter, args.output, args.batch)
    elif args.subcommand == "task":
        from GDPy.task.task import run_task
        run_task(args.params, potter, referee, args.run, args.report)
    elif args.subcommand == "valid":
        from GDPy.validator import run_validation
        run_validation(args.directory, args.INPUTS, potter)
    elif args.subcommand == "graph":
        from GDPy.graph.graph_main import graph_main
        graph_main(args.n_jobs, args.CONFIG, args.structure_file, args.indices, args.mode)
    else:
        pass


if __name__ == "__main__":
    main()
