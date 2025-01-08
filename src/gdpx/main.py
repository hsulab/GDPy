#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import logging
import pathlib

import numpy as np

from . import config
from .core.register import import_all_modules_for_register, registers
from .utils.command import dict2str, parse_input_file


def main():
    # - register
    import_all_modules_for_register()

    description = "gdpx: Generating Deep Potential with Python\n"

    # - arguments
    parser = argparse.ArgumentParser(prog="gdp", description=description)

    parser.add_argument(
        "-rs", "--random_seed", default=None, type=int, help="global random seed"
    )

    parser.add_argument(
        "-d", "--directory", default=pathlib.Path.cwd(), help="working directory"
    )

    # the workflow tracker
    parser.add_argument(
        "-p",
        "--potential",
        default=None,
        help="target potential related configuration (json/yaml)",
    )

    parser.add_argument(
        "-nj", "--n_jobs", default=1, type=int, help="number of processors"
    )

    parser.add_argument(
        "--debug", action="store_true", help="debug mode that gives more information"
    )

    parser.add_argument("--log", default="gdp.out", help="logging output file")

    # subcommands in the entire workflow
    subparsers = parser.add_subparsers(
        title="available subcommands", dest="subcommand", help="sub-command help"
    )

    # - run session
    parser_session = subparsers.add_parser(
        "session",
        help="run gdpy session",
        description=str(registers.variable) + "\n" + str(registers.operation),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_session.add_argument(
        "SESSION", help="session configuration file (json/yaml)"
    )
    parser_session.add_argument(
        "--feed", default=None, nargs="+", help="session placeholders"
    )
    parser_session.add_argument(
        "--timewait",
        default=-1,
        type=float,
        help="waiting time between repeated running",
    )

    # - build structures
    parser_build = subparsers.add_parser(
        "build",
        help="build structures",
        description=str(registers.builder),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_build.add_argument("CONFIG", help="builder configuration file (json/yaml)")
    parser_build.add_argument(
        "-s",
        "--substrates",
        default=None,
        help="file that stores substrates (e.g. *.xyz)",
    )
    parser_build.add_argument(
        "-n", "--number", default=1, type=int, help="number of structures to build"
    )

    # - convert dataset format
    parser_convert = subparsers.add_parser(
        "convert",
        help="convert dataset formats",
        description=str(registers.dataloader),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_convert.add_argument("INPUT", help="path of the input dataset")
    parser_convert.add_argument(
        "-i", "--input_format", required=True, help="the format of the input dataset"
    )
    parser_convert.add_argument(
        "-o", "--output_format", required=True, help="the format of the output dataset"
    )

    # - automatic training
    parser_train = subparsers.add_parser(
        "train",
        help="automatic training utilities",
        description=str(registers.trainer) + "\n" + str(registers.dataloader),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_train.add_argument("CONFIG", help="training configuration file (json/yaml)")

    # --- compute interface
    parser_compute = subparsers.add_parser(
        "compute",
        help="compute structures with basic methods (MD, MIN, and ...)",
        description=str(registers.manager),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_compute.add_argument(
        "STRUCTURE",
        nargs="*",
        help="a structure file that stores one or more structures",
    )
    parser_compute.add_argument(
        "-b",
        "--batch",
        default=None,
        type=int,
        help="run selected batch number (useful when queue run)",
    )
    parser_compute.add_argument(
        "--spawn",
        action="store_true",
        help="If the computation is spawned, it will not save results until all jobs are finished.",
    )
    parser_compute.add_argument(
        "--archive",
        action="store_true",
        help="whether archive computation folders when retrieve",
    )

    # --- expedition interface
    parser_explore = subparsers.add_parser(
        "explore",
        help="explore structures with advanced methods (GA, MC, and ...)",
        description=str(registers.expedition),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_explore.add_argument(
        "CONFIG", help="json/yaml file that stores parameters for a task"
    )
    parser_explore.add_argument(
        "--spawn",
        default=None,
        help="The batch indices spawned by a host worker.",
    )
    parser_explore.add_argument(
        "--wait", default=None, type=float, help="wait time after each run"
    )

    # selection
    parser_select = subparsers.add_parser(
        "select",
        help="apply various selection operations",
        description=str(registers.selector),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_select.add_argument("CONFIG", help="selection configuration file")
    parser_select.add_argument(
        "-s", "--structure", required=True, help="structure generator"
    )

    # describer
    parser_describe = subparsers.add_parser(
        "describe",
        help="compute descriptors for given structures",
        description=str(registers.describer),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_describe.add_argument("CONFIG", help="describer configuration")
    parser_describe.add_argument("-s", "--structures", required=True, help="structures")

    # --- validation
    parser_validate = subparsers.add_parser(
        "validate",
        help="validate properties with trained models",
        description=str(registers.validator),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_validate.add_argument("CONFIG", help="validation configuration file")

    # === execute
    args = parser.parse_args()

    # - update global configuration
    if args.debug:
        config.logger.setLevel(logging.DEBUG)

    curr_wdir = pathlib.Path(args.directory)
    if not curr_wdir.exists():
        curr_wdir.mkdir(parents=True)

    if args.log:
        logfpath = curr_wdir / args.log
        if logfpath.exists():
            fh = logging.FileHandler(logfpath, mode="a")
        else:
            fh = logging.FileHandler(logfpath, mode="w")
        fh.setFormatter(config.formatter)
        config.logger.addHandler(fh)

    # -- set LOGO
    for line in config.LOGO_LINES:
        config._print(line)

    # -- set njobs
    config.NJOBS = args.n_jobs
    if config.NJOBS != 1:
        config._print(f"Use {config.NJOBS} processors.")

    # -- set rng
    # TODO: load random state from a file???
    random_seed = args.random_seed
    if random_seed is None:
        # NOTE: np.random should only be called here once...
        random_seed = np.random.randint(0, 1e8)
    else:
        ...

    config.GRNG = np.random.Generator(np.random.PCG64(random_seed))

    config._print(f"GLOBAL RANDOM SEED : {random_seed}")
    rng_state = config.GRNG.bit_generator.state
    for l in dict2str(rng_state).split("\n"):
        config._print(l)

    # - potential
    if args.potential:
        # a worker or a List of worker
        from .cli.compute import convert_input_to_computer

        computer = convert_input_to_computer(args.potential)
        workers = computer.value
    else:
        computer = None
        workers = [None]

    # - use subcommands
    if args.subcommand == "session":
        from gdpx.core.session import run_session

        run_session(args.SESSION, args.feed, args.timewait, args.directory)
    elif args.subcommand == "convert":
        from .cli.convert import convert_dataset

        convert_dataset(args.INPUT, args.input_format, args.output_format, curr_wdir)
    elif args.subcommand == "train":
        from .cli.train import run_trainer

        run_trainer(args.CONFIG, args.directory)
    elif args.subcommand == "build":
        build_config = parse_input_file(args.CONFIG)
        from .cli.build import build_structures

        build_structures(build_config, args.substrates, args.number, args.directory)
    elif args.subcommand == "select":
        from .cli.select import run_selection

        run_selection(args.CONFIG, args.structure, args.directory)
    elif args.subcommand == "describe":
        from .cli.describe import describe_structures

        desc_config = parse_input_file(args.CONFIG)
        describe_structures(desc_config, args.structures, args.directory)
    elif args.subcommand == "compute":
        from .cli.compute import run_computation

        run_computation(
            args.STRUCTURE,
            computer,
            batch=args.batch,
            spawn=args.spawn,
            archive=args.archive,
            directory=args.directory,
        )
    elif args.subcommand == "explore":
        from .cli.explore import run_expedition

        params = parse_input_file(args.CONFIG)
        run_expedition(params, args.wait, args.directory, workers[0], spawn=args.spawn)
    elif args.subcommand == "validate":
        from .cli.validate import run_validation

        params = parse_input_file(args.CONFIG)
        run_validation(params, args.directory, workers[0])
    else:
        ...

    # - report the end random state
    config._print(f"GLOBAL RANDOM SEED : {random_seed}")
    rng_state = config.GRNG.bit_generator.state
    for l in dict2str(rng_state).split("\n"):
        config._print(l)

    return


if __name__ == "__main__":
    main()
