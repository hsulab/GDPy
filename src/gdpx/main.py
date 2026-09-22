#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import logging
import pathlib

import numpy as np

from gdpx import config
from gdpx.bootstrap import bootstrap_registries
from gdpx.utils.parser import parse_input_file
from gdpx.utils.strconv import dictionary_to_string


def main():
    # Load all components
    bootstrap_registries(disable_import_info=False)

    # The arguments
    description = "gdpx: Generating Deep Potential with Python\n"

    parser = argparse.ArgumentParser(prog="gdp", description=description)

    parser.add_argument("-rs", "--random_seed", default=None, type=int, help="global random seed")

    parser.add_argument("-d", "--directory", default=pathlib.Path.cwd(), help="working directory")

    # Runtime shared by exploration and validation commands.
    parser.add_argument(
        "-r",
        "--runtime",
        default=None,
        help="schema-v3 runtime configuration (json/yaml)",
    )

    parser.add_argument("-nj", "--n_jobs", default=1, type=int, help="number of processors")

    parser.add_argument("--debug", action="store_true", help="debug mode that gives more information")

    parser.add_argument("--log", default="gdp.out", help="logging output file")

    # subcommands in the entire workflow
    subparsers = parser.add_subparsers(title="available subcommands", dest="subcommand", help="sub-command help")

    # - run session
    parser_session = subparsers.add_parser(
        "session",
        help="run gdpy session",
        description="Run a declarative GDPy workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_session.add_argument("SESSION", help="session configuration file (json/yaml)")
    parser_session.add_argument("--feed", default=None, nargs="+", help="session placeholders")
    parser_session.add_argument(
        "--timewait",
        default=-1,
        type=float,
        help="the waiting time between repeated running",
    )
    parser_session.add_argument("--timemax", default=-1, type=float, help="the maximum time for the entire session")
    parser_session.add_argument("--repeats", default=1000, type=int, help="number of repeat times")

    # - build structures
    parser_build = subparsers.add_parser(
        "build",
        help="build structures",
        description="Build atomic structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_build.add_argument("CONFIG", help="builder configuration file (json/yaml)")
    parser_build.add_argument(
        "-s",
        "--substrates",
        default=None,
        help="file that stores substrates (e.g. *.xyz)",
    )
    parser_build.add_argument("-n", "--number", default=1, type=int, help="number of structures to build")

    # - convert dataset format
    parser_convert = subparsers.add_parser(
        "convert",
        help="convert dataset formats",
        description="Convert dataset formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_convert.add_argument("INPUT", help="path of the input dataset")
    parser_convert.add_argument("-i", "--input_format", required=True, help="the format of the input dataset")
    parser_convert.add_argument("-o", "--output_format", required=True, help="the format of the output dataset")

    # - automatic training
    parser_train = subparsers.add_parser(
        "train",
        help="automatic training utilities",
        description="Train a provider-owned potential model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_train.add_argument("CONFIG", help="training configuration file (json/yaml)")

    # --- compute interface
    parser_compute = subparsers.add_parser(
        "compute",
        help="compute structures with basic methods (MD, MIN, and ...)",
        description="Execute structures using a schema-v3 runtime.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_compute.add_argument(
        "STRUCTURE",
        nargs="*",
        help="structure files, or a lifecycle action: prepare/submit/run/status/resubmit/collect",
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
        help="archive computation folders to cand.tar.zst when retrieving",
    )
    parser_compute.add_argument("--plan", default=None, help="prepared compute plan (defaults to DIRECTORY/_meta/compute-plan.json)")
    parser_compute.add_argument("--job", default=None, help=argparse.SUPPRESS)
    parser_compute.add_argument("--worker", default=0, type=int, help=argparse.SUPPRESS)

    # --- exploration interface
    parser_explore = subparsers.add_parser(
        "explore",
        help="explore structures with advanced methods (GA, MC, and ...)",
        description="Run a structural exploration method.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_explore.add_argument("CONFIG", help="json/yaml file that stores parameters for a task")
    parser_explore.add_argument(
        "--spawn",
        default=None,
        help="The batch indices spawned by a host worker.",
    )
    parser_explore.add_argument("--wait", default=None, type=float, help="wait time after each run")

    # selection
    parser_select = subparsers.add_parser(
        "select",
        help="apply various selection operations",
        description="Select structures for downstream workflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_select.add_argument("CONFIG", help="selection configuration file")
    parser_select.add_argument("-s", "--structures", required=True, nargs="*", help="structure generator")

    # describer
    parser_describe = subparsers.add_parser(
        "describe",
        help="compute descriptors for given structures",
        description="Compute structure descriptors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_describe.add_argument("CONFIG", help="describer configuration")
    parser_describe.add_argument("-s", "--structures", required=True, help="structures")

    # validation
    parser_validate = subparsers.add_parser(
        "validate",
        help="validate properties with trained models",
        description="Validate model predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_validate.add_argument("CONFIG", help="validation configuration file")

    # Excute the parsed subcommand
    args = parser.parse_args()

    # Update global configuration
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

    # Display the package logo
    if args.subcommand != "explore":
        for line in config.LOGO_LINES:
            config._print(line)

    # Set the number of processors
    config.NJOBS = args.n_jobs
    if config.NJOBS != 1 and args.subcommand != "explore":
        config._print(f"Use {config.NJOBS} processors.")

    # Set the global random state
    random_seed = args.random_seed
    if random_seed is not None:
        config.GRNG = np.random.default_rng(random_seed)
    else:
        random_seed = config._random_seed
    state_print = config._debug if args.subcommand == "explore" else config._print
    state_print(f"GLOBAL RANDOM SEED : {random_seed}")

    rng_state = config.GRNG.bit_generator.state
    for l in dictionary_to_string(rng_state).split("\n"):
        state_print(l)

    if args.subcommand == "explore":
        from .cli.explore import run_exploration
        from .exploration.output import exploration_output

        try:
            with exploration_output(args.directory, args.CONFIG, random_seed):
                params = parse_input_file(args.CONFIG)
                runtime = parse_input_file(args.runtime) if args.runtime else None
                run_exploration(params, args.wait, args.directory, runtime, spawn=args.spawn)
        finally:
            config._debug(f"GLOBAL RANDOM SEED : {random_seed}")
            for line in dictionary_to_string(config.GRNG.bit_generator.state).split("\n"):
                config._debug(line)
        return

    runtime = parse_input_file(args.runtime) if args.runtime and args.subcommand != "compute" else None

    # - use subcommands
    if args.subcommand == "session":
        from .cli.session import run_session

        run_session(args.SESSION, args.feed, args.timewait, args.timemax, args.repeats, args.directory)
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

        run_selection(args.CONFIG, structures=args.structures, directory=args.directory)
    elif args.subcommand == "describe":
        from .cli.describe import describe_structures

        desc_config = parse_input_file(args.CONFIG)
        describe_structures(desc_config, args.structures, args.directory)
    elif args.subcommand == "compute":
        from .cli.compute import run_computation

        run_computation(
            args.STRUCTURE,
            args.runtime,
            batch=args.batch,
            spawn=args.spawn,
            archive=args.archive,
            directory=args.directory,
            plan=args.plan,
            job=args.job,
            worker_index=args.worker,
        )
    elif args.subcommand == "validate":
        from .cli.validate import run_validation
        from .execution.factory import create_worker

        params = parse_input_file(args.CONFIG)
        run_validation(params, args.directory, None if runtime is None else create_worker(runtime))
    else:
        ...

    # Report the end random state
    config._print(f"GLOBAL RANDOM SEED : {random_seed}")
    rng_state = config.GRNG.bit_generator.state
    for l in dictionary_to_string(rng_state).split("\n"):
        config._print(l)

    return


if __name__ == "__main__":
    main()
