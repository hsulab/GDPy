#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import pathlib
from pathlib import Path

import numpy as np

from ase.io import read, write

from GDPy.data.analyser import DataOperator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general options for reading structures
    parser.add_argument(
        "-d", "--main_dir", 
        default = "/users/40247882/scratch2/PtOx-dataset",
        help = "main directory that contains systemwise xyz files"
    )
    parser.add_argument(
        "-n", "--name", default = "ALL",
        help = "system name"
    )
    parser.add_argument(
        "-p", "--pattern", default = "*.xyz",
        help = "xyz search pattern"
    )

    parser.add_argument(
        "-pot", "--potential", default = None,
        help = "potential related json"
    )

    # subcommands in the entire workflow 
    subparsers = parser.add_subparsers(
        title="available subcommands", 
        dest="subcommand", 
        help="sub-command help"
    )

    # --- statistics the dataset 
    parser_stat = subparsers.add_parser(
        "stat",
        help="give out the dataset statistics"
    )

    # --- compress the dataset 
    parser_compress = subparsers.add_parser(
        "compress",
        help="compress the dataset using energy histogram and cur decomposition"
    )
    parser_compress.add_argument(
        "-num", "--number", 
        default = -1, type=int,
        help = "number of selection"
    )
    parser_compress.add_argument(
        "-et", "--energy_tolerance", 
        default = 0.020, type = float,
        help = "energy tolerance per atom"
    )
    parser_compress.add_argument(
        "-es", "--energy_shift", 
        default = 0.0, type = float,
        help = "add energy correction for each structure"
    )
    parser_compress.add_argument(
        "-sp", "--selection_params", 
        default = -1, type=int,
        help = "paramter json "
    )
    parser_compress.add_argument(
        "-cons", "--constraint", 
        default = -1, type=int,
        help = "constraint indices"
    )

    # --- reduce the dataset using iterative training and cur decomposition
    parser_reduce = subparsers.add_parser(
        "reduce", 
        help="reduce dataset by comparing errors in trained and untrained structures"
    )    
    parser_reduce.add_argument(
        "-num", "--number", 
        default = -1, type=int,
        help = "number of selection"
    )
    parser_reduce.add_argument(
        "-c", "--count", required = True,
        type = int,
        help = "number of selection"
    )

    # --- calculate with machine learning potential
    parser_calc = subparsers.add_parser(
        "calc", 
        help="calculate structures with machine learning potential"
    )    
    parser_calc.add_argument(
        "-m", "--mode", 
        help = "calculation mode"
    )

    parser_plot = subparsers.add_parser(
        "plot", 
        help="plot result data"
    )

    args = parser.parse_args()

    # ====== start working =====
    # create data analyser class and read related structures
    do = DataOperator(args.name)
    if args.name != "ALL":
        do.frames = do.read_frames(
            pathlib.Path(args.main_dir) / args.name,
            pattern = args.pattern
        )
    else:
        do.frames = do.read_all_frames(
            pathlib.Path(args.main_dir), 
            pattern = args.pattern
        )
    do.remove_large_force_structures()

    # load potential
    from GDPy.potential.manager import create_manager
    if args.potential is not None:
        atypes = None
        pm = create_manager(args.potential)
        print(pm.models)
        calc = pm.generate_calculator(atypes)
    else:
        calc = None
    
    do.register_calculator(calc)

    if args.subcommand == "stat":
        do.check_xyz()
    elif args.subcommand == "compress":
        if calc is None:
            print("use descriptor-based dataset compression...")
            do.compress_frames(args.number, None, args.energy_shift)
        else:
            print("use calculator-assisted dataset compression...")
            do.compress_based_on_deviation(
                calc, args.number, args.energy_tolerance, args.energy_shift
            )
    elif args.subcommand == "calc":
        # perform operations
        mode = args.mode
        if mode == "reduce":
            # compare trained and untrained structures if related info are known
            used_frames, other_frames = do.split_frames(args.count)
            do.test_frames(used_frames, calc, args.name+"-used.png")
            do.test_frames(other_frames, calc, args.name+"-other.png")
        elif mode == "simple":
            # compare trained and untrained structures if related info are known
            #do.test_frames(do.frames, calc, exists_data=True, saved_figure=f"{args.name}-all.png")
            fig_name = Path(args.main_dir).resolve().name + "-" + f"{args.name}-all.png"
            print(fig_name)
            do.test_frames(fig_name)
            #do.check_xyz()
        elif mode == "uncertainty":
            do.test_uncertainty_consistent(do.frames, calc, args.name+"-m-svar.png")