#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from GDPy.data.analyser import DataOperator
from GDPy.utils.command import parse_input_file

def data_main(
    data_inputs,
    potential,
    subcommand,
    mode,
    name, pattern,
    number,# number of selection
    etol,  # energy tolerance
    eshift, # energy shift for structure
    count = 0 # TODO: for reduction
):
    # check data inputs
    input_dict = parse_input_file(data_inputs)
    main_dir = Path(input_dict["database"])

    # ===== parse systems =====
    systems = input_dict["systems"]

    # ====== start working =====
    # create data analyser class and read related structures
    do = DataOperator(
        main_dir, systems, 
        name, pattern, input_dict["sift"]
    )

    do.register_potential(potential)

    if subcommand == "dryrun":
        pass
    elif subcommand == "stat":
        do.show_statistics(input_dict["convergence"]["fmax"])
    elif subcommand == "compress":
        # TODO: parameters
        # number, etol, eshift
        if do.calc is None:
            print("use descriptor-based dataset compression...")
            do.compress_frames(number, eshift)
        else:
            print("use calculator-assisted dataset compression...")
            do.compress_based_on_deviation(number, etol, eshift)
    elif subcommand == "calc":
        # perform operations
        # TODO: fix reduce mode
        if mode == "reduce":
            # compare trained and untrained structures if related info are known
            used_frames, other_frames = do.split_frames(count)
            do.test_frames(used_frames, calc, name+"-used.png")
            do.test_frames(other_frames, calc, name+"-other.png")
        elif mode == "uncertainty":
            do.test_uncertainty_consistent(do.frames, calc, name+"-m-svar.png")


if __name__ == "__main__":
    pass
