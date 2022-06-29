#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from GDPy.data.analyser import DataOperator
from GDPy.utils.command import parse_input_file

def data_main(
    data_inputs,
    pot_manager,
    subcommand,
    mode,
    system_file,
    name, pattern,
    number,# number of selection
    etol,  # energy tolerance
    eshift, # energy shift for structure
    count = 0 # TODO: for reduction
):
    # ===== parse systems =====
    # check data inputs
    input_dict = parse_input_file(data_inputs)

    if system_file is None:
        main_dir = Path(input_dict["database"])
        systems = input_dict["systems"]
        global_type_list = input_dict["type_list"]
    else:
        sys_dict = parse_input_file(system_file)
        main_dir = Path(sys_dict.pop("database", None))
        global_type_list = sys_dict.pop("type_list", None)
        systems = sys_dict.copy()

    # ====== start working =====
    # parse potential
    calc = None
    if pot_manager:
        calc = pot_manager.calc

    # create data analyser class and read related structures
    do = DataOperator(
        main_dir, systems, global_type_list,
        name, pattern, 
        calc,
        input_dict["convergence"],
        input_dict["sift"],
        input_dict.get("compress", None),
        input_dict.get("selection", None)
    )

    if subcommand == "dryrun":
        pass
    elif subcommand == "stat":
        do.show_statistics()
    elif subcommand == "compress":
        do.compress_systems(number, etol, eshift)
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
