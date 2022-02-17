#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from GDPy.data.analyser import DataOperator

def data_main(
    potential,
    subcommand,
    mode,
    main_dir, name, pattern,
    number,# number of selection
    etol,  # energy tolerance
    eshift, # energy shift for structure
    count = 0 # TODO: for reduction
):
    # ====== start working =====
    # create data analyser class and read related structures
    do = DataOperator(name)
    if name != "ALL":
        do.frames = do.read_frames(
            Path(main_dir) / name,
            pattern = pattern
        )
    else:
        do.frames = do.read_all_frames(
            Path(main_dir), 
            pattern = pattern
        )
    do.remove_large_force_structures()

    # load potential
    from GDPy.potential.manager import create_manager
    if potential is not None:
        atypes = None
        pm = create_manager(potential)
        calc = pm.generate_calculator(atypes)
        print("MODELS: ", pm.models)
    else:
        calc = None
    
    do.register_calculator(calc)

    if subcommand == "stat":
        do.check_xyz()
    elif subcommand == "compress":
        # TODO: parameters
        # number, etol, eshift
        if calc is None:
            print("use descriptor-based dataset compression...")
            do.compress_frames(number, None, eshift)
        else:
            print("use calculator-assisted dataset compression...")
            do.compress_based_on_deviation(
                calc, number, etol, eshift
            )
    elif subcommand == "calc":
        # perform operations
        # TODO: fix reduce mode
        if mode == "reduce":
            # compare trained and untrained structures if related info are known
            used_frames, other_frames = do.split_frames(count)
            do.test_frames(used_frames, calc, name+"-used.png")
            do.test_frames(other_frames, calc, name+"-other.png")
        elif mode == "simple":
            # compare trained and untrained structures if related info are known
            #do.test_frames(do.frames, calc, exists_data=True, saved_figure=f"{args.name}-all.png")
            fig_name = Path(main_dir).resolve().name + "-" + f"{name}-all.png"
            print(fig_name)
            do.test_frames(fig_name)
            #do.check_xyz()
        elif mode == "uncertainty":
            do.test_uncertainty_consistent(do.frames, calc, name+"-m-svar.png")


if __name__ == "__main__":
    pass
