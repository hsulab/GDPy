#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from ase.io import read, write

from GDPy.utils.command import parse_input_file
from GDPy.selector.abstract import DeviationSelector, Selector

def selection_main(protocols, stru_fpath, input_fpath, potential, n_jobs=1):
    """"""
    input_dict = parse_input_file(input_fpath)
    selection_params = input_dict.get("selection", None)
    if selection_params is None:
        raise RuntimeError("no selection parameters was found...")

    print("Use selection protocols: ", protocols)

    print("read data: ", stru_fpath)
    frames = []
    for p in stru_fpath:
        cur_frames = read(p, ":")
        for i, a in enumerate(cur_frames):
            a.info["source"] = p
            a.info["index"] = i
        print(f"  nframes {len(cur_frames)} in {p}")
        frames.extend(cur_frames)
    print("total nframes: ", len(frames))

    natoms_array = [len(a) for a in frames]
    
    # TODO: apply a number of selections
    for protocol in protocols:
        all_params = selection_params[protocol]
        if protocol == "GeoSel":
            params = parse_input_file(all_params["params"])
            selection = Selector(
                params["soap"], params["selection"],
                Path.cwd(), njobs=n_jobs
            )
            features = selection.calc_desc(frames)
            selected = selection.select_structures(features, all_params["number"])
            write(protocol+"_selected.xyz", [frames[s] for s in selected])
        elif protocol == "DeviSel":
            selection = DeviationSelector(
                selection_params["DeviSel"],
                potential
            )

            ( 
                energies, maxforces, 
                energy_deviations, force_deviations
            ) = selection.calculate(frames)

            selected = selection.select(energy_deviations, force_deviations)

            selected_frames = [frames[i] for i in selected]
            write("selected.xyz", selected_frames)

            # TODO: write rersults, maybe move to selection objects
            title = ("{:<12s}  "*6+"\n").format(
                "#IDX", "Energy", "AEDevi", "TEDEVI", "Fmax", "FDevi"
            )
            content = ""
            for i in selected:
                content += ("{:<12d}  "+"{:<12.4f}  "*5+"\n").format(
                    i, energies[i], 
                    energy_deviations[i], energy_deviations[i]*natoms_array[i],
                    maxforces[i], force_deviations[i]
                )
            with open("devi.dat", "w") as fopen:
                fopen.write(title+content)
        else:
            raise RuntimeError(f"no selection protocol {protocol}...")

    return


def previous_main():
    # arguments 
    parser = argparse.ArgumentParser(
        prog='dprss', 
        description='DeepRSS: Random Structure Sampling with DeepMD Potential'
    )
    
    # the workflow tracker
    parser.add_argument(
        '-s', '--status', 
        help='pickle file with info on the current workflow'
    )
    
    # subcommands in the entire workflow 
    subparsers = parser.add_subparsers(
        title='available subcommands', 
        dest='subcommand', 
        help='sub-command help'
    )
    
    # rss
    parser_rss = subparsers.add_parser(
        'rss', help='randomly generate structures with given seed cell'
    )
    parser_rss.add_argument(
        '-s', '--seed', required=True, 
        help='seed file'
    )
    parser_rss.add_argument(
        '-n', '--number', type=int, 
        help='number of structures generated'
    )
    parser_rss.add_argument(
        '-o', '--output', default='random_structures.xyz', 
        help='filename (in xyz format)'
    )
    
    # cur 
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

    # val 
    parser_val = subparsers.add_parser(
        'val', 
        help='perform evaluation using trained potential (single point calculation)'
    )
    parser_val.add_argument(
        '-i', '--input', required=True, 
        help='filename (in xyz format)'
    )
    parser_val.add_argument(
        '-p', '--param', default='params.json', 
        help='calculation-related parameters in json file'
    )
    
    # dyn
    parser_dyn = subparsers.add_parser(
        'dyn', 
        help='perform dynamics (relaxation/MD) using current potential'
    )
    parser_dyn.add_argument(
        '-i', '--input', required=True, 
        help='filename (in xyz format)'
    )
    parser_dyn.add_argument(
        '-w', '--work', required=True, 
        help='working directory'
    )
    
    # blz
    parser_blz = subparsers.add_parser(
        'blz', 
        help='perform Boltzmann histogram selection on dp trajectories with minimum'
    )
    parser_blz.add_argument(
        '-w', '--work', required=True, 
        help='working directory'
    )
    parser_blz.add_argument(
        '-n', '--number', type=int, 
        help='number of minima needed'
    )
    parser_blz.add_argument(
        '--read', action="store_true", 
        help='only read trajectories'
    )
    read_group = parser_blz.add_argument_group(title='Read Trajectories')
    read_group.add_argument(
        '-i', '--indices', default=':', 
        help='read selected trajectories'
    )
    
    # others
    
    # === execute 
    args = parser.parse_args()

    # always check the current workflow before continuing to subcommands 
    # also, the global logger will be initialised 
    # TODO: track the workflow 
    tracker = track_workflow(args.status)

    # use subcommands
    if args.subcommand == 'rss':
        write_rss_frames(tracker, args.seed, args.number, args.output)
    elif args.subcommand == 'cur':
        select_structures(args.input, args.descriptor, args.number, args.njobs, args.feature, args.output) 
    elif args.subcommand == 'val':
        from dprss.validation import validate_function
        validate_function(args.input, args.param)
    elif args.subcommand == 'dyn':
        # only load tensorflow when dp is called 
        from dprss.dynamics_routine import run_dpdyn
        run_dpdyn(args.input, args.work)
    elif args.subcommand == 'blz':
        from dprss.trajectory_selection import select_minima, read_massive_trajs
        if args.read:
            read_massive_trajs(args.work, 'ucf-', args.indices)
        else:
            #select_trajectory(args.work, args.number)
            select_minima(num_minima=args.number)
    else:
        dprss_logger.info('only check the current work status')


if __name__ == '__main__':
    main()
    pass
