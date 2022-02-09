#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pathlib
import argparse

import numpy as np

import matplotlib as mpl
mpl.use('Agg') #silent mode
from matplotlib import pyplot as plt
plt.style.use('presentation')

from ase import Atoms

from GDPy.utils.data.dpsets import set2frames, sets2results

""" read raw data and test
    this part should be moved to data analyser
"""

def calc_results(system_dirs, calc, res_dir='./results'):
    """
    """
    res_dir = pathlib.Path(res_dir)
    for cur_system in system_dirs:
        name = cur_system.name
        print('calc system %s...' %name)
        # read elements
        type_list = np.loadtxt(cur_system / 'type_map.raw', dtype=str)
        atype = np.loadtxt(cur_system / 'type.raw', dtype=int)
        #amass = np.array([mass_list[x] for x in atype])
        chemical_symbols = [type_list[a] for a in atype]

        # run over sets
        set_dirs = list(cur_system.glob('set*'))
        set_dirs.sort()

        # test
        (
            ref_energies, ref_forces, mlp_energies, mlp_forces
        ) = sets2results(set_dirs[-1:], calc, chemical_symbols)
        #print(mlp_energies)
        #print(mlp_forces)

        energies = np.array([ref_energies, mlp_energies])
        forces = np.array([ref_forces, mlp_forces])
        np.save(res_dir / (name+'-energies.npy'), energies)
        np.save(res_dir / (name+'-forces.npy'), forces)

    return


def plot_results(system_dirs, calc_name, res_dir='./results', elementwise=True):
    from GDPy.utils.comparasion import parity_plot_dict
    res_dir = pathlib.Path(res_dir)
    for idx, cur_system in enumerate(system_dirs):
        # basic info
        name = cur_system.name
        # read elements
        type_list = np.loadtxt(cur_system / 'type_map.raw', dtype=str)
        atype = np.loadtxt(cur_system / 'type.raw', dtype=int)
        chemical_symbols = [type_list[a] for a in atype]
        natoms = len(atype)
        #print(atype)

        print('plot system %s...' %name)
        # plot all data
        fig, axarr = plt.subplots(
            nrows=1, ncols=2, 
            gridspec_kw={'hspace': 0.3}, figsize=(16,12)
        )
        axarr = axarr.flatten()

        endata_name = pathlib.Path(
            res_dir / (name + '-energies.npy')
        ) # energies data
        nframes = -1
        if endata_name.exists():
            print('find ', endata_name)
            data = np.load(endata_name)
            ax = axarr[0]
            parity_plot_dict(
                {'energy': data[0]/natoms},
                {'energy': data[1]/natoms},
                ax,
                {
                    'xlabel': 'DFT [eV]',
                    'ylabel': '%s [eV]' %calc_name,
                    'title': 'Atomic Energies'
                }
            )
            nframes = data.shape[1]
            print('nframes', nframes)

        fig.suptitle(name + '-' + str(nframes))

        fordata_name = pathlib.Path(
            res_dir / (name + '-forces.npy')
        ) # forces data
        if fordata_name.exists():
            print('find ', fordata_name)
            data = np.load(fordata_name)
            #print(data.dtype)
            emulation_data, reference_data = {}, {}
            if elementwise and (nframes > 0): # means energy has been read
                global_indices = (np.arange(nframes)*natoms).reshape(-1,1)
                print('global shape ', global_indices.shape)
                for sym_idx, sym in enumerate(type_list):
                    sym_indices = np.argwhere(atype == sym_idx).flatten()
                    print('selected symbol ', sym, len(sym_indices))
                    if len(sym_indices) != 0:
                        force_indices = np.stack(
                            (sym_indices*3+0, sym_indices*3+1, sym_indices*3+2)
                        )
                        #print(force_indices)
                        force_indices = force_indices.reshape(1,-1)
                        sym_force_indices = (global_indices + force_indices).flatten()
                        #print(sym_force_indices.shape)
                        reference_data[sym] = data[0, sym_force_indices]
                        emulation_data[sym] = data[1, sym_force_indices]
                        #print(reference_data[sym].dtype)
            else:
                reference_data['force'] = data[0, :]
                emulation_data['force'] = data[1, :]
            ax = axarr[1]
            parity_plot_dict(
                reference_data,
                emulation_data,
                ax,
                {
                    'xlabel': 'DFT [eV/AA]',
                    'ylabel': '%s [eV/AA]' %calc_name,
                    'title': 'Forces'
                }
            )

        plt.savefig(res_dir / (name+'.png'))
        plt.clf() # clear current window

        #exit()
        #if idx > 2:
        #    exit()
    
    return

if __name__ == '__main__':
    # exclusive data
    parser = argparse.ArgumentParser()
    #analyse_mode = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        '-r', '--results', default='./results', 
        help='results directory'
    )
    parser.add_argument(
        '-c', '--calc', action='store_true',
        help='calculate data'
    )
    parser.add_argument(
        '-p', '--plot', action='store_true',
        help='plot data'
    )
    args = parser.parse_args()

    res_dir = pathlib.Path(args.results)

    # systems
    type_list = ['O', 'Pt']
    type_map = {'O': 0, 'Pt': 1}
    mass_list = [16.00, 195.08]

    if True:
        # Use DP
        from deepmd.calculator import DP 
        calc = DP(
            model = "/users/40247882/projects/oxides/gdp-main/it-0011/ensemble/model-0x/graph.pb",
            type_dict = type_map
        )
    else:
        # Use EANN
        import torch
        import sys
        sys.path.append('../')
        # from eann.src.trainer import EmbeddedAtomTrainer
        from eann.interface.ase.calculator import Eann
        dtype = torch.float64
        if torch.cuda.is_available():
            device = torch.device('gpu')
        else:
            device = torch.device('cpu')

        calc = Eann(
            model = 'EANN_PES_DOUBLE.pt',
            type_map = type_map
        )

    # start looping over
    dpdata_path = pathlib.Path('/users/40247882/projects/oxides/gdp-main/it-0011/raw_data')
    print(dpdata_path)

    system_dirs = list(dpdata_path.glob('O*'))
    system_dirs.sort()
    print("Number of Systems: ", len(system_dirs))

    if args.calc:
        if res_dir.exists():
            raise FileExistsError('results directory is not empty...')
        else:
            res_dir.mkdir(parents=True)
        calc_results(system_dirs, calc, args.results)
    
    if args.plot:
        if res_dir.exists():
            plot_results(system_dirs, calc.name, args.results, elementwise=False)
        else:
            raise FileNotFoundError()