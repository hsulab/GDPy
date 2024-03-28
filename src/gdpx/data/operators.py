#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pathlib
from typing import Union

import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from gdpx.utils.comparasion import parity_plot, parity_plot_dict, PropInfo


import matplotlib as mpl
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except:
    ...


def use_dpeval():
    import matplotlib as mpl
    mpl.use('Agg') #silent mode
    from matplotlib import pyplot as plt
    plt.style.use('presentation')

    import deepmd.DeepPot as DeepPot

    type_map = {"O": 0, "Pt": 1}
    inverse_type_map = {0: "O", 1: "Pt"}

    raw_dirs = pathlib.Path('../oxides/gdp-main/it-0011/raw_data')

    # find system
    sys_map = {"O": 1, "Pt": 36}
    sys_name = []
    for k, v in sys_map.items():
        sys_name.extend([str(k), str(v)])
    sys_name = ''.join(sys_name)

    test = raw_dirs / sys_name
    print(test)

    natoms = np.sum(list(sys_map.values())) 

    set_dirs = list(test.glob('set*'))
    set_dirs.sort()

    # train data
    train_boxes, train_coords, train_energies, train_forces = [], [], [], []
    for set_dir in set_dirs[:-1]:
        train_boxes.extend( np.load(set_dir / 'box.npy').tolist() )
        train_coords.extend( np.load(set_dir / 'coord.npy').tolist() )
        train_energies.extend( np.load(set_dir / 'energy.npy').tolist() )
        train_forces.extend( np.load(set_dir / 'force.npy').tolist() )
        #print(boxes.shape)
        #print(boxes)

    # test_data
    test_boxes, test_coords, test_energies, test_forces = [], [], [], []
    for set_dir in set_dirs[-1:]:
        test_boxes.extend( np.load(set_dir / 'box.npy').tolist() )
        test_coords.extend( np.load(set_dir / 'coord.npy').tolist() )
        test_energies.extend( np.load(set_dir / 'energy.npy').tolist() )
        test_forces.extend( np.load(set_dir / 'force.npy').tolist() )

    # read types
    atype = np.loadtxt(test / 'type.raw', dtype=int)
    print(atype)

    # start with a square Figure
    fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(16,16))
    axarr = axarr.flatten()

    plt.suptitle('Dataset Overview')
    #plt.tight_layout()
    plt.subplots_adjust(
        left=0.10, right=0.95,
        bottom=0.10, top=0.85,
        wspace=0.20, hspace=0.30
    )

    plot_hist(axarr[0], np.array(train_energies).flatten()/natoms, 'Energy [eV]', 'Number of Frames')
    plot_hist(axarr[1], np.array(train_forces).flatten(), 'Force [eV/AA]', 'Number of Frames')
    plot_hist(axarr[0], np.array(test_energies).flatten()/natoms, 'Energy [eV]', 'Number of Frames')
    plot_hist(axarr[1], np.array(test_forces).flatten(), 'Force [eV/AA]', 'Number of Frames')

    # use dp to check test
    dp = DeepPot('../oxides/gdp-main/it-0011/ensemble/model-0/graph.pb')
    dp_input = {'coords': test_coords, 'cells': test_boxes, 'atom_types': atype, 'atomic': True}
    e, f, v, ae, av = dp.eval(**dp_input)

    prop_info = PropInfo(xlabel='miaow', ylabel='miaow', title='xxx')
    parity_plot(['DP', e.flatten()/natoms], ['DFT', np.array(test_energies).flatten()/natoms], axarr[2], ('energy', '[eV/atom]'), sys_name)
    #parity_plot(['DP', f.flatten()], ['DFT', np.array(test_forces).flatten()], axarr[3], ('force', '[eV/AA]'), sys_name)
    parity_plot_dict(transform_forces(atype, f.reshape(-1,natoms*3)), transform_forces(atype, test_forces), axarr[3], prop_info)

    dp_input = {'coords': train_coords, 'cells': train_boxes, 'atom_types': atype, 'atomic': True}
    e, f, v, ae, av = dp.eval(**dp_input)

    #parity_plot(['DP', e.flatten()/natoms], ['DFT', np.array(test_energies).flatten()/natoms], axarr[4], ('energy', '[eV/atom]'), sys_name)
    #parity_plot_dict(transform_forces(atype, f.reshape(-1,natoms*3)), transform_forces(atype, test_forces), axarr[5], prop_info)

    plt.savefig('wang.png')

    return

def set2inputs(set_dir):
    """ extract info from dpdata
    """
    boxes = np.load(set_dir / 'box.npy')
    coords = np.load(set_dir / 'coord.npy')
    energies = np.load(set_dir / 'energy.npy')
    forces = np.load(set_dir / 'force.npy')

    nframes = boxes.shape[0]
    boxes = boxes.reshape(-1,3,3)
    #boxes = torch.from_numpy(boxes)
    positions = coords.reshape(nframes,-1,3)
    #positions = torch.from_numpy(positions)
    natoms = positions.shape[1]
    energies = energies.flatten().tolist()
    forces = forces.flatten().tolist()

    return (
        nframes, natoms, 
        energies, forces, # 1D-List
        positions, boxes # tensor
    )

def sets2results(set_dirs, calc, chemical_symbols):
    """ test set_dirs
    """
    ref_energies, ref_forces = [], []
    mlp_energies, mlp_forces = [], []
    for set_dir in set_dirs:
        set_start_time = time.time()
        (
            nframes, natoms, 
            energies, forces,
            positions, boxes
        ) = set2inputs(set_dir)
        ref_energies.extend(energies)
        ref_forces.extend(forces)

        print('nframes in train', nframes)
        for i in range(nframes):
            atoms = Atoms(
                symbols=chemical_symbols, positions=positions[i], 
                cell=boxes[i], pbc=[1,1,1]
            )
            #results = {'energy': energies[i], 'forces': forces[i,:].reshape(-1,3)}
            #spc = SinglePointCalculator(atoms, **results)
            #atoms.calc = spc
            calc.reset()
            atoms.calc = calc
            #print(type(atoms.get_potential_energy()))
            #print(type(atoms.get_forces()))
            mlp_energies.append(float(atoms.get_potential_energy()))
            mlp_forces.extend(atoms.get_forces().flatten().tolist())
        set_end_time = time.time()
        print('test time: ', set_end_time - set_start_time)

    return ref_energies, ref_forces, mlp_energies, mlp_forces

def append_predictions(
    frames, calc = None, other_props = [], calc_uncertainty=False
):
    """ test xyz
    """
    if calc is not None:
        calc_name = calc.name.lower()

    tot_energies, tot_forces = [], {}
    tot_props = {}

    for atoms in frames: # free energy per atom
        # set calculator
        calc_atoms = atoms.copy()
        if calc is not None:
            calc.reset()
            if calc_uncertainty:
                calc.calc_uncertainty = True # EANN specific
            calc_atoms.calc = calc
            new_forces = calc_atoms.get_forces(apply_constraint=False)
            new_energy = calc_atoms.get_potential_energy()
            atoms.info[calc_name+"_energy"] = new_energy
            atoms.arrays[calc_name+"_forces"] = new_forces.copy()
        
    return frames

def merge_forces(frames):
    """ convert forces of all frames into a dict,
        which helps further comparison
    """
    tot_forces = {}
    for atoms in frames: # free energy per atom
        # basic info
        symbols = atoms.get_chemical_symbols()

        # force
        forces = atoms.get_forces().copy()
        for sym, force in zip(symbols, forces.tolist()):
            if sym in tot_forces.keys():
                tot_forces[sym].extend(force)
            else:
                tot_forces[sym] = force

    return tot_forces

def merge_predicted_forces(frames, calc_name):
    """ convert forces of all frames into a dict,
        which helps further comparison
    """
    tot_forces = {}
    for atoms in frames: # free energy per atom
        # basic info
        symbols = atoms.get_chemical_symbols()

        # force
        forces = atoms.arrays[calc_name+"_forces"].copy()
        for sym, force in zip(symbols, forces.tolist()):
            if sym in tot_forces.keys():
                tot_forces[sym].extend(force)
            else:
                tot_forces[sym] = force

    return tot_forces


def xyz2results(frames, calc = None, other_props = []):
    """ test xyz
    """
    #new_frames = frames.copy()

    tot_energies, tot_forces = [], {}
    tot_props = {}

    for atoms in frames: # free energy per atom
        # basic info
        symbols = atoms.get_chemical_symbols()

        # set calculator
        if calc is not None:
            calc.reset()
            calc.calc_uncertainty = True # EANN specific
            atoms.calc = calc

        # energy
        energy = atoms.get_potential_energy() 
        tot_energies.append(energy)

        # force
        forces = atoms.get_forces(apply_constraint=False)
        for sym, force in zip(symbols, forces.tolist()):
            if sym in tot_forces.keys():
                tot_forces[sym].extend(force)
            else:
                tot_forces[sym] = force
        
        # other properties
        if calc is not None:
            for prop in other_props:
                if prop in tot_props.keys():
                    tot_props[prop].extend([atoms.calc.results[prop]])
                else:
                    tot_props[prop] = [atoms.calc.results[prop]]

    if tot_props:
        return tot_energies, tot_forces, tot_props
    else:
        return tot_energies, tot_forces

def calc_and_compare_results(frames, calc):
    """"""
    ref_energies, ref_forces = xyz2results(frames, calc=None)
    natoms_array = np.array([len(x) for x in frames])

    calc_name = calc.name.lower()
    #print("Calculating with MLP...")
    new_frames = append_predictions(frames, calc=calc)
    #mlp_energies, mlp_forces = xyz2results(frames, calc=calc)
    mlp_energies = [a.info[calc_name+"_energy"] for a in new_frames]
    mlp_forces = merge_predicted_forces(new_frames, calc_name) 

    forces = np.array([ref_forces, mlp_forces])
    energies = np.array([ref_energies, mlp_energies]) / natoms_array

    return energies, forces


def plot_comparasion(calc_name, energies, forces, saved_figure):
    """"""
    fig, axarr = plt.subplots(
        nrows=1, ncols=2,
        gridspec_kw={'hspace': 0.3}, figsize=(16, 12)
    )
    axarr = axarr.flatten()

    _, nframes = energies.shape
    plt.suptitle(f"Number of frames {nframes}")

    ax = axarr[0]
    en_rmse_results = parity_plot_dict(
        {"energy": energies[0, :]}, {"energy": energies[1, :]}, 
        ax, 
        {
            "xlabel": "DFT [eV]", "ylabel": "%s [eV]" % calc_name, "title": "Energy"
        }
    )

    ax = axarr[1]
    for_rmse_results = parity_plot_dict(
        forces[0], forces[1],
        ax,
        {
            "xlabel": "DFT [eV/AA]",
            "ylabel": "%s [eV/AA]" % calc_name,
            "title": "Forces"
        }
    )

    plt.savefig(saved_figure)

    return en_rmse_results, for_rmse_results