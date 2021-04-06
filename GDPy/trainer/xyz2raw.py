#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from ase.io import read, write

import dpdata 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-df', '--datafile', 
        default='bias', help='time series files'
    )

    args = parser.parse_args()
    # sanity check, dpdata only needs species, pos, Z, force, virial 
    # ase-extxyz is inconsistent with quip-xyz, especially the force 
    frames = read(args.datafile, ':')

    atomic_properties = ['numbers', 'positions', 'forces']
    calc_props = ['energy', 'forces']

    for atoms in frames:
        # remove extra properties in atoms
        cur_properties = list(atoms.arrays.keys())
        for prop in cur_properties:
            if prop not in atomic_properties:
                #atoms.arrays.pop(prop)
                del atoms.arrays[prop]
        # atoms info 
        # del atoms.info['feature_vector']
        # TODO: check if calculator exists 
        atoms.calc = None # ase copys xyz info to SinglePointCalculator?
        atoms.arrays['force'] = atoms.arrays['forces'].copy()
        del atoms.arrays['forces']

        # calc
        #cur_calc_props = list(atoms.calc.results.keys())
        #for prop in cur_calc_props:
        #    if prop not in calc_props:
        #        del atoms.calc.results[prop]
        # move forces to force

    write('dp_raw.xyz', frames)

    # 
    xyz_multi_systems = dpdata.MultiSystems.from_file(
        file_name='./dp_raw.xyz', 
        fmt='quip/gap/xyz'
    )
    print(xyz_multi_systems)
    xyz_multi_systems.to_deepmd_raw('./raw_data/')
    pass
