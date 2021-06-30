#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Structure Sampling Generation (RSS-G)
using rss to generate random structures
"""

import os
import time
import subprocess
import argparse 

import numpy as np 

# avoid ase-castep warnings 
import warnings
warnings.filterwarnings('ignore')

from ase.io import read, write
from ase.io.castep import read_castep_cell

BUILDCELL = "/mnt/scratch/chemistry-apps/dkb01416/airss/developed/airss-0.9.1/bin/buildcell"


def show_build_progress(ncells, cur_idx):
    cur_per = int(( (cur_idx+1) / float(ncells) ) *100.)
    print("\r"+"|"*cur_per+" "+str(cur_per)+"%", end="")

    return 

def airss_buildcell(seed_cell, max_cells):
    """ use airss to randomly generate cells, 
        and return a list of ase atoms
    """
    # init log file
    #dprss_logger.info('# ===== AIRSS buildcell\n')
    with open('build.log', 'w') as fopen:
        fopen.write('# ===== AIRSS buildcell\n')

    # start build
    frames = []
    command = BUILDCELL + " < %s" %(seed_cell)

    for idx in range(max_cells):
        content = 'Cell Number %s\n' %(str(idx).zfill(8))
        proc = subprocess.Popen(
            command, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            encoding = 'utf-8'
        )
        errorcode = proc.wait(timeout=120) # 10 seconds 
        if errorcode:
            raise ValueError('Error in generating random cells.')

        atoms = read_castep_cell(proc.stdout)

        # sift info and arrays
        del atoms.arrays['initial_magmoms']
        del atoms.arrays['castep_labels']

        # sort atoms by symbols and z-positions especially for supercells 
        numbers = atoms.numbers 
        zposes = atoms.positions[:,2].tolist()
        sorted_indices = np.lexsort((zposes,numbers))
        atoms = atoms[sorted_indices]

        frames.append(atoms)

        for line in proc.stderr:
            content += line
        content += '\n'
        #dprss_logger.info(content)
        with open('build.log', 'a') as fopen:
            fopen.write(content)

        show_build_progress(max_cells, idx)
    print('')

    return frames

def write_rss_frames(tracker, seed, ncell, output):
    """"""
    print('Generating random structures...')
    frames = airss_buildcell(seed, ncell)
    write(output, frames)

    # TODO: add the tracker
    #tracker.track_progress('rss', nrs=ncell)

    return 

if __name__ == '__main__':
    # === stage 1. generate structures randomly 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seed', required=True, 
        help='rss seed cell'
    )
    parser.add_argument(
        '-n', '--ncell', 
        default=1000, type=int, 
        help='number of cells generated'
    )
    parser.add_argument(
        '-o', '--output', 
        default='rss.xyz', 
        help='histogram data file'
    )

    args = parser.parse_args()

    frames = airss_buildcell(args.seed, args.ncell)
    write(args.output, frames)

    pass
