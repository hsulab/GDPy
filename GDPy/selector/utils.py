#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
some utility functions for manipulating data 
"""

def read_arrays(datafile):
    """read array-like data file
        np.loadtxt works similarily but only for one datatype
    """
    with open(datafile, 'r') as reader:
        lines = reader.readlines()

    lines = [line.strip().split() for line in lines if not line.startswith('#')]

    #data = np.array(lines, dtype=float)
    data = lines

    return data

def show_build_progress(ncells, cur_idx):
    cur_per = int(( (cur_idx+1) / float(ncells) ) *100.)
    print("\r"+"|"*cur_per+" "+str(cur_per)+"%", end="")

    return 

def parse_indices(indices=None):
    """parse indices for reading xyz by ase, get start for counting"""
    if indices is not None:
        start, end = indices.split(':')
    else:
        start = 0
        end = ''

    return (start,end)


if __name__ == '__main__':
    pass
