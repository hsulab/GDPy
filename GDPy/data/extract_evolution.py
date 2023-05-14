#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time

from pathlib import Path
from collections import namedtuple

import numpy as np

from ase.io import read, write

from tqdm import tqdm

from joblib import Parallel, delayed

def extract_mctraj():
    suspects = Path("./suspects")
    step_dirs = []
    #step_dirs = list(suspects.glob("step[0-9]"))
    #step_dirs.extend(list(suspects.glob("step[0-9][0-9]")))
    #step_dirs.extend(list(suspects.glob("step[0-9][0-9][0-9]")))
    #step_dirs.extend(list(suspects.glob("step[0-9][0-9][0-9][0-9]")))
    
    for p in suspects.iterdir():
        if re.match(r"step[0-9]{1,4}$", str(p.name)):
            step_dirs.append(p)
    print(len(step_dirs))
    
    step_dirs_sorted = sorted(step_dirs, key=lambda k: int(k.name[4:])) # sort by name
    #print(step_dirs_sorted)
    
    StepInfo = namedtuple("StepInfo", ["natoms", "energy", "deviation"])
    
    def extract_data(p):
        # total energy deviation
        devi_file = p / "model_devi.out"
        devi_info = np.loadtxt(devi_file) # EANN
        en_devi = float(devi_info[1])
        #print(en_devi)
        # total energy
        with open(p / "log.lammps", "r") as fopen:
            lines = fopen.readlines()
        for idx, line in enumerate(lines):
            if line.startswith('Minimization stats:'):
                stat_idx = idx
                break
        else:
            raise ValueError('error in lammps minimization.')
        en = float(lines[stat_idx+3].split()[-1])
        # natoms
        dump_atoms = read(
            p / "surface.dump", ':', 'lammps-dump-text', 
            specorder=["O","Pt"], units="metal"
        )[-1]
        natoms = len(dump_atoms)
        #print(natoms)
        cur_data = StepInfo(natoms, en, en_devi)
    
        return cur_data
    
    
    data = []
    for p in tqdm(step_dirs_sorted[:10]):
        cur_data = extract_data(p)
        data.append(cur_data)
    
    # write to file
    content = "#{}  {}  {}  {}\n".format("step   ", "natoms  ", "energy  ", "deviation")
    for i, info in enumerate(data):
        content += ("{:8d}  "*2+"{:8.4f}  "*2+"\n").format(
            i, info.natoms, info.energy, info.deviation
        )
    with open("evolution.dat", "w") as fopen:
        fopen.write(content)
    
    exit()
    
    import matplotlib as mpl
    mpl.use('Agg') #silent mode
    from matplotlib import pyplot as plt
    plt.style.use('presentation')
    
    fig, axarr = plt.subplots(
        nrows=1, ncols=1, 
        gridspec_kw={'hspace': 0.3}, figsize=(16,12)
    )
    axarr = [axarr]
    plt.suptitle("GCMC Evolution")
    
    ax = axarr[0]
    # element_numbers = Counter()
    natoms_array = np.array([a.natoms for a in data])
    # oxygen_array = np.array([Counter(a.get_chemical_symbols())["O"] for a in frames])
    steps = range(len(data))
    energies = np.array([a.energy for a in data]) / natoms_array
    en_stdvars = np.array([a.deviation for a in data]) / natoms_array
    
    ax.set_xlabel("MC Step")
    ax.set_ylabel("Energy [eV]")
    
    ax.plot(steps, energies, label="energy per atom")
    apex = 10.
    ax.fill_between(
        steps, energies-apex*en_stdvars, energies+apex*en_stdvars, alpha=0.2,
        label="10 times deviation per atom"
    )
    ax.legend()
    
    plt.savefig("m-stdvar.png")


if __name__ == '__main__':
    pass
