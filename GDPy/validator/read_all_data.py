#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from ase.io import read, write

def read_vasp():
    pass

systems = ['Pt', 'PtO', 'aPtO2', 'bPtO2', 'Pt3O4']

frames = []
for s in systems:
    p = Path(s)
    vasprun = p / 'vasprun.xml'
    atoms = read(vasprun, format='vasp-xml')
    frames.append(atoms)

    eos_dirs = []
    p_eos = Path(s+'_eos')
    for e in p_eos.glob('*'):
        eos_dirs.append(e)
    eos_dirs.sort()
    for e in eos_dirs:
        vasprun = e / 'vasprun.xml'
        atoms = read(vasprun, format='vasp-xml')
        frames.append(atoms)

write('bulks.xyz', frames)

if __name__ == '__main__':
    pass
