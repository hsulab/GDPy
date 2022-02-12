#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from ase.io import read, write
from ase.io.lammpsdata import read_lammps_data, write_lammps_data

init_sys = Path('./init-systems')
for p in init_sys.glob('O*'):
    atoms = read(p, format='lammps-data', style='atomic', units='metal')
    write(init_sys/'charge'/p.name, atoms, format='lammps-data', units='real', atom_style='charge')


if __name__ == '__main__':
    pass
