#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import List

from ase import Atoms


def get_file_md5(f):
    import hashlib
    m = hashlib.md5()
    while True:
        # if not using binary
        #data = f.read(1024).encode('utf-8')
        data = f.read(1024) # read in block
        if not data:
            break
        m.update(data)
    return m.hexdigest()


def copy_minimal_frames(prev_frames: List[Atoms]):
    """Copy atoms without extra information.

    Do not copy atoms.info since it is a dict and does not maitain order.

    """
    curr_frames, curr_info = [], []
    for prev_atoms in prev_frames:
        # - copy geometry
        curr_atoms = Atoms(
            symbols=copy.deepcopy(prev_atoms.get_chemical_symbols()),
            positions=copy.deepcopy(prev_atoms.get_positions()),
            cell=copy.deepcopy(prev_atoms.get_cell(complete=True)),
            pbc=copy.deepcopy(prev_atoms.get_pbc()),
            tags = prev_atoms.get_tags() # retain this for molecules
        )
        if prev_atoms.get_kinetic_energy() > 0.: # retain this for MD
            curr_atoms.set_momenta(prev_atoms.get_momenta()) 
        curr_frames.append(curr_atoms)
        # - save info
        confid = prev_atoms.info.get("confid", -1)
        dynstep = prev_atoms.info.get("step", -1)
        prev_wdir = prev_atoms.info.get("wdir", "null")
        curr_info.append((confid,dynstep,prev_wdir))

    return curr_frames, curr_info


def read_cache_info(wdir, length=36):
    # - read extra info data
    _info_data = []
    for p in (wdir/"_data").glob("*_info.txt"):
        identifier = p.name[:length] # MD5
        with open(p, "r") as fopen:
            for line in fopen.readlines():
                if not line.startswith("#"):
                    _info_data.append(line.strip().split())
    _info_data = sorted(_info_data, key=lambda x: int(x[0]))

    return _info_data


if __name__ == "__main__":
    ...