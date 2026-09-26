#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np
from ase import Atoms


def get_file_md5(f):
    import hashlib

    m = hashlib.md5()
    while True:
        # if not using binary
        # data = f.read(1024).encode('utf-8')
        data = f.read(1024)  # read in block
        if not data:
            break
        m.update(data)
    return m.hexdigest()


def copy_minimal_frames(prev_frames: List[Atoms]):
    """Copy atoms without extra information.

    Keep calculation inputs; exclude bookkeeping and attached calculators.

    """
    curr_frames, curr_info = [], []
    for prev_atoms in prev_frames:
        # Preserve all calculation inputs, including constraints and custom arrays.
        curr_atoms = prev_atoms.copy()
        curr_atoms.info = {}
        curr_frames.append(curr_atoms)
        # - save info
        confid = prev_atoms.info.get("confid", -1)
        dynstep = prev_atoms.info.get("step", -1)
        prev_wdir = prev_atoms.info.get("wdir", "null")
        curr_info.append((confid, dynstep, prev_wdir))

    return curr_frames, curr_info


def read_cache_info(wdir, length=36):
    # - read extra info data
    _info_data = []
    for p in (wdir / "_data").glob("*_info.txt"):
        identifier = p.name[:length]  # MD5
        with open(p, "r") as fopen:
            for line in fopen.readlines():
                if not line.startswith("#"):
                    _info_data.append(line.strip().split())
    _info_data = sorted(_info_data, key=lambda x: int(x[0]))

    return _info_data


def split_batches(nframes: int, batchsize: int = 1) -> tuple[list[int], list[int]]:
    """Split nframes into groups."""
    num_batches = int(np.floor(1.0 * nframes / batchsize))
    batch_indices = [0]
    for i in range(num_batches):
        batch_indices.append((i + 1) * batchsize)
    if batch_indices[-1] != nframes:
        batch_indices.append(nframes)
    starts, ends = batch_indices[:-1], batch_indices[1:]
    assert len(starts) == len(ends), "Inconsistent start and end indices."

    return (starts, ends)


if __name__ == "__main__":
    ...
