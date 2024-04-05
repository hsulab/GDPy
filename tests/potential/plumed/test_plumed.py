#!/usr/bin/env python3
# -*- coding: utf-8 -*

import pytest

import numpy as np

from ase.io import read, write

ASE_NEQUIP_PLUMED_CONFIG = """potential:
  name: mixer
  params:
    backend: ase
    potters:
      - name: nequip
        params:
          backend: ase
          type_list: ["C", "H", "N", "O", "S"]
          model:
            - /mnt/scratch2/users/40247882/porous/nqtrain/r0/_ensemble/0004.train/m0/nequip.pth
      - name: plumed
        params:
          backend: ase
driver:
  task: md
  init:
    md_style: nvt
    timestep: 0.5
    temp: 360
    Tdamp: 100
    remove_translation: true
    remove_rotation: true
    dump_period: 2
  run:
    steps: 10
"""

def xxx():
    frames = read("./xxx/cand0/traj.xyz", ":")
    for atoms in frames:
        positions = atoms.positions
        dis = np.linalg.norm(positions[0] - positions[1])
        print(dis)

def run():

    return

if __name__ == "__main__":
    ...