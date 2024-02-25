#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use("presentation")

from ase.io import read, write

parser = argparse.ArgumentParser()
parser.add_argument("TRAJECTORY")
args = parser.parse_args()


frames = read(args.TRAJECTORY, ":")
nsteps = len(frames)
energies = np.array([a.get_potential_energy() for a in frames])
energies -= energies[0]
steps = range(nsteps)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_title("Monte Carlo")
ax.set_ylabel("Potential Energy [eV]")
ax.set_xlabel("MC Step")

ax.plot(steps, energies, alpha=0.5, marker="o")


plt.tight_layout()
plt.savefig("./mctraj.png")


if __name__ == "__main__":
    ...
