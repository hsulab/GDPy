"""Print the stored MC states, including repeated states after rejection."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
frames = read(args.directory / "mc.xyz", ":")
print("frame  atoms  composition  energy_eV")
for index, atoms in enumerate(frames):
    counts = " ".join(f"{symbol}:{count}" for symbol, count in sorted(Counter(atoms.symbols).items()))
    print(f"{index:5d}  {len(atoms):5d}  {counts:16s}  {atoms.get_potential_energy():.6f}")
print("Fixed cell:", all(np.allclose(a.cell, frames[0].cell) for a in frames))
