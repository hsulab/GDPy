"""Regenerate the small EMT demonstration structures (requires ASE)."""
from pathlib import Path

from ase import Atoms
from ase.build import bulk
from ase.io import write

assets = Path(__file__).resolve().parent / "assets"
assets.mkdir(exist_ok=True)
copper = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
write(assets / "cu32.xyz", copper)
alloy = copper.copy()
alloy.numbers[::2] = 28  # Cu16Ni16 on the same fixed lattice.
write(assets / "cu16ni16.xyz", alloy)
# One Au atom remains throughout exchange, so the calculator never sees an
# empty structure. This is a toy reservoir example, not a bulk phase model.
gas = Atoms("AuCu8", positions=[(4, 4, 4)] + [
    (x, y, z) for x in (1, 7) for y in (1, 7) for z in (1, 7)
], cell=(8, 8, 8), pbc=True)
write(assets / "aucu8.xyz", gas)
