"""Generate inexpensive Cu structures and relaxed Al/Au NEB endpoints."""
from pathlib import Path

from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import BFGS

root = Path(__file__).resolve().parent
frames = []
for distance in (2.0, 2.5, 3.0):
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [distance, 0, 0]],
                  cell=[20, 20, 20], pbc=True)
    atoms.center()
    frames.append(atoms)
write(root / "dimers.xyz", frames)
write(root / "bulk.xyz", bulk("Cu", "fcc", a=3.8, cubic=True))
write(root / "md.xyz", bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2)))

initial = fcc100("Al", size=(2, 2, 3))
add_adsorbate(initial, "Au", 1.7, "hollow")
initial.center(axis=2, vacuum=4.0)
final = initial.copy()
final[-1].x += final.cell[0, 0] / 2.0
for atoms in (initial, final):
    atoms.info.pop("adsorbate_info", None)
    atoms.set_constraint(FixAtoms(indices=range(8)))
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=0.08)
write(root / "endpoints.xyz", [initial, final])
print(f"Wrote example structures to {root}")
