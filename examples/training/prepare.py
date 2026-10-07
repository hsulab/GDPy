"""Generate 24 reproducible EMT-labeled Cu3Au1 training structures."""

import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write


def prepare_dataset(directory: Path) -> Path:
    rng = np.random.default_rng(1112)
    frames = []
    for _ in range(24):
        atoms = bulk("Cu", "fcc", a=3.73, cubic=True)
        atoms[rng.integers(len(atoms))].symbol = "Au"
        atoms.set_cell(atoms.cell * rng.uniform(0.97, 1.03), scale_atoms=True)
        atoms.positions += rng.normal(0.0, 0.04, size=atoms.positions.shape)
        atoms.wrap()
        atoms.calc = EMT()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        # Keep virial in the XYZ header through GDPy's dataset conversion.
        atoms.info["virial"] = -atoms.get_volume() * stress
        atoms.calc = SinglePointCalculator(
            atoms, energy=energy, forces=forces, stress=stress
        )
        frames.append(atoms)

    destination = directory / "init-Cu3Au1-bulk" / "data.xyz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write(destination, frames, format="extxyz")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Dataset root (default: dataset beside this script).",
    )
    args = parser.parse_args()
    print(prepare_dataset(args.directory).resolve())
