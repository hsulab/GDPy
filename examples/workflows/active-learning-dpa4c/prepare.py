"""Prepare the shared 24-frame EMT Cu3Au1 dataset and an unlabeled MD start."""

from pathlib import Path
import runpy

from ase.io import read, write


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    generator = runpy.run_path(str(directory.parents[1] / "training" / "prepare.py"))
    dataset = generator["prepare_dataset"](directory / "dataset")
    atoms = read(dataset, index=0)
    atoms.calc = None
    atoms.info.clear()
    write(directory / "cu3au1.xyz", atoms, format="extxyz")
    print(dataset)
