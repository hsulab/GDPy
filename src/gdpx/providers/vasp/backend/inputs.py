"""Prepare conventional VASP inputs while sharing immutable restart inputs."""
import filecmp
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from ase.io import read
from ase.calculators.vasp.create_input import format_kpoints, open_potcar

SHARED_INPUTS = ("POTCAR", "KPOINTS", "ase-sort.dat")


def relative_link(path, target):
    """Publish a relative symlink atomically, without writing through a link."""
    path, target = Path(path), Path(target)
    relative = os.path.relpath(target, path.parent)
    if path.is_symlink() and os.readlink(path) == relative:
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".vasp-link-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    temporary.unlink()
    try:
        temporary.symlink_to(relative)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def detach_input_links(directory):
    """A fresh input write must never follow links into an older calculation."""
    for name in ("INCAR", "POSCAR", *SHARED_INPUTS):
        path = Path(directory) / name
        if path.is_symlink():
            path.unlink()


def share_checkpoint_inputs(directory, checkpoint, moved_links):
    """Rebase moved links and share identical legacy copies with 0000.run."""
    directory, checkpoint = Path(directory), Path(checkpoint)
    for name, target in moved_links.items():
        if (checkpoint / name).is_symlink():
            relative_link(checkpoint / name, target)
    anchor = directory / "0000.run"
    for old in sorted(directory.glob("[0-9][0-9][0-9][0-9].run")):
        if old == anchor:
            continue
        for name in SHARED_INPUTS:
            source, target = old / name, anchor / name
            if not target.is_file() or target.is_symlink():
                continue
            # Recover the known link layout if interrupted after moving the link.
            if source.is_symlink() and os.readlink(source) == f"0000.run/{name}":
                relative_link(source, target)
            elif source.is_file() and filecmp.cmp(source, target, shallow=False):
                relative_link(source, target)


def prepare_restart_inputs(calc, atoms, checkpoint, directory):
    """Write mutable inputs only; link unchanged inputs to the first run."""
    directory, checkpoint = Path(directory), Path(checkpoint)
    anchor = directory / "0000.run"

    def incompatible(message):
        raise ValueError(f"Cannot share VASP restart inputs: {message}. Start a new calculation directory.")

    for name in ("POTCAR", "ase-sort.dat", "POSCAR"):
        if not (anchor / name).is_file() or (anchor / name).is_symlink():
            incompatible(f"0000.run/{name} must be a regular file")
    for name in SHARED_INPUTS:
        original, previous = anchor / name, checkpoint / name
        if original.exists() != previous.exists():
            incompatible(f"{name} is missing from the original or previous run")
        if original.exists() and not filecmp.cmp(original, previous, shallow=False):
            incompatible(f"{name} changed between checkpoints")
    if not (checkpoint / "CONTCAR").is_file():
        incompatible("previous CONTCAR is missing")

    calc.initialize(atoms)
    original_atoms = read(anchor / "POSCAR", format="vasp")
    previous_atoms = read(checkpoint / "CONTCAR", format="vasp")
    symbols = calc.atoms_sorted.get_chemical_symbols()
    if original_atoms.get_chemical_symbols() != symbols or previous_atoms.get_chemical_symbols() != symbols:
        incompatible("species order or atom count changed")
    sorting = np.loadtxt(anchor / "ase-sort.dat", dtype=int, ndmin=2)
    if sorting.shape != (len(atoms), 2) or not np.array_equal(
        sorting, np.column_stack((calc.sort, calc.resort))
    ):
        incompatible("atom sorting changed")

    expected = hashlib.sha256()
    # Match ASE's POTCAR concatenation without generating another POTCAR.
    for filename in calc.ppp_list:
        with open_potcar(filename=filename) as stream:
            for line in stream:
                expected.update(line.encode())
    actual = hashlib.sha256()
    with (anchor / "POTCAR").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            actual.update(chunk)
    if expected.digest() != actual.digest():
        incompatible("pseudopotential contents or ordering changed")

    kspacing = calc.float_params["kspacing"]
    if kspacing is not None and kspacing <= 0:
        incompatible("KSPACING must be positive")
    if kspacing is not None or calc.input_params["kpts"] is None:
        kpoints = None
    else:
        kpoints = format_kpoints(
            kpts=calc.input_params["kpts"], atoms=atoms,
            reciprocal=calc.input_params["reciprocal"],
            gamma=calc.input_params["gamma"],
        )
    original_kpoints = (anchor / "KPOINTS").read_text() if (anchor / "KPOINTS").exists() else None
    if kpoints != original_kpoints:
        incompatible("k-point settings changed")

    # Validation finishes before publishing links or changing any inputs.
    for name in ("INCAR", "POSCAR"):
        path = directory / name
        if path.is_symlink():
            path.unlink()
    calc.write_incar(atoms, directory=str(directory))
    shutil.copyfile(checkpoint / "CONTCAR", directory / "POSCAR")
    for name in SHARED_INPUTS:
        if (anchor / name).is_file():
            relative_link(directory / name, anchor / name)
    calc.copy_vdw_kernel(directory=str(directory))
