import os
from pathlib import Path
import shutil
import tarfile
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp

from gdpx.providers.vasp.backend.inputs import (
    SHARED_INPUTS, detach_input_links, prepare_restart_inputs,
)
from gdpx.providers.vasp.driver import VaspDriver


@pytest.fixture
def calculation(tmp_path):
    pp = tmp_path / "pp"
    for symbol in ("H", "O"):
        (pp / symbol).mkdir(parents=True)
        (pp / symbol / "POTCAR").write_text(f"Synthetic test potential {symbol}\n")
    directory = tmp_path / "calc"
    directory.mkdir()
    atoms = Atoms("HOH", positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=[8, 8, 8], pbc=True)
    calc = Vasp(xc="PBE", pp=str(pp), kpts=(1, 1, 1), directory=directory)
    calc.write_input(atoms)
    shutil.copyfile(directory / "POSCAR", directory / "CONTCAR")
    driver = VaspDriver(calc, {"task": "min", "steps": 20}, directory=directory)
    return directory, atoms, calc, driver, pp


def test_repeated_restart_uses_relative_links_without_shared_writers(calculation):
    directory, atoms, calc, driver, _ = calculation
    anchor = driver._save_checkpoint()
    assert anchor.name == "0000.run"
    originals = {name: (anchor / name).read_bytes() for name in SHARED_INPUTS}
    mtimes = {name: (anchor / name).stat().st_mtime_ns for name in SHARED_INPUTS}
    with patch.object(calc, "write_input", side_effect=AssertionError("full writer")), patch.object(
        calc, "write_potcar", side_effect=AssertionError("POTCAR writer")
    ), patch.object(calc, "write_kpoints", side_effect=AssertionError("KPOINTS writer")), patch.object(
        calc, "write_sort_file", side_effect=AssertionError("sort writer")
    ), patch("gdpx.providers.vasp.driver.run_ase_calculator"):
        for _ in range(3):
            checkpoint = sorted(directory.glob("*.run"))[-1]
            driver._irun(atoms, ckpt_wdir=checkpoint, cache_traj=[atoms])
            for name in SHARED_INPUTS:
                assert os.readlink(directory / name) == f"0000.run/{name}"
                assert (directory / name).read_bytes() == originals[name]
            assert not (directory / "INCAR").is_symlink()
            assert not (directory / "POSCAR").is_symlink()
            assert (directory / "POSCAR").read_bytes() == (checkpoint / "CONTCAR").read_bytes()
            shutil.copyfile(directory / "POSCAR", directory / "CONTCAR")
            new_checkpoint = driver._save_checkpoint()
            for name in SHARED_INPUTS:
                assert os.readlink(new_checkpoint / name) == f"../0000.run/{name}"
                assert (new_checkpoint / name).read_bytes() == originals[name]
    for name in SHARED_INPUTS:
        assert not (anchor / name).is_symlink()
        assert (anchor / name).stat().st_mtime_ns == mtimes[name]
        assert (anchor / name).read_bytes() == originals[name]


@pytest.mark.parametrize("change", ["potential", "kpoints", "sorting", "species"])
def test_incompatible_restart_is_rejected(calculation, change):
    directory, atoms, calc, driver, pp = calculation
    anchor = driver._save_checkpoint()
    originals = {name: (anchor / name).read_bytes() for name in SHARED_INPUTS}
    if change == "potential":
        (pp / "H" / "POTCAR").write_text("Changed potential\n")
    elif change == "kpoints":
        calc.set(kpts=(2, 2, 2))
    elif change == "sorting":
        (anchor / "ase-sort.dat").write_text("0 0\n1 1\n2 2\n")
        originals["ase-sort.dat"] = (anchor / "ase-sort.dat").read_bytes()
    else:
        atoms = Atoms("HHH", cell=[8, 8, 8], pbc=True)
    with pytest.raises(ValueError, match="Start a new calculation directory"):
        prepare_restart_inputs(calc, atoms, anchor, directory)
    assert not (directory / "INCAR").exists()
    for name in SHARED_INPUTS:
        assert (anchor / name).read_bytes() == originals[name]


def test_existing_regular_checkpoints_are_deduplicated(calculation):
    directory, atoms, calc, driver, _ = calculation
    anchor = driver._save_checkpoint()
    # Simulate the old restart writer creating ordinary duplicate files.
    calc.write_input(atoms)
    shutil.copyfile(anchor / "CONTCAR", directory / "CONTCAR")
    second = driver._save_checkpoint()
    for name in SHARED_INPUTS:
        assert os.readlink(second / name) == f"../0000.run/{name}"
    # A historically different input is retained, never silently relinked.
    calc.write_input(atoms)
    (directory / "KPOINTS").write_text("different historical grid")
    third = driver._save_checkpoint()
    assert not (third / "KPOINTS").is_symlink()
    assert (third / "KPOINTS").read_text() == "different historical grid"
    with pytest.raises(ValueError, match="KPOINTS changed"):
        prepare_restart_inputs(calc, atoms, third, directory)


def test_links_survive_relocation_and_tar_roundtrip(calculation, tmp_path):
    directory, atoms, calc, driver, _ = calculation
    anchor = driver._save_checkpoint()
    prepare_restart_inputs(calc, atoms, anchor, directory)
    shutil.copyfile(directory / "POSCAR", directory / "CONTCAR")
    second = driver._save_checkpoint()
    prepare_restart_inputs(calc, atoms, second, directory)
    moved = tmp_path / "moved"
    directory.rename(moved)
    with tarfile.open(tmp_path / "calc.tar", "w") as archive:
        archive.add(moved, arcname="calc")
    destination = tmp_path / "extracted"
    destination.mkdir()
    with tarfile.open(tmp_path / "calc.tar") as archive:
        archive.extractall(destination, filter="data")
    for root in (moved, destination / "calc"):
        for name in SHARED_INPUTS:
            assert (root / name).read_bytes() == (root / "0000.run" / name).read_bytes()
            assert (root / "0001.run" / name).read_bytes() == (root / "0000.run" / name).read_bytes()
            assert not os.path.isabs(os.readlink(root / name))


def test_missing_anchor_fails_without_rewriting(calculation):
    directory, atoms, calc, driver, _ = calculation
    anchor = driver._save_checkpoint()
    (anchor / "POTCAR").unlink()
    with pytest.raises(ValueError, match="0000.run/POTCAR"):
        prepare_restart_inputs(calc, atoms, anchor, directory)
    assert not (directory / "POTCAR").exists()


def test_fresh_write_detaches_links_and_preserves_targets(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    active = tmp_path / "active"
    active.mkdir()
    for name in ("INCAR", "POSCAR", *SHARED_INPUTS):
        (shared / name).write_text("original")
        (active / name).symlink_to(f"../shared/{name}")
    detach_input_links(active)
    for name in ("INCAR", "POSCAR", *SHARED_INPUTS):
        (active / name).write_text("new")
        assert (shared / name).read_text() == "original"


def test_kspacing_restart_has_no_kpoints(calculation):
    directory, atoms, calc, driver, _ = calculation
    calc.set(kspacing=0.3)
    (directory / "KPOINTS").unlink()
    calc.write_input(atoms)
    anchor = driver._save_checkpoint()
    prepare_restart_inputs(calc, atoms, anchor, directory)
    assert not (directory / "KPOINTS").exists()
    assert (directory / "POTCAR").is_symlink()


def test_fresh_run_refuses_existing_checkpoint_history(calculation):
    _, atoms, _, driver, _ = calculation
    driver._save_checkpoint()
    with pytest.raises(ValueError, match="new calculation directory"):
        driver._irun(atoms)
