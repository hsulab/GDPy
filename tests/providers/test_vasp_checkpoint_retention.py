import json
import os
import itertools
from pathlib import Path
import shutil
import struct
from unittest.mock import patch

import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import ParseError, write

from gdpx.providers.configuration import validate_electronic_checkpoint
from gdpx.providers.vasp import checkpoint as cp
from gdpx.providers.vasp.driver import VaspDriver


TIMESTAMPS = itertools.count(1000000000000000000, 1000000)


def wavecar(path, tag=45200, endian="<"):
    record = 128
    spins, nk = 2, 2
    nb = 2 if tag in (45200, 45210) else 10
    header_records = ((4 + 3 * nb) * 8 + record - 1) // record
    records = 2 + spins * nk * (header_records + nb)
    with path.open("wb") as stream:
        stream.truncate(records * record)
        stream.write(struct.pack(endian + "3d", record, spins, tag))
        stream.seek(record)
        stream.write(struct.pack(endian + "13d", nk, nb, 520, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0))
        for group in range(spins * nk):
            stream.seek((2 + group * (header_records + nb)) * record)
            stream.write(struct.pack(endian + "4d", 2, 0, 0, 0))
    return record, records


def outputs(directory):
    directory.mkdir(exist_ok=True)
    wavecar(directory / "WAVECAR")
    (directory / "OUTCAR").write_text(
        "General timing and accounting informations for this job:\nElapsed time (sec): 100.0\n"
    )
    (directory / "OSZICAR").write_text(
        "RMM: 6 -10.0 1.0E-7 -2.0E-7 40 0.001\n 1 F= -10.0 E0= -10.0 d E =0\n"
    )
    (directory / "INCAR").write_text("EDIFF = 1.0E-5\nLWAVE = .TRUE.\n")
    write(directory / "CONTCAR", Atoms("H", cell=[5, 5, 5], pbc=True), format="vasp")
    (directory / "CHGCAR").write_text("historical charge payload")
    for name in ("POTCAR", "KPOINTS", "ase-sort.dat"):
        (directory / name).write_text("preserve")
    # Distinct deterministic timestamps ensure each failure reaches its own
    # validation branch, even on filesystems with coarse timestamp updates.
    stamp = next(TIMESTAMPS)
    for name in cp.EVIDENCE:
        os.utime(directory / name, ns=(stamp, stamp))


def certified(directory):
    directory.mkdir(exist_ok=True)
    before = cp.begin_electronic_checkpoint(directory)
    outputs(directory)
    assert cp.finalize_electronic_checkpoint(directory, before, None) == []
    assert cp.certified_checkpoint(directory)


@pytest.mark.parametrize("tag", [45200, 45210, 53300, 53310])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_supported_wavecar_record_formats(tmp_path, tag, endian):
    path = tmp_path / "WAVECAR"
    record, records = wavecar(path, tag, endian)
    assert cp.complete_wavecar(path)
    with path.open("r+b") as stream:
        stream.truncate((records - 1) * record + 1)
    assert not cp.complete_wavecar(path)


@pytest.mark.parametrize("content", [b"", b"fake wave", b"\0" * 256])
def test_nonempty_is_not_sufficient(tmp_path, content):
    path = tmp_path / "WAVECAR"
    path.write_bytes(content)
    assert not cp.complete_wavecar(path)


def test_keep_two_archives_plus_active_and_preserve_other_files(tmp_path):
    for index in range(4):
        certified(tmp_path / f"{index:04d}.run")
    before = cp.begin_electronic_checkpoint(tmp_path)
    outputs(tmp_path)
    removed = cp.finalize_electronic_checkpoint(tmp_path, before, 2)
    assert len(removed) == 4
    for index in range(4):
        archive = tmp_path / f"{index:04d}.run"
        assert (archive / "WAVECAR").exists() == (index >= 2)
        assert (archive / "CHGCAR").exists() == (index >= 2)
        assert all((archive / name).exists() for name in ("INCAR", "CONTCAR", "POTCAR", "KPOINTS", "ase-sort.dat", "OUTCAR"))
    assert cp.certified_checkpoint(tmp_path)
    assert cp.certified_checkpoint(tmp_path / "0002.run")


def test_legacy_and_invalid_archives_do_not_count_as_backups(tmp_path):
    outputs(tmp_path / "0000.run")  # No success certificate.
    certified(tmp_path / "0001.run")
    certified(tmp_path / "0002.run")
    (tmp_path / "0002.run" / "WAVECAR").write_bytes(b"truncated")
    before = cp.begin_electronic_checkpoint(tmp_path)
    outputs(tmp_path)
    assert cp.finalize_electronic_checkpoint(tmp_path, before, 2) == []
    assert (tmp_path / "0000.run" / "WAVECAR").exists()


@pytest.mark.parametrize("failure", ["truncated", "no_footer", "unconverged", "unfinished_scf", "stale_wave", "stale_output", "bad_contcar"])
def test_bad_current_checkpoint_never_prunes(tmp_path, failure):
    for index in range(3):
        certified(tmp_path / f"{index:04d}.run")
    outputs(tmp_path)
    before = cp.begin_electronic_checkpoint(tmp_path)
    outputs(tmp_path)
    if failure == "truncated":
        (tmp_path / "WAVECAR").write_bytes(b"partial")
    elif failure == "no_footer":
        (tmp_path / "OUTCAR").write_text("still running")
    elif failure == "unconverged":
        (tmp_path / "OSZICAR").write_text("RMM: 180 -10 0.1 0.2 10 1\n1 F= -10 E0= -10\n")
    elif failure == "unfinished_scf":
        with (tmp_path / "OSZICAR").open("a") as stream:
            stream.write("DAV: 1 -10 1 2 3 4\n")
    elif failure == "stale_wave":
        before["WAVECAR"] = cp.signature(tmp_path / "WAVECAR")
    elif failure == "stale_output":
        before["OUTCAR"] = cp.signature(tmp_path / "OUTCAR")
    else:
        (tmp_path / "CONTCAR").write_text("corrupted")
    assert cp.finalize_electronic_checkpoint(tmp_path, before, 2) == []
    assert not (tmp_path / cp.MARKER).exists()
    assert all((tmp_path / f"{index:04d}.run" / "WAVECAR").exists() for index in range(3))


def test_certificate_survives_checkpoint_move_but_not_modification(tmp_path):
    active = tmp_path / "active"
    certified(active)
    archive = tmp_path / "0000.run"
    active.rename(archive)
    assert cp.certified_checkpoint(archive)
    (archive / "CONTCAR").write_text("changed")
    assert not cp.certified_checkpoint(archive)


def test_null_keep_disables_deletion(tmp_path):
    for index in range(3):
        certified(tmp_path / f"{index:04d}.run")
    before = cp.begin_electronic_checkpoint(tmp_path)
    outputs(tmp_path)
    assert cp.finalize_electronic_checkpoint(tmp_path, before, None) == []
    assert (tmp_path / "0000.run" / "WAVECAR").exists()


def test_retention_does_not_follow_symlinks(tmp_path):
    old = tmp_path / "0000.run"
    old.mkdir()
    external = tmp_path / "external-wave"
    external.write_bytes(b"external")
    (old / "WAVECAR").symlink_to("../external-wave")
    certified(tmp_path / "0001.run")
    before = cp.begin_electronic_checkpoint(tmp_path)
    outputs(tmp_path)
    cp.finalize_electronic_checkpoint(tmp_path, before, 1)
    assert external.read_bytes() == b"external"
    assert (old / "WAVECAR").is_symlink()


@pytest.mark.parametrize("keep", [0, -1, True, 1.5, "2"])
def test_invalid_keep_is_rejected(keep):
    with pytest.raises(ValueError):
        validate_electronic_checkpoint({"enabled": True, "keep": keep})


def test_driver_failure_never_calls_retention(tmp_path):
    calc = Vasp()
    driver = VaspDriver(calc, {
        "task": "min", "steps": 20, "electronic_checkpoint": {"enabled": True, "keep": 2}
    }, directory=tmp_path)
    with patch.object(calc, "write_input"), patch(
        "gdpx.providers.vasp.driver.run_ase_calculator", side_effect=RuntimeError("Slurm killed job")
    ), patch("gdpx.providers.vasp.driver.finalize_electronic_checkpoint") as finalize:
        with pytest.raises(RuntimeError):
            driver._irun(Atoms("H", cell=[5, 5, 5], pbc=True))
    finalize.assert_not_called()


def test_default_keep_is_two():
    assert validate_electronic_checkpoint({"enabled": True})["keep"] == 2


@pytest.mark.parametrize("error", [AssertionError("velocity block"), ParseError("bad geometry")])
def test_geometry_parser_errors_do_not_prune_or_fail_completed_job(tmp_path, error):
    certified(tmp_path / "0000.run")
    certified(tmp_path / "0001.run")
    before = cp.begin_electronic_checkpoint(tmp_path)
    outputs(tmp_path)
    with patch("ase.io.read", side_effect=error):
        assert cp.finalize_electronic_checkpoint(tmp_path, before, 1) == []
    assert (tmp_path / "0000.run" / "WAVECAR").exists()
    assert not (tmp_path / cp.MARKER).exists()


def test_retention_ignores_symlinked_checkpoint_directories(tmp_path):
    external = tmp_path / "external"
    certified(external)
    active = tmp_path / "active"
    active.mkdir()
    (active / "0000.run").symlink_to("../external", target_is_directory=True)
    certified(active / "0001.run")
    before = cp.begin_electronic_checkpoint(active)
    outputs(active)
    cp.finalize_electronic_checkpoint(active, before, 1)
    assert (external / "WAVECAR").exists()
    assert (external / "CHGCAR").exists()
