from types import SimpleNamespace

from ase import Atoms

from gdpx.providers.lammps.execution.driver import LmpDriver


def _driver(tmp_path, *, task="md", frames=None):
    driver = object.__new__(LmpDriver)
    driver.directory = tmp_path
    driver.setting = SimpleNamespace(task=task, dump_period=1, timestep=1.0)
    driver.calc = SimpleNamespace(units="metal")
    driver._print = lambda message: None
    if frames is not None:
        driver.read_trajectory = lambda: list(frames)
    return driver


def test_lost_atoms_with_partial_md_trajectory_is_accepted(tmp_path):
    driver = _driver(tmp_path, frames=[Atoms("H"), Atoms("H")])
    (tmp_path / "lmp.out").write_text(
        "Step PotEng\n0 -1.0\n1000 -0.5\n"
        "ERROR: Lost atoms: original 2 current 1\n"
        "For more information see https://docs.lammps.org/err0008\n"
    )

    assert driver.read_convergence_from_logfile()
    assert (tmp_path / "EARLYSTOP").read_text().strip() == "lost_atoms"


def test_lost_atoms_without_propagated_trajectory_is_rejected(tmp_path):
    driver = _driver(tmp_path, frames=[Atoms("H")])
    (tmp_path / "lmp.out").write_text("ERROR: Lost atoms: original 2 current 1\n")

    assert not driver.read_convergence_from_logfile()
    assert not (tmp_path / "EARLYSTOP").exists()


def test_other_lammps_error_is_rejected(tmp_path):
    driver = _driver(tmp_path, frames=[Atoms("H"), Atoms("H")])
    (tmp_path / "lmp.out").write_text("ERROR: Invalid pair style\n")

    assert not driver.read_convergence_from_logfile()
    assert not (tmp_path / "EARLYSTOP").exists()


def test_earlystop_reason_is_attached_to_last_frame(tmp_path):
    frames = [Atoms("H"), Atoms("H")]
    driver = _driver(tmp_path)
    driver._aggregate_trajectories = lambda **kwargs: frames
    (tmp_path / "EARLYSTOP").write_text("lost_atoms\n")

    result = driver.read_trajectory()

    assert "earlystop" not in result[0].info
    assert result[-1].info["earlystop"] == "lost_atoms"
