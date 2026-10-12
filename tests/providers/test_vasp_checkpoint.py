import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp import Vasp

from gdpx.providers.vasp.driver import VaspDriver


class TestVaspCheckpoint(unittest.TestCase):
    def test_without_checkpoint_starts_from_scratch(self):
        for state in ("absent", "empty", "inputs", "empty_outcar", "empty_xml", "invalid_xml"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "cand"
                if state != "absent":
                    directory.mkdir()
                if state == "inputs":
                    (directory / "POSCAR").write_text("unfinished input preparation")
                if state == "empty_outcar":
                    (directory / "OUTCAR").touch()
                if state == "empty_xml":
                    (directory / "vasprun.xml").touch()
                if state == "invalid_xml":
                    (directory / "vasprun.xml").write_text("<modeling>")
                driver = VaspDriver(Vasp(), {"task": "spc"}, directory=directory)
                self.assertFalse(driver._verify_checkpoint())
                atoms = Atoms("H")
                with patch.object(driver, "_irun") as run:
                    driver._run_step(atoms, system_changed=False, read_ckpt=True)
                self.assertTrue(directory.is_dir())
                run.assert_called_once_with(atoms)  # Fresh run without a checkpoint.

    def test_valid_xml_without_outcar_remains_restartable(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "vasprun.xml").write_text("valid XML supplied by patched reader")
            atoms = Atoms("H")
            atoms.calc = SinglePointCalculator(atoms, forces=[[0.0, 0.0, 0.0]])
            driver = VaspDriver(Vasp(), {"task": "spc"}, directory=directory)
            with patch("gdpx.providers.vasp.driver.read", return_value=atoms):
                self.assertTrue(driver._verify_checkpoint())

    def test_bad_geometry_outcar_remains_recognized(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "OUTCAR").write_text(
                "|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----       |\n"
            )
            driver = VaspDriver(Vasp(), {"task": "spc"}, directory=directory)
            self.assertTrue(driver._verify_checkpoint())
