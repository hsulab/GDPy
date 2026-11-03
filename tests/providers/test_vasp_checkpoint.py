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


    def test_incomplete_outcar_is_unfinished(self):
        from ase.io import ParseError
        from gdpx.providers.vasp import driver as module

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("vasprun.xml", "OUTCAR", "OSZICAR"):
                (directory / name).write_text("interrupted output")
            atoms = Atoms("H")
            driver = VaspDriver(Vasp(), {"task": "spc"}, directory=directory)

            def read_output(source, *args, **kwargs):
                if kwargs.get("format") == "vasp-out":
                    raise ParseError("Incomplete OUTCAR")
                return [atoms]

            with patch.object(module, "read", side_effect=read_output), patch.object(
                module, "read_outcar_scf", return_value={"ispin": 2}
            ):
                self.assertEqual(driver._read_a_single_trajectory(directory), [])
                self.assertFalse(driver.read_convergence())
            self.assertEqual((directory / "OUTCAR").read_text(), "interrupted output")

    def test_other_outcar_parse_errors_are_not_hidden(self):
        from ase.io import ParseError
        from gdpx.providers.vasp import driver as module

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("vasprun.xml", "OUTCAR", "OSZICAR"):
                (directory / name).write_text("output")
            driver = VaspDriver(Vasp(), {"task": "spc"}, directory=directory)
            with patch.object(
                module, "read", side_effect=[[Atoms("H")], ParseError("Unexpected corruption")]
            ), patch.object(module, "read_outcar_scf", return_value={"ispin": 2}):
                with self.assertRaisesRegex(ParseError, "Unexpected corruption"):
                    driver._read_a_single_trajectory(directory)
