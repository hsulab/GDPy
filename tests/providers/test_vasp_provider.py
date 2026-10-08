from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp

from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.vasp import VaspExecutorFactory, VaspPotentialFactory
from gdpx.providers.vasp.manager import VaspManager


def test_vasp_provider_exposes_native_and_path_execution():
    manager = get_provider_manager()

    assert isinstance(manager.require("vasp", CapabilityKind.POTENTIAL, "default"), VaspPotentialFactory)
    assert isinstance(manager.require("vasp", CapabilityKind.EXECUTOR, "min"), VaspExecutorFactory)
    assert isinstance(manager.require("vasp", CapabilityKind.EXECUTOR, "neb"), VaspExecutorFactory)


def test_vasp_potential_configuration_is_backend_neutral_and_immutable():
    factory = VaspPotentialFactory()
    source = {"backend": "vasp_interactive", "kpts": [1, 1, 1]}

    potential = factory.create(source)
    source["kpts"].append(2)

    assert not hasattr(potential, "interface")
    assert potential.parameters["kpts"] == (1, 1, 1)


def test_remote_vasp_reads_incar_on_execution_host(monkeypatch):
    incar = Path(__file__).parents[1] / "computation" / "vasp" / "assets" / "INCAR"
    monkeypatch.setenv("VASP_PP_PATH", "original-pp-path")
    monkeypatch.setenv("ASE_VASP_VDW", "original-vdw-path")
    manager = VaspManager()

    manager.register_calculator(
        {
            "backend": "vasp",
            "remote": True,
            "command": "vasp_std",
            "incar": str(incar.resolve()),
            "pp_path": str(incar.parent.resolve()),
            "vdw_path": str(incar.parent.resolve()),
        }
    )

    assert manager.calc.float_params["encut"] == 300
    assert manager.calc.int_params["nelm"] == 180


def test_remote_vasp_defers_missing_incar(monkeypatch, tmp_path):
    monkeypatch.setenv("VASP_PP_PATH", "original-pp-path")
    monkeypatch.setenv("ASE_VASP_VDW", "original-vdw-path")
    manager = VaspManager()

    manager.register_calculator(
        {
            "backend": "vasp",
            "remote": True,
            "command": "vasp_std",
            "incar": str(tmp_path / "remote" / "INCAR"),
            "pp_path": str((tmp_path / "remote" / "potpaw").resolve()),
            "vdw_path": str((tmp_path / "remote" / "potpaw").resolve()),
        }
    )

    assert manager.calc.float_params["encut"] is None


@pytest.mark.parametrize("remote", [False, True])
def test_vasp_writes_species_hubbard_u_and_magnetic_initialization(monkeypatch, tmp_path, remote):
    from gdpx.providers.vasp.driver import VaspDriver

    # Start from an INCAR with no U or MAGMOM overrides, as with INCAR_LABEL.
    template = tmp_path / "INCAR_TEMPLATE"
    template.write_text("ISPIN = 2\n")
    monkeypatch.setenv("VASP_PP_PATH", "original-pp-path")
    monkeypatch.setenv("ASE_VASP_VDW", "original-vdw-path")
    manager = VaspManager()
    manager.register_calculator(
        {
            "backend": "vasp",
            "remote": remote,
            "command": "vasp_std",
            "incar": str(template),
            "pp_path": str(tmp_path),
            "vdw_path": str(tmp_path),
            "dft_plus_u": {"Ti": {"L": 2, "U": 3.5, "J": 0.0}},
            "magmom_init": {"Co": 2.0, "Ti": 0.0, "O": 0.0},
        }
    )
    atoms = Atoms(
        "TiCoOTi",
        positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
        cell=[10, 10, 10],
        pbc=True,
    )

    class InputWritten(Exception):
        pass

    # Exercise ASE's real initialization, sorting, and INCAR writer, but
    # skip licensed POTCAR discovery and stop before executing VASP.
    monkeypatch.setattr(manager.calc, "_build_pp_list", lambda *args, **kwargs: [])

    def write_input_only(atoms):
        manager.calc.initialize(atoms)
        manager.calc.write_incar(atoms, directory=tmp_path)
        raise InputWritten

    monkeypatch.setattr(manager.calc, "write_input", write_input_only)
    driver = VaspDriver(manager.calc, {"task": "spc"}, directory=tmp_path)
    with pytest.raises(InputWritten):
        driver._irun(atoms)

    written = Vasp()
    written.read_incar(tmp_path / "INCAR")
    assert written.bool_params["ldau"] is True
    assert written.int_params["ldautype"] == 2
    assert written.int_params["ldauprint"] == 1
    assert written.list_int_params["ldaul"] == [2, -1, -1]
    assert written.list_float_params["ldauu"] == [3.5, 0.0, 0.0]
    assert written.list_float_params["ldauj"] == [0.0, 0.0, 0.0]
    assert written.list_float_params["magmom"] == [0.0, 0.0, 2.0, 0.0]
    assert written.int_params["nsw"] == 0
