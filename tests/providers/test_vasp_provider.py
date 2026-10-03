from pathlib import Path

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
