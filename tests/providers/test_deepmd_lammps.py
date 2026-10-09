import pytest
from ase.build import bulk


def test_deepmd_materializes_for_lammps_without_importing_deepmd_runtime(tmp_path):
    from gdpx.providers.targets import LammpsPotentialMaterialization
    from gdpx.providers import get_provider_manager
    from gdpx.providers.deepmd import DeepMDPotential

    model = tmp_path / "model.pb"
    model.write_bytes(b"fixture")
    runtime = get_provider_manager().resolve_runtime(
        {
            "schema_version": 4,
            "potential": {
                "provider": "deepmd",
                "parameters": {"models": [str(model)], "type_list": ["Cu"]},
            },
            "executor": {
                "provider": "lammps",
                "method": "min",
                "parameters": {"steps": 1},
            },
        }
    )

    assert isinstance(runtime.provider_potential, DeepMDPotential)
    assert isinstance(runtime.materialization, LammpsPotentialMaterialization)
    assert "pair_style deepmd" in runtime.materialization.commands[0]
    assert runtime.executor.setting.task == "min"


def test_lammps_provider_advertises_single_point_execution():
    from gdpx.providers import CapabilityKind, get_provider_manager

    assert get_provider_manager().supports("lammps", CapabilityKind.EXECUTOR, "spc")


@pytest.mark.parametrize(
    "suffix,atom_modify,expected",
    [(".pb", None, None), (".pt2", None, "map yes"), (".pt2", "map hash", "map hash")],
)
@pytest.mark.parametrize("restart", [False, True])
def test_deepmd_pt2_input_enables_atom_map_before_reading_atoms(tmp_path, suffix, atom_modify, expected, restart):
    from gdpx.providers import get_provider_manager

    model = tmp_path / f"model{suffix}"
    model.write_bytes(b"fixture")
    parameters = {"model": str(model), "type_list": ["Cu"]}
    if atom_modify is not None:
        parameters["atom_modify"] = atom_modify
    runtime = get_provider_manager().resolve_runtime(
        {
            "potential": {"provider": "deepmd", "parameters": parameters},
            "executor": {"provider": "lammps", "method": "spc"},
        }
    )
    calc = runtime.materialization.calculator
    calc.type_list = ["Cu"]
    calc.directory = str(tmp_path / "calculation")
    if restart:
        calc.set(read_restart="restart.100.data")
    calc.write_input(bulk("Cu"))
    text = (tmp_path / "calculation" / "in.lammps").read_text()

    if expected is None:
        assert "atom_modify" not in text
    else:
        command = f"atom_modify {expected}"
        assert command in text
        assert text.index(command) < text.index("read_restart" if restart else "read_data")
