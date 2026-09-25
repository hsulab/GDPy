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
