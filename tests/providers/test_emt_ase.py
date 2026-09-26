import copy

from ase import Atoms

from gdpx.providers.targets import AseCalculatorMaterialization
import pytest

from gdpx.providers import CapabilityKind, ProviderConfigurationError, RuntimeConfig, get_provider_manager
from gdpx.providers.emt import EmtPotential


def test_emt_ase_vertical_slice_resolves_and_evaluates():
    runtime = get_provider_manager().resolve_runtime(
        {
            "schema_version": 4,
            "potential": {"provider": "emt", "parameters": {}},
            "executor": {"provider": "ase", "method": "spc", "parameters": {}},
        }
    )

    assert isinstance(runtime.provider_potential, EmtPotential)
    assert isinstance(runtime.materialization, AseCalculatorMaterialization)
    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = runtime.materialization.calculator
    assert isinstance(atoms.get_potential_energy(), float)


def test_nested_executor_parameters_are_thawed_before_factory_use():
    runtime = get_provider_manager().resolve_runtime(
        {
            "schema_version": 4,
            "potential": {"provider": "emt", "parameters": {}},
            "executor": {
                "provider": "ase",
                "method": "md",
                "parameters": {
                    "controller": {"name": "langevin", "params": {"friction": 0.01}},
                    "ensemble": "nvt",
                    "dump_period": 2,
                    "velocity_seed": 1112,
                    "steps": 3,
                },
            },
        }
    )

    assert runtime.executor.setting.dump_period == 2
    assert runtime.executor.setting.steps == 3


def test_structured_md_executor_parameters_are_resolved_and_preserved():
    source = {
        "schema_version": 4,
        "potential": {"provider": "emt", "parameters": {}},
        "executor": {
            "provider": "ase",
            "method": "md",
            "parameters": {
                "random_seed": 19,
                "setup": {
                    "ensemble": "nvt",
                    "timestep": 1.0,
                    "velocities": {"initialize": "always", "seed": 23},
                    "regulator": {
                        "name": "berendsen",
                        "targets": {"temperature": 300},
                        "parameters": {"Tdamp": 100.0},
                    },
                },
                "output": {"trajectory": {"period": 5}},
                "stop": {"steps": 20},
            },
        },
    }

    runtime = get_provider_manager().resolve_runtime(source)

    assert runtime.config.to_dict()["executor"]["parameters"] == source["executor"]["parameters"]
    assert runtime.executor.random_seed == 19
    assert runtime.executor.setting.temp == 300
    assert runtime.executor.setting.velocity_seed == 23
    assert runtime.executor.setting.ignore_atoms_velocities
    assert runtime.executor.setting.dump_period == 5
    assert runtime.executor.setting.steps == 20


def test_structured_executor_parameters_reject_mixed_and_inconsistent_md():
    base = {
        "potential": {"provider": "emt"},
        "executor": {"provider": "ase", "method": "md"},
    }
    mixed = copy.deepcopy(base)
    mixed["executor"]["parameters"] = {"setup": {}, "steps": 10}
    with pytest.raises(ProviderConfigurationError, match="Do not mix flat and structured"):
        RuntimeConfig.from_mapping(mixed)

    invalid = copy.deepcopy(base)
    invalid["executor"]["parameters"] = {
        "setup": {
            "ensemble": "nvt",
            "regulator": {
                "name": "berendsen",
                "targets": {"pressure": 1.0},
            },
        }
    }
    with pytest.raises(ProviderConfigurationError, match="NVT.*temperature"):
        RuntimeConfig.from_mapping(invalid)


def test_legacy_emt_schema_is_rejected():
    with pytest.raises(ProviderConfigurationError, match="Legacy runtime fields are not supported"):
        get_provider_manager().resolve_runtime(
            {
                "potter": {"name": "emt", "params": {"backend": "ase"}},
                "driver": {"backend": "ase", "task": "min", "steps": 1},
            }
        )


def test_manager_materialize_selects_the_requested_target():
    providers = get_provider_manager()
    potential = providers.require(
        "emt", CapabilityKind.POTENTIAL, "default"
    ).create({})

    materialization = providers.materialize(
        potential, "ase.calculator", provider_name="emt"
    )

    assert materialization.calculator.name == "emt"
