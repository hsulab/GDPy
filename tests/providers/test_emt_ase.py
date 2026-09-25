from ase import Atoms

from gdpx.providers.targets import AseCalculatorMaterialization
import pytest

from gdpx.providers import CapabilityKind, ProviderConfigurationError, get_provider_manager
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
