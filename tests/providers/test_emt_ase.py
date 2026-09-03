from ase import Atoms

from gdpx.execution.targets import AseCalculatorMaterialization
from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.emt import EmtPotential


def test_emt_ase_vertical_slice_resolves_and_evaluates():
    runtime = get_provider_manager().resolve_runtime(
        {
            "schema_version": 2,
            "potential": {"provider": "emt", "parameters": {}},
            "executor": {"provider": "ase", "method": "spc", "parameters": {}},
        }
    )

    assert isinstance(runtime.provider_potential, EmtPotential)
    assert isinstance(runtime.materialization, AseCalculatorMaterialization)
    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = runtime.materialization.calculator
    assert isinstance(atoms.get_potential_energy(), float)


def test_legacy_emt_schema_uses_new_provider_implementation():
    runtime = get_provider_manager().resolve_runtime(
        {
            "potter": {"name": "emt", "params": {"backend": "ase"}},
            "driver": {"backend": "ase", "task": "min", "steps": 1},
        }
    )

    assert isinstance(runtime.provider_potential, EmtPotential)
    assert runtime.executor.setting.task == "min"


def test_manager_materialize_selects_the_requested_target():
    providers = get_provider_manager()
    potential = providers.require(
        "emt", CapabilityKind.POTENTIAL, "default"
    ).create({})

    materialization = providers.materialize(
        potential, "ase.calculator", provider_name="emt"
    )

    assert materialization.calculator.name == "emt"
