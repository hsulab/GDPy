from gdpx.providers import get_provider_manager
from gdpx.providers.targets import AseCalculatorMaterialization


def test_ase_potential_and_executor_resolve_without_manager():
    runtime = get_provider_manager().resolve_runtime(
        {
            "schema_version": 4,
            "potential": {"provider": "ase", "parameters": {"method": "lj", "epsilon": 0.5}},
            "executor": {"provider": "ase", "method": "spc", "parameters": {}},
        }
    )

    assert isinstance(runtime.materialization, AseCalculatorMaterialization)
    assert runtime.materialization.calculator.parameters["epsilon"] == 0.5

