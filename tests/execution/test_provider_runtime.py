import copy

from ase.calculators.emt import EMT

from gdpx.execution import Runtime
from gdpx.providers import CapabilityKind, Provider, ProviderManager


class PotentialFactory:
    def create(self, parameters, **context):
        class Potential:
            calc = EMT()

        return Potential()


class ExecutorFactory:
    def create(self, parameters, **context):
        class Executor:
            def run(self, inputs, **kwargs):
                return (inputs, parameters["value"])

        return Executor()


def test_runtime_resolves_from_provider_capabilities_without_mutating_config():
    manager = ProviderManager()
    manager.register(Provider("model", capabilities={CapabilityKind.POTENTIAL: {"default": PotentialFactory()}}))
    manager.register(Provider("engine", capabilities={CapabilityKind.EXECUTOR: {"evaluate": ExecutorFactory()}}))
    source = {
        "schema_version": 2,
        "potential": {"provider": "model", "parameters": {}},
        "executor": {"provider": "engine", "method": "evaluate", "parameters": {"value": 7}},
    }
    original = copy.deepcopy(source)

    runtime = manager.resolve_runtime(source)

    assert isinstance(runtime, Runtime)
    assert runtime.run("atoms") == ("atoms", 7)
    assert source == original


def test_default_manager_exposes_existing_integrations_lazily():
    from gdpx.providers import get_provider_manager

    manager = get_provider_manager()
    assert manager.require("emt", CapabilityKind.POTENTIAL, "default")
    assert manager.require("ase", CapabilityKind.EXECUTOR, "min")


def test_runtime_uses_selected_potential_method():
    chosen = 17
    manager = ProviderManager()
    manager.register(Provider("model", capabilities={CapabilityKind.POTENTIAL: {"chosen": PotentialFactory()}}))
    manager.register(Provider("engine", capabilities={CapabilityKind.EXECUTOR: {"evaluate": ExecutorFactory()}}))

    runtime = manager.resolve_runtime(
        {
            "schema_version": 2,
            "potential": {"provider": "model", "method": "chosen", "parameters": {}},
            "executor": {"provider": "engine", "method": "evaluate", "parameters": {"value": chosen}},
        }
    )

    assert runtime.potential.method == "chosen"
    assert runtime.run("atoms") == ("atoms", chosen)
