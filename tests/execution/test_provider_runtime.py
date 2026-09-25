import copy
import sys
import types

import pytest

from ase.calculators.emt import EMT

from gdpx.execution import Runtime
from gdpx.providers import CapabilityKind, Provider, ProviderManager
from gdpx.providers.schedulers import scheduler_providers, transport_providers


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


@pytest.mark.parametrize("dispatch", ["direct", "slurm", "lsf", "pbs"])
@pytest.mark.parametrize("transport", ["local", "ssh"])
def test_scheduler_and_transport_resolve_independently(monkeypatch, dispatch, transport):
    fake = types.ModuleType("paramiko")
    fake.SSHClient = object
    monkeypatch.setitem(sys.modules, "paramiko", fake)
    monkeypatch.delitem(sys.modules, "gdpx.execution.schedulers.remote", raising=False)
    manager = ProviderManager()
    manager.register(Provider("model", capabilities={CapabilityKind.POTENTIAL: {"default": PotentialFactory()}}))
    manager.register(Provider("engine", capabilities={CapabilityKind.EXECUTOR: {"evaluate": ExecutorFactory()}}))
    for provider in (*scheduler_providers(), *transport_providers()):
        manager.register(provider)
    parameters = {} if transport == "local" else {"hostname": "cluster", "remote_wdir": "/scratch/jobs"}
    runtime = manager.resolve_runtime({
        "schema_version": 4,
        "potential": {"provider": "model"},
        "executor": {"provider": "engine", "method": "evaluate", "parameters": {"value": 7}},
        "scheduler": {
            "provider": dispatch,
            "transport": {"provider": transport, "parameters": parameters},
        },
    })
    assert runtime.scheduler.name == dispatch
    assert runtime.scheduler.transport_name == transport
    assert runtime.scheduler.is_direct == (dispatch == "direct")
    sys.modules.pop("gdpx.execution.schedulers.remote", None)


def test_runtime_resolves_from_provider_capabilities_without_mutating_config():
    manager = ProviderManager()
    manager.register(Provider("model", capabilities={CapabilityKind.POTENTIAL: {"default": PotentialFactory()}}))
    manager.register(Provider("engine", capabilities={CapabilityKind.EXECUTOR: {"evaluate": ExecutorFactory()}}))
    source = {
        "schema_version": 4,
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
            "schema_version": 4,
            "potential": {"provider": "model", "method": "chosen", "parameters": {}},
            "executor": {"provider": "engine", "method": "evaluate", "parameters": {"value": chosen}},
        }
    )

    assert runtime.potential.method == "chosen"
    assert runtime.run("atoms") == ("atoms", chosen)
