import copy
import sys
import types

import pytest

from ase.calculators.emt import EMT

from gdpx.execution import Runtime
from gdpx.providers import (
    CapabilityKind,
    Provider,
    ProviderConfigurationError,
    ProviderManager,
    expand_runtime_configs,
)
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


def _broadcast_runtime():
    return {
        "potential": {"provider": "emt"},
        "executor": {
            "provider": "ase",
            "method": "md",
            "parameters": {
                "ensemble": "nvt",
                "controller": {"name": "berendsen", "params": {}},
            },
            "broadcast": {
                "temp": [300, 600],
                "controller.params.Tdamp": [50.0, 100.0],
            },
        },
    }


def test_executor_broadcast_expands_cartesian_parameters_without_mutation():
    source = _broadcast_runtime()
    original = copy.deepcopy(source)

    configs = expand_runtime_configs(source)

    assert [
        (
            config.executor.parameters["temp"],
            config.executor.parameters["controller"]["params"]["Tdamp"],
        )
        for config in configs
    ] == [(300, 50.0), (300, 100.0), (600, 50.0), (600, 100.0)]
    assert all("broadcast" not in config.executor.to_dict() for config in configs)
    assert source == original


def test_executor_broadcast_supports_parameter_list_indices():
    source = _broadcast_runtime()
    source["executor"]["parameters"]["stages"] = [{"steps": 10}]
    source["executor"]["broadcast"] = {"stages.0.steps": [20, 40]}

    configs = expand_runtime_configs(source)

    assert [config.executor.parameters["stages"][0]["steps"] for config in configs] == [20, 40]


def test_modifier_broadcast_combines_with_executor_broadcast_without_mutation():
    source = _broadcast_runtime()
    source["executor"]["broadcast"] = {"temp": [300, 600]}
    source["modifiers"] = [
        {
            "provider": "builtin",
            "method": "distance_harmonic",
            "parameters": {"group": [0, 1], "kspring": 5.0},
            "broadcast": {"center": [1.0, 1.5]},
        }
    ]
    original = copy.deepcopy(source)

    configs = expand_runtime_configs(source)

    assert [
        (
            config.executor.parameters["temp"],
            config.modifiers[0].parameters["center"],
        )
        for config in configs
    ] == [(300, 1.0), (300, 1.5), (600, 1.0), (600, 1.5)]
    assert all("broadcast" not in config.modifiers[0].to_dict() for config in configs)
    assert source == original


@pytest.mark.parametrize(
    ("broadcast", "message"),
    [
        ({}, "nonempty mapping"),
        ({"temp": []}, "nonempty list"),
        ({"temp": 300}, "nonempty list"),
        ({"missing.temp": [300]}, "Unknown executor broadcast parent"),
        ({"controller": [{}], "controller.params.Tdamp": [50]}, "Overlapping"),
        ({"stages.2.steps": [10]}, "list index"),
    ],
)
def test_invalid_executor_broadcast_is_rejected(broadcast, message):
    source = _broadcast_runtime()
    source["executor"]["parameters"]["stages"] = [{"steps": 10}]
    source["executor"]["broadcast"] = broadcast

    with pytest.raises(ProviderConfigurationError, match=message):
        expand_runtime_configs(source)


@pytest.mark.parametrize(
    ("broadcast", "message"),
    [
        ({}, "nonempty mapping"),
        ({"center": []}, "nonempty list"),
        ({"missing.center": [1.0]}, "Unknown modifier 0 broadcast parent"),
    ],
)
def test_invalid_modifier_broadcast_is_rejected(broadcast, message):
    source = _broadcast_runtime()
    source["executor"].pop("broadcast")
    source["modifiers"] = [
        {
            "provider": "builtin",
            "method": "distance_harmonic",
            "parameters": {"group": [0, 1], "kspring": 5.0},
            "broadcast": broadcast,
        }
    ]

    with pytest.raises(ProviderConfigurationError, match=message):
        expand_runtime_configs(source)
