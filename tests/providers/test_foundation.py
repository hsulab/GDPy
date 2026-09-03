import copy

import pytest

from gdpx.providers import (
    AmbiguousCapabilityError,
    CapabilityKind,
    DuplicateProviderError,
    MissingCapabilityError,
    Provider,
    ProviderConfigurationError,
    ProviderManager,
    RuntimeConfig,
    UnknownProviderError,
)


def test_provider_manager_registers_and_resolves_capabilities():
    potential_factory = object()
    provider = Provider(
        "demo",
        "1",
        {CapabilityKind.POTENTIAL: {"default": potential_factory}},
    )
    manager = ProviderManager()
    manager.register(provider)

    assert manager.get("demo") is provider
    assert manager.require("demo", CapabilityKind.POTENTIAL) is potential_factory

    with pytest.raises(DuplicateProviderError):
        manager.register(provider)
    with pytest.raises(MissingCapabilityError):
        manager.require("demo", CapabilityKind.TRAINER)
    with pytest.raises(UnknownProviderError):
        manager.get("missing")


def test_provider_manager_loads_lazy_provider_once():
    calls = []
    manager = ProviderManager()

    def load():
        calls.append(True)
        return Provider("lazy")

    manager.register_lazy("lazy", load)
    assert manager.get("lazy").name == "lazy"
    assert manager.get("lazy").name == "lazy"
    assert len(calls) == 1


def test_provider_manager_extends_distinct_capabilities():
    manager = ProviderManager()
    potential = object()
    trainer = object()
    manager.register(Provider("demo", capabilities={CapabilityKind.POTENTIAL: {"default": potential}}))
    merged = manager.extend(Provider("demo", capabilities={CapabilityKind.TRAINER: {"default": trainer}}))

    assert merged.implementations(CapabilityKind.POTENTIAL)["default"] is potential
    assert merged.implementations(CapabilityKind.TRAINER)["default"] is trainer

    with pytest.raises(DuplicateProviderError):
        manager.extend(Provider("demo", capabilities={CapabilityKind.TRAINER: {"default": object()}}))


def test_provider_manager_introspection_does_not_create_components():
    factory = object()
    manager = ProviderManager()
    manager.register(Provider("demo", capabilities={CapabilityKind.EXECUTOR: {"md": factory}}))

    assert manager.supports("demo", CapabilityKind.EXECUTOR)
    assert manager.supports("demo", CapabilityKind.EXECUTOR, "md")
    assert not manager.supports("demo", CapabilityKind.EXECUTOR, "min")
    assert not manager.supports("missing", CapabilityKind.EXECUTOR)
    assert manager.list_capabilities("demo") == {CapabilityKind.EXECUTOR: ("md",)}


def test_potential_method_round_trips_and_selects_factory():
    source = {
        "schema_version": 2,
        "potential": {"provider": "models", "method": "small", "parameters": {}},
        "modifiers": [],
        "executor": {"provider": "engine", "method": "md", "parameters": {}},
        "options": {},
    }
    config = RuntimeConfig.from_mapping(source)

    assert config.potential_spec().method == "small"
    assert config.to_dict() == source


def test_ambiguous_capability_requires_an_implementation_name():
    manager = ProviderManager()
    manager.register(Provider("demo", capabilities={CapabilityKind.EXECUTOR: {"md": object(), "min": object()}}))

    with pytest.raises(AmbiguousCapabilityError):
        manager.require("demo", CapabilityKind.EXECUTOR)


def test_schema_v2_is_immutable_and_round_trips():
    source = {
        "schema_version": 2,
        "potential": {"provider": "deepmd", "parameters": {"models": ["m.pb"]}},
        "modifiers": [],
        "executor": {"provider": "lammps", "method": "md", "parameters": {"steps": 10}},
        "scheduler": {"provider": "local", "parameters": {}},
        "options": {"batch_size": 2},
    }
    original = copy.deepcopy(source)
    config = RuntimeConfig.from_mapping(source)

    source["potential"]["parameters"]["models"].append("changed.pb")

    assert config.potential.parameters["models"] == ("m.pb",)
    assert original == config.to_dict()


def test_legacy_runtime_is_rejected_without_mutating_input():
    source = {
        "potter": {"name": "deepmd", "params": {"backend": "ase", "model": "m.pb"}},
        "driver": {"backend": "external", "task": "md", "steps": 20},
        "scheduler": {"backend": "local", "cores": 2},
        "batchsize": 4,
    }
    original = copy.deepcopy(source)

    with pytest.raises(ProviderConfigurationError, match="Legacy fields found: driver, potter"):
        RuntimeConfig.from_mapping(source)

    assert source == original


def test_legacy_dimer_controller_is_rejected():
    with pytest.raises(ProviderConfigurationError, match="schema_version: 2"):
        RuntimeConfig.from_mapping(
            {
                "potential": {"name": "cp2k", "params": {"backend": "cp2k"}},
                "driver": {"task": "ts", "controller": {"name": "dimer_ts"}},
            }
        )

def test_builtin_potentials_are_exposed_through_provider_manager():
    from gdpx.providers import get_provider_manager

    manager = get_provider_manager()
    for name in ("emt", "deepmd", "mace", "nequip", "reann"):
        assert manager.require(name, CapabilityKind.POTENTIAL, "default")
