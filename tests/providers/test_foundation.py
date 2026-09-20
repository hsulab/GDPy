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


def test_provider_manager_discovers_entry_points_lazily(monkeypatch):
    loaded = []

    class EntryPoint:
        name = "external"

        def load(self):
            loaded.append(True)
            return lambda: Provider("external")

    class EntryPoints(list):
        def select(self, *, group):
            assert group == "gdpx.providers"
            return self

    monkeypatch.setattr(
        "gdpx.providers.manager.importlib.metadata.entry_points",
        lambda: EntryPoints([EntryPoint()]),
    )
    manager = ProviderManager()
    manager.discover()

    assert loaded == []
    assert manager.get("external").name == "external"
    assert loaded == [True]


def test_training_parameters_are_thawed_before_factory_use():
    received = []

    class TrainerFactory:
        def create(self, parameters, **context):
            parameters["nested"]["epochs"] = 2
            received.append(parameters)
            return object()

    manager = ProviderManager()
    manager.register(
        Provider("demo", capabilities={CapabilityKind.TRAINER: {"default": TrainerFactory()}})
    )
    manager.create_training(
        {"provider": "demo", "parameters": {"nested": {"epochs": 1}}}
    )

    assert received == [{"nested": {"epochs": 2}}]


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
        "schema_version": 3,
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


def test_schema_v3_is_immutable_and_round_trips():
    source = {
        "schema_version": 3,
        "potential": {"provider": "deepmd", "parameters": {"models": ["m.pb"]}},
        "modifiers": [],
        "executor": {"provider": "lammps", "method": "md", "parameters": {"steps": 10}},
        "scheduler": {
            "provider": "direct",
            "parameters": {},
            "transport": {"provider": "local", "parameters": {}},
        },
        "options": {"batch_size": 2},
    }
    original = copy.deepcopy(source)
    config = RuntimeConfig.from_mapping(source)

    source["potential"]["parameters"]["models"].append("changed.pb")

    assert config.potential.parameters["models"] == ("m.pb",)
    assert original == config.to_dict()


def test_schema_v3_scheduler_defaults_remain_optional():
    source = {
        "schema_version": 3,
        "potential": {"provider": "models", "parameters": {}},
        "executor": {"provider": "engine", "method": "md", "parameters": {}},
    }
    config = RuntimeConfig.from_mapping(source)

    assert config.scheduler is None
    assert "scheduler" not in config.to_dict()

    with_scheduler = dict(source)
    with_scheduler["scheduler"] = {"provider": "direct", "parameters": {}}
    config = RuntimeConfig.from_mapping(with_scheduler)
    assert config.scheduler.transport is None
    assert config.to_dict()["scheduler"] == with_scheduler["scheduler"]


def test_schema_v2_is_rejected():
    with pytest.raises(ProviderConfigurationError, match="schema_version: 3"):
        RuntimeConfig.from_mapping({"schema_version": 2})


@pytest.mark.parametrize("transport", ["ssh", {"provider": "ssh", "parameters": []}])
def test_malformed_transport_is_rejected(transport):
    from gdpx.providers.configuration import scheduler_component

    with pytest.raises(ProviderConfigurationError, match="must be a mapping"):
        scheduler_component({"provider": "direct", "transport": transport})


@pytest.mark.parametrize(
    ("provider", "message"),
    [("local", "use `direct`"), ("remote", "transport.provider: ssh")],
)
def test_schema_v3_rejects_old_scheduler_provider_names(provider, message):
    source = {
        "schema_version": 3,
        "potential": {"provider": "models", "parameters": {}},
        "executor": {"provider": "engine", "method": "md", "parameters": {}},
        "scheduler": {"provider": provider, "parameters": {}},
    }

    with pytest.raises(ProviderConfigurationError, match=message):
        RuntimeConfig.from_mapping(source)


def test_legacy_runtime_is_rejected_without_mutating_input():
    source = {
        "potter": {"name": "deepmd", "params": {"backend": "ase", "model": "m.pb"}},
        "driver": {"backend": "external", "task": "md", "steps": 20},
        "scheduler": {"backend": "local", "cores": 2},
        "batchsize": 4,
    }
    original = copy.deepcopy(source)

    with pytest.raises(ProviderConfigurationError, match="Legacy runtime fields are not supported: driver, potter"):
        RuntimeConfig.from_mapping(source)

    assert source == original


def test_legacy_dimer_controller_is_rejected():
    with pytest.raises(ProviderConfigurationError, match="Legacy runtime fields are not supported: driver"):
        RuntimeConfig.from_mapping(
            {
                "potential": {"name": "cp2k", "params": {"backend": "cp2k"}},
                "driver": {"task": "ts", "controller": {"name": "dimer_ts"}},
            }
        )

def test_omitted_schema_uses_current_version_and_serializes_explicitly():
    from gdpx.providers import SCHEMA_VERSION

    source = {
        "potential": {"provider": "emt"},
        "executor": {"provider": "ase", "method": "min"},
    }
    original = copy.deepcopy(source)
    config = RuntimeConfig.from_mapping(source)
    assert config.schema_version == SCHEMA_VERSION
    assert config.to_dict()["schema_version"] == SCHEMA_VERSION
    assert source == original
    assert RuntimeConfig.from_mapping(config.to_dict()) == config


@pytest.mark.parametrize("version", [None, 1, 2, 4, "3"])
def test_explicit_unsupported_schema_is_rejected(version):
    with pytest.raises(ProviderConfigurationError, match="schema_version: 3"):
        RuntimeConfig.from_mapping({"schema_version": version})


def test_builtin_potentials_are_exposed_through_provider_manager():
    from gdpx.providers import get_provider_manager

    manager = get_provider_manager()
    for name in ("emt", "deepmd", "mace", "nequip", "reann"):
        assert manager.require(name, CapabilityKind.POTENTIAL, "default")
