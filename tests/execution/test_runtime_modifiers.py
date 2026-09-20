import pytest

from gdpx.execution.resolver import RuntimeResolver
from gdpx.providers import CapabilityKind, Provider, ProviderManager
from gdpx.providers import MaterializationError
from gdpx.providers.targets import AseCalculatorMaterialization, LammpsPotentialMaterialization


class Factory:
    def __init__(self, value):
        self.value = value

    def create(self, parameters, **context):
        return self.value


class Materializer:
    def __init__(self, value):
        self.value = value

    def materialize(self, potential, target=None, **context):
        return self.value


class ExecutorFactory:
    def __init__(self, target):
        self.target = target

    def create(self, parameters, **context):
        return context["materialization"]


def runtime_config(target):
    return {
        "schema_version": 3,
        "potential": {"provider": "model", "parameters": {}},
        "modifiers": [{"provider": "mods", "method": "bias", "parameters": {}}],
        "executor": {"provider": "engine", "method": "md", "parameters": {}},
    }


def test_ase_modifier_is_resolved_and_composed():
    from ase.calculators.emt import EMT

    base, bias = EMT(), EMT()
    providers = ProviderManager()
    providers.register(Provider("model", capabilities={
        CapabilityKind.POTENTIAL: {"default": Factory(object())},
        CapabilityKind.MATERIALIZER: {"ase.calculator": Materializer(AseCalculatorMaterialization(base))},
    }))
    providers.register(Provider("mods", capabilities={CapabilityKind.MODIFIER: {"bias": Factory(bias)}}))
    providers.register(Provider("engine", capabilities={CapabilityKind.EXECUTOR: {"md": ExecutorFactory("ase.calculator")}}))

    runtime = RuntimeResolver(providers).resolve(runtime_config("ase.calculator"))

    assert runtime.modifier_instances == (bias,)
    assert runtime.materialization.calculator.mixer.calcs == [base, bias]


def test_modifier_is_not_silently_ignored_for_unsupported_target():
    providers = ProviderManager()
    providers.register(Provider("model", capabilities={
        CapabilityKind.POTENTIAL: {"default": Factory(object())},
        CapabilityKind.MATERIALIZER: {"lammps.potential": Materializer(LammpsPotentialMaterialization(()))},
    }))
    providers.register(Provider("mods", capabilities={CapabilityKind.MODIFIER: {"bias": Factory(object())}}))
    providers.register(Provider("engine", capabilities={CapabilityKind.EXECUTOR: {"md": ExecutorFactory("lammps.potential")}}))

    with pytest.raises(MaterializationError, match="Modifiers are not supported"):
        RuntimeResolver(providers).resolve(runtime_config("lammps.potential"))


def test_incompatible_potential_and_executor_has_typed_error():
    providers = ProviderManager()
    providers.register(Provider("model", capabilities={
        CapabilityKind.POTENTIAL: {"default": Factory(object())},
    }))
    providers.register(Provider("engine", capabilities={
        CapabilityKind.EXECUTOR: {"md": ExecutorFactory("missing.target")},
    }))
    config = runtime_config("missing.target")
    config["modifiers"] = []

    with pytest.raises(MaterializationError, match="cannot materialize target"):
        RuntimeResolver(providers).resolve(config)
