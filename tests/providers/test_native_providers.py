from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.abacus import AbacusExecutorFactory, AbacusPotentialFactory
from gdpx.providers.lasp import LaspExecutorFactory, LaspPotentialFactory


def test_native_providers_advertise_only_supported_methods():
    providers = get_provider_manager()
    assert isinstance(providers.require("abacus", CapabilityKind.POTENTIAL, "default"), AbacusPotentialFactory)
    assert isinstance(providers.require("abacus", CapabilityKind.EXECUTOR, "scf"), AbacusExecutorFactory)
    assert not providers.supports("abacus", CapabilityKind.EXECUTOR, "freq")
    assert isinstance(providers.require("lasp", CapabilityKind.POTENTIAL, "default"), LaspPotentialFactory)
    assert isinstance(providers.require("lasp", CapabilityKind.EXECUTOR, "md"), LaspExecutorFactory)
    assert not providers.supports("lasp", CapabilityKind.EXECUTOR, "freq")
