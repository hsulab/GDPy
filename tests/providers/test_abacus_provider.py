from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.abacus import AbacusExecutorFactory, AbacusPotentialFactory


def test_abacus_provider_owns_potential_and_native_executor():
    providers = get_provider_manager()
    assert isinstance(providers.require("abacus", CapabilityKind.POTENTIAL, "default"), AbacusPotentialFactory)
    assert isinstance(providers.require("abacus", CapabilityKind.EXECUTOR, "scf"), AbacusExecutorFactory)
    assert not providers.supports("abacus", CapabilityKind.EXECUTOR, "freq")
