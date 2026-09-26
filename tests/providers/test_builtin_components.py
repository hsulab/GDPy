from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.builtin_components import RegistryComponentFactory


def test_builtin_biases_and_colvars_are_provider_capabilities():
    manager = get_provider_manager()

    modifier = manager.require("builtin", CapabilityKind.MODIFIER, "distance_harmonic")
    colvar = manager.require("builtin", CapabilityKind.COLLECTIVE_VARIABLE, "distance")

    assert isinstance(modifier, RegistryComponentFactory)
    assert isinstance(colvar, RegistryComponentFactory)

