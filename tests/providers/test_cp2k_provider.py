from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.cp2k import Cp2kExecutorFactory, Cp2kPotential, Cp2kPotentialFactory


def test_cp2k_provider_owns_native_and_transition_state_execution():
    providers = get_provider_manager()

    assert isinstance(providers.require("cp2k", CapabilityKind.POTENTIAL, "default"), Cp2kPotentialFactory)
    assert isinstance(providers.require("cp2k", CapabilityKind.EXECUTOR, "dimer"), Cp2kExecutorFactory)
    assert isinstance(providers.require("cp2k", CapabilityKind.EXECUTOR, "neb"), Cp2kExecutorFactory)


def test_cp2k_potential_is_backend_neutral():
    source = {"backend": "cp2k_shell", "cutoff": 400}
    potential = Cp2kPotentialFactory().create(source)

    assert isinstance(potential, Cp2kPotential)
    assert potential.interface == "cp2k_shell"
    assert "backend" not in potential.parameters
    assert source == {"backend": "cp2k_shell", "cutoff": 400}
