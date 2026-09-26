from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.vasp import VaspExecutorFactory, VaspPotentialFactory


def test_vasp_provider_exposes_native_and_path_execution():
    manager = get_provider_manager()

    assert isinstance(manager.require("vasp", CapabilityKind.POTENTIAL, "default"), VaspPotentialFactory)
    assert isinstance(manager.require("vasp", CapabilityKind.EXECUTOR, "min"), VaspExecutorFactory)
    assert isinstance(manager.require("vasp", CapabilityKind.EXECUTOR, "neb"), VaspExecutorFactory)


def test_vasp_potential_configuration_is_backend_neutral_and_immutable():
    factory = VaspPotentialFactory()
    source = {"backend": "vasp_interactive", "kpts": [1, 1, 1]}

    potential = factory.create(source)
    source["kpts"].append(2)

    assert not hasattr(potential, "interface")
    assert potential.parameters["kpts"] == (1, 1, 1)
