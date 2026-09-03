from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.adapters import ProviderPotential


def test_provider_adapters_create_backend_neutral_potentials():
    providers = get_provider_manager()
    factory = providers.require("eam", CapabilityKind.POTENTIAL, "default")

    potential = factory.create({"backend": "lammps", "model": "unused.eam"})

    assert isinstance(potential, ProviderPotential)
    assert "backend" not in potential.parameters
    assert potential.parameters["model"] == "unused.eam"


def test_software_provider_owns_trainer_and_materializer():
    providers = get_provider_manager()
    assert providers.require("mace", CapabilityKind.TRAINER, "default") is not None
    assert providers.require("mace", CapabilityKind.MATERIALIZER, "ase.calculator") is not None
    assert providers.require("mace", CapabilityKind.DATASET_CODEC, "default") is not None


def test_no_global_legacy_factory_layer_remains():
    import gdpx.providers as providers

    assert not hasattr(providers, "LegacyPotentialFactory")
    assert not hasattr(providers, "LegacyExecutorFactory")
    assert not hasattr(providers, "ManagedPotentialMaterializer")
