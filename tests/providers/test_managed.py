from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.managed import ManagedPotential


def test_remaining_integrations_create_backend_neutral_potentials():
    providers = get_provider_manager()
    factory = providers.require("eam", CapabilityKind.POTENTIAL, "default")

    potential = factory.create({"backend": "lammps", "model": "unused.eam"})

    assert isinstance(potential, ManagedPotential)
    assert "backend" not in potential.parameters
    assert potential.parameters["model"] == "unused.eam"


def test_managed_provider_retains_legacy_trainer_capability():
    providers = get_provider_manager()
    assert providers.require("mace", CapabilityKind.TRAINER, "default") is not None
    assert providers.require("mace", CapabilityKind.MATERIALIZER, "ase.calculator") is not None


def test_legacy_executor_adapter_materializes_neutral_potential(monkeypatch):
    from gdpx.providers.legacy import LegacyExecutorFactory

    registered = {}

    class Manager:
        def register_calculator(self, parameters):
            registered.update(parameters)

    monkeypatch.setitem(__import__("gdpx.potential", fromlist=["REGISTER"]).REGISTER._dict, "demo", Manager)
    potential = ManagedPotential("demo", {"model": "model.bin"})
    sentinel = object()
    monkeypatch.setattr("gdpx.execution.legacy.create_legacy_executor", lambda manager, params: sentinel)

    result = LegacyExecutorFactory("ase", "md").create({}, potential=potential)

    assert result is sentinel
    assert registered == {"model": "model.bin", "backend": "ase"}
