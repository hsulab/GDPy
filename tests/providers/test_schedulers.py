from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.scheduler.local import LocalScheduler


def test_scheduler_is_resolved_as_a_provider_capability():
    factory = get_provider_manager().require("local", CapabilityKind.SCHEDULER, "default")

    assert isinstance(factory.create({}), LocalScheduler)

