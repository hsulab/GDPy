"""Multi-replica execution provider."""

from ..adapters import ExecutorFactory
from ..capabilities import CapabilityKind
from ..provider import Provider

REPLICA_PROVIDER = Provider(
    "replica", "2",
    {CapabilityKind.EXECUTOR: {
        "md": ExecutorFactory(
            "gdpx.providers.replica.driver", "ReplicaDriver", "md", "lammps.potential"
        )
    }},
)

__all__ = ["REPLICA_PROVIDER"]
