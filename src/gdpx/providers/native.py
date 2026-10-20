"""GDPy-owned composite sampling executors."""

from dataclasses import dataclass

from .capabilities import CapabilityKind
from .provider import Provider


@dataclass(frozen=True)
class NativeExecutorFactory:
    method: str

    def resolve_composite(self, config, resolver):
        from gdpx.execution.monte_carlo.factory import resolve_sampling_runtime

        return resolve_sampling_runtime(config, resolver)


NATIVE_PROVIDER = Provider(
    name="native",
    version="1",
    capabilities={
        CapabilityKind.EXECUTOR: {
            name: NativeExecutorFactory(name)
            for name in ("monte_carlo", "biased_monte_carlo", "hybrid_monte_carlo")
        },
    },
)
