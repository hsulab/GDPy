"""JAX-MD execution provider."""

from ..adapters import ExecutorFactory
from ..capabilities import CapabilityKind
from ..provider import Provider

JAX_PROVIDER = Provider(
    "jax", "2",
    {CapabilityKind.EXECUTOR: {
        "md": ExecutorFactory("gdpx.providers.jax.driver", "JarexDriver", "md", "ase.calculator")
    }},
)

__all__ = ["JAX_PROVIDER"]
