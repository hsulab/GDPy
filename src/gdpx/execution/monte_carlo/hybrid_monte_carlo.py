"""Hybrid MD/MC delegates proposal semantics to its MC children."""
from gdpx.execution.composite import CompositeExecutor


class HybridMonteCarlo(CompositeExecutor):
    pass


class HybridBiasedMonteCarlo(CompositeExecutor):
    """Alternate MD with MC blocks whose proposals undergo minimisation."""
