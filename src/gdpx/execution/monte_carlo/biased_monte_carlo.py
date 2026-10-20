"""Biased MC chains evaluated by fixed-cell minimisation."""
import math

from gdpx.providers.configuration import resolve_executor_parameters
from .monte_carlo import MonteCarlo


def validate_minimisation(parameters):
    params = resolve_executor_parameters(parameters, "min")
    steps, fmax = params.get("steps"), params.get("fmax")
    if (isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0
            or isinstance(fmax, bool) or not isinstance(fmax, (int, float))
            or not math.isfinite(fmax) or fmax <= 0):
        raise ValueError("Biased MC minimisation requires positive stop.steps and finite positive stop.fmax.")


class BiasedMonteCarlo(MonteCarlo):
    """Accept or reject relaxed endpoints, optionally filtering proposals."""

    name = "biased_monte_carlo"
    runtime_method_name = "biased_monte_carlo"
    required_evaluation_method = "min"

    def register_worker(self, worker, *args, **kwargs):
        candidate = worker[0] if isinstance(worker, list) and worker else worker
        config = candidate.runtime.config.executor
        if config.method != self.required_evaluation_method:
            raise ValueError("biased_monte_carlo requires runtime.executor.method: min.")
        validate_minimisation(config.parameters)
        super().register_worker(worker, *args, **kwargs)
