"""Prepare the strategy-owned MD protocol without mutating user input."""
import copy
import math
from collections.abc import Mapping

from gdpx.providers.configuration import child_executor, resolve_executor_parameters


def simulation_executor(definition, temperature=1.0):
    """Inject a cooling target before ordinary MD validation (1 K for validation only)."""
    if not isinstance(definition, Mapping):
        raise ValueError("SA strategy.simulation must be an MD executor mapping.")
    if set(definition) - {"provider", "method", "parameters"}:
        raise ValueError("strategy.simulation accepts only provider, method, and parameters; runtime settings belong in runtime.")
    executor = copy.deepcopy(dict(definition))
    if executor.get("method") != "md":
        raise ValueError("SA strategy.simulation requires method: md.")
    params = executor.setdefault("parameters", {})
    if not isinstance(params, Mapping):
        raise ValueError("simulation.parameters must be a mapping.")
    setup = params.get("setup", {})
    if not isinstance(setup, Mapping):
        raise ValueError("simulation.parameters.setup must be a mapping.")
    ensemble = setup.get("ensemble", {})
    if not isinstance(ensemble, Mapping) or ensemble.get("method") not in {"nvt", "npt"}:
        raise ValueError("SA requires thermostatted NVT or NPT dynamics.")
    if ("temperature" in ensemble or {"temp", "tend"} & params.keys()
            or {"temp", "tend"} & setup.keys()
            or "temperature" in setup.get("velocities", {})):
        raise ValueError("Remove simulation temperature targets; use strategy.temperatures.")
    ensemble["temperature"] = temperature
    resolved = resolve_executor_parameters(params, "md")
    steps = resolved.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("SA MD stop.steps must be positive.")
    if executor.get("provider") == "ase" and resolved.get("controller", {}).get("name") == "nose_hoover_chain":
        raise ValueError("ASE nose_hoover_chain lacks set_temperature; use langevin or berendsen.")
    return child_executor(executor)


def validate_minimum(config):
    """Validate the shared minimisation runtime without loading providers."""
    if config.executor.method != "min":
        raise ValueError("Population SA requires a minimisation runtime (method: min); move MD into strategy.simulation.")
    params = resolve_executor_parameters(config.executor.parameters, "min")
    for key in ("steps", "fmax"):
        value = params.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
                or (key == "steps" and not isinstance(value, int))):
            raise ValueError("SA minimisation requires positive stop.steps and stop.fmax.")
    if config.modifiers or config.dispatch.share_workdir or config.dispatch.worker != "batch":
        raise ValueError("SA requires an unmodified potential, dispatch.worker: batch, and separate candidate directories.")
