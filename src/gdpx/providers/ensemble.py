"""Shared thermodynamic input for MD and Monte Carlo executors."""
import copy
import math
from collections.abc import Mapping

from .errors import ProviderConfigurationError


SUPPORTED_ENSEMBLES = {
    "md": {"nve", "nvt", "npt"},
    "monte_carlo": {"nvt", "uvt", "suvt"},
    "hybrid_monte_carlo": {"nvt", "uvt", "suvt"},
}


def resolve_ensemble(value, method):
    """Validate thermodynamic fields and the executor's supported ensembles."""
    if not isinstance(value, Mapping):
        raise ProviderConfigurationError("parameters.ensemble must be a mapping with method.")
    result = copy.deepcopy(dict(value))
    name = result.get("method")
    if not isinstance(name, str) or name not in {"nve", "nvt", "npt", "uvt", "suvt"}:
        raise ProviderConfigurationError("parameters.ensemble.method must be nve, nvt, npt, uvt, or suvt.")
    if name not in SUPPORTED_ENSEMBLES.get(method, set()):
        raise ProviderConfigurationError(f"{method} does not support the {name} ensemble.")
    required = {"method"}
    if name != "nve":
        required.add("temperature")
    if name == "npt":
        required.add("pressure")
    if name in {"uvt", "suvt"}:
        required.add("chemical_potentials")
    if set(result) != required:
        raise ProviderConfigurationError(
            f"The {name} ensemble requires exactly: {', '.join(sorted(required))}.")
    for key in ("temperature", "pressure"):
        if key in result:
            value = result[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or (key == "temperature" and value <= 0)):
                raise ProviderConfigurationError(f"ensemble.{key} must be finite" +
                                                 (" and positive." if key == "temperature" else "."))
    if "chemical_potentials" in result:
        values = result["chemical_potentials"]
        if not isinstance(values, Mapping) or not values or any(
                not isinstance(key, str) or not key or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)
                for key, value in values.items()):
            raise ProviderConfigurationError("ensemble.chemical_potentials must map particle names to finite values.")
    return result


def inherit_ensemble(definition, ensemble):
    """Give HMC children the applicable part of the parent ensemble."""
    if not isinstance(definition, Mapping):
        raise ProviderConfigurationError("Child executor must be a mapping.")
    child = copy.deepcopy(dict(definition))
    params = child.setdefault("parameters", {})
    if not isinstance(params, Mapping):
        raise ProviderConfigurationError("Child executor parameters must be a mapping.")
    setup = params.get("setup", {})
    if not isinstance(setup, Mapping):
        raise ProviderConfigurationError("HMC child setup must be a mapping.")
    if "ensemble" in params or "ensemble" in setup:
        raise ProviderConfigurationError("HMC children inherit parameters.ensemble; define it only on the parent.")
    if {"tend", "pend"}.intersection(setup) or {"tend", "pend"}.intersection(params):
        raise ProviderConfigurationError("HMC children cannot override the parent ensemble with temperature or pressure ramps.")
    inherited = copy.deepcopy(ensemble)
    if child.get("method") == "md" and ensemble["method"] in {"uvt", "suvt"}:
        inherited = {"method": "nvt", "temperature": ensemble["temperature"]}
    params["ensemble"] = inherited
    return child
