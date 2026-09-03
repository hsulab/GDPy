"""Built-in, software-independent modifier and collective-variable capabilities."""

import copy
import importlib
from dataclasses import dataclass
from typing import Any, Mapping

from .capabilities import CapabilityKind
from .provider import Provider


@dataclass(frozen=True)
class RegistryComponentFactory:
    module: str
    registry_key: str

    def create(self, parameters: Mapping[str, Any], **context: Any):
        registry = importlib.import_module(self.module).REGISTER
        implementation = registry[self.registry_key]
        return implementation(**copy.deepcopy(dict(parameters)))


MODIFIERS = {
    name: RegistryComponentFactory("gdpx.bias", name)
    for name in (
        "afir",
        "bondboost",
        "nuclei_repulsion",
        "distance_harmonic",
        "plane_harmonic",
        "bond_gaussian",
        "center_of_mass_gaussian",
        "distance_gaussian",
        "rmsd_gaussian",
    )
}

COLLECTIVE_VARIABLES = {
    "distance": RegistryComponentFactory("gdpx.colvar", "DistanceColvar"),
    "rmsd": RegistryComponentFactory("gdpx.colvar", "RmsdColvar"),
    "fingerprint": RegistryComponentFactory("gdpx.colvar", "FingerprintColvar"),
    "position": RegistryComponentFactory("gdpx.colvar", "position"),
}

BUILTIN_PROVIDER = Provider(
    name="builtin",
    version="1",
    capabilities={
        CapabilityKind.MODIFIER: MODIFIERS,
        CapabilityKind.COLLECTIVE_VARIABLE: COLLECTIVE_VARIABLES,
    },
)

