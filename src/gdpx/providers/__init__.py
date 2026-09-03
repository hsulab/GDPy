"""Public provider and integration API."""

from .capabilities import CapabilityKind, Factory, Materializer
from .configuration import ComponentConfig, RuntimeConfig, SCHEMA_VERSION
from .errors import (
    AmbiguousCapabilityError,
    DuplicateProviderError,
    MaterializationError,
    MissingCapabilityError,
    ProviderConfigurationError,
    ProviderError,
    UnknownProviderError,
)
from .manager import ProviderManager, get_provider_manager
from .provider import Provider
from .specs import Artifact, Materialization, ModifierSpec, PotentialSpec, TrainingSpec
from .targets import AseCalculatorMaterialization, LammpsPotentialMaterialization, NativeInputMaterialization

__all__ = [
    "AmbiguousCapabilityError",
    "AseCalculatorMaterialization",
    "Artifact",
    "CapabilityKind",
    "ComponentConfig",
    "DuplicateProviderError",
    "Factory",
    "Materialization",
    "MaterializationError",
    "Materializer",
    "LammpsPotentialMaterialization",
    "MissingCapabilityError",
    "ModifierSpec",
    "NativeInputMaterialization",
    "PotentialSpec",
    "Provider",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderManager",
    "RuntimeConfig",
    "SCHEMA_VERSION",
    "TrainingSpec",
    "UnknownProviderError",
    "get_provider_manager",
]
