"""Public provider and integration API."""

from .capabilities import CapabilityKind, Factory, Materializer
from .configuration import (
    SCHEMA_VERSION,
    ComponentConfig,
    DispatchConfig,
    ModifierConfig,
    PotentialConfig,
    RuntimeConfig,
    SchedulerConfig,
    expand_runtime_configs,
)
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
    "DispatchConfig",
    "DuplicateProviderError",
    "Factory",
    "Materialization",
    "MaterializationError",
    "Materializer",
    "LammpsPotentialMaterialization",
    "MissingCapabilityError",
    "ModifierSpec",
    "ModifierConfig",
    "NativeInputMaterialization",
    "PotentialSpec",
    "PotentialConfig",
    "Provider",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderManager",
    "RuntimeConfig",
    "SCHEMA_VERSION",
    "SchedulerConfig",
    "TrainingSpec",
    "UnknownProviderError",
    "expand_runtime_configs",
    "get_provider_manager",
]
