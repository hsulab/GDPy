"""Errors raised by the provider and materialization APIs."""


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class UnknownProviderError(ProviderError):
    """A requested provider is not registered."""


class DuplicateProviderError(ProviderError):
    """A provider name was registered more than once."""


class MissingCapabilityError(ProviderError):
    """A provider does not expose a requested capability."""


class AmbiguousCapabilityError(ProviderError):
    """A capability request omitted a required implementation name."""


class MaterializationError(ProviderError):
    """A potential cannot be materialized for an executor target."""


class ProviderConfigurationError(ProviderError, ValueError):
    """Provider or runtime configuration is invalid."""

