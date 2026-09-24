"""Registration and lazy discovery for GDPy providers."""

from __future__ import annotations

import importlib.metadata
import importlib
from collections.abc import Callable, Iterator
from typing import Dict, Mapping, Optional, Tuple, Union

from .capabilities import CapabilityKind
from .errors import (
    AmbiguousCapabilityError,
    DuplicateProviderError,
    MissingCapabilityError,
    UnknownProviderError,
)
from .provider import Provider

ProviderLoader = Callable[[], Provider]


class ProviderManager:
    """Own provider discovery and capability lookup, but no runtime state."""

    entry_point_group = "gdpx.providers"

    def __init__(self) -> None:
        self._providers: Dict[str, Union[Provider, ProviderLoader]] = {}
        self._discovered = False

    def register(self, provider: Provider, *, replace: bool = False) -> Provider:
        if not isinstance(provider, Provider):
            raise TypeError(f"Expected Provider, got {type(provider).__name__}.")
        if provider.name in self._providers and not replace:
            raise DuplicateProviderError(f"Provider {provider.name!r} is already registered.")
        self._providers[provider.name] = provider
        return provider

    def register_lazy(self, name: str, loader: ProviderLoader, *, replace: bool = False) -> None:
        if not name:
            raise ValueError("Provider name cannot be empty.")
        if name in self._providers and not replace:
            raise DuplicateProviderError(f"Provider {name!r} is already registered.")
        self._providers[name] = loader

    def extend(self, provider: Provider) -> Provider:
        """Merge a provider fragment into an existing named provider.

        Integrations commonly contribute their potential, executor, and trainer
        capabilities from separate modules.  Extending preserves those pieces
        while rejecting ambiguous implementation replacements.
        """
        if provider.name not in self._providers:
            return self.register(provider)
        current = self.get(provider.name)
        capabilities = {
            kind: dict(implementations)
            for kind, implementations in current.capabilities.items()
        }
        for kind, implementations in provider.capabilities.items():
            target = capabilities.setdefault(kind, {})
            overlap = set(target).intersection(implementations)
            if overlap:
                names = ", ".join(sorted(overlap))
                raise DuplicateProviderError(
                    f"Provider {provider.name!r} already defines {kind.value!r} implementations: {names}."
                )
            target.update(implementations)
        merged = Provider(provider.name, provider.version, capabilities)
        self._providers[provider.name] = merged
        return merged

    def discover(self) -> None:
        if self._discovered:
            return
        entry_points = importlib.metadata.entry_points()
        selected = (
            entry_points.select(group=self.entry_point_group)
            if hasattr(entry_points, "select")
            else entry_points.get(self.entry_point_group, ())
        )
        for entry_point in selected:
            def load_provider(entry=entry_point):
                exported = entry.load()
                return exported() if callable(exported) else exported

            self.register_lazy(entry_point.name, load_provider)
        self._discovered = True

    def get(self, name: str) -> Provider:
        if name not in self._providers:
            self.discover()
        if name not in self._providers:
            available = ", ".join(sorted(self._providers)) or "none"
            raise UnknownProviderError(f"Unknown provider {name!r}; available providers: {available}.")
        value = self._providers[name]
        if not isinstance(value, Provider):
            loaded = value()
            if not isinstance(loaded, Provider):
                raise TypeError(f"Provider loader {name!r} returned {type(loaded).__name__}, not Provider.")
            if loaded.name != name:
                raise ValueError(f"Provider entry {name!r} loaded provider named {loaded.name!r}.")
            self._providers[name] = loaded
            value = loaded
        return value

    def require(
        self,
        provider_name: str,
        kind: CapabilityKind,
        implementation: Optional[str] = None,
    ) -> object:
        provider = self.get(provider_name)
        implementations = provider.implementations(kind)
        if not implementations:
            available = ", ".join(item.value for item in provider.capabilities) or "none"
            raise MissingCapabilityError(
                f"Provider {provider_name!r} has no {kind.value!r} capability; available capabilities: {available}."
            )
        if implementation is None:
            if len(implementations) != 1:
                choices = ", ".join(sorted(implementations))
                raise AmbiguousCapabilityError(
                    f"Provider {provider_name!r} has multiple {kind.value!r} implementations: {choices}."
                )
            return next(iter(implementations.values()))
        if implementation not in implementations:
            choices = ", ".join(sorted(implementations))
            raise MissingCapabilityError(
                f"Provider {provider_name!r} has no {kind.value!r} implementation {implementation!r}; "
                f"available implementations: {choices}."
            )
        return implementations[implementation]

    def supports(
        self,
        provider_name: str,
        kind: CapabilityKind,
        implementation: Optional[str] = None,
    ) -> bool:
        """Return whether a provider advertises a capability without creating it."""
        try:
            implementations = self.get(provider_name).implementations(kind)
        except UnknownProviderError:
            return False
        if implementation is None:
            return bool(implementations)
        return implementation in implementations

    def list_capabilities(
        self, provider_name: str
    ) -> Mapping[CapabilityKind, Tuple[str, ...]]:
        """Return a stable, read-only description of a provider's implementations."""
        provider = self.get(provider_name)
        return {
            kind: tuple(sorted(implementations))
            for kind, implementations in provider.capabilities.items()
        }

    def resolve_runtime(self, config):
        resolver_module = importlib.import_module("gdpx.execution.resolver")
        return resolver_module.RuntimeResolver(self).resolve(config)

    def materialize(self, potential, target, *, provider_name: str, implementation: str | None = None, **context):
        materializer = self.require(
            provider_name, CapabilityKind.MATERIALIZER, implementation or target
        )
        from .adapters import BackendMaterializer, select_backend
        backend = context.pop("backend", None)
        selected = select_backend(materializer, backend)
        if isinstance(materializer, BackendMaterializer):
            context["backend"] = selected
        return materializer.materialize(potential, target, **context)

    def create_training(self, config):
        from .configuration import ComponentConfig
        from .specs import thaw

        component = config if isinstance(config, ComponentConfig) else ComponentConfig(**config)
        factory = self.require(component.provider, CapabilityKind.TRAINER, component.method or "default")
        return factory.create(thaw(component.parameters))

    def __contains__(self, name: object) -> bool:
        return name in self._providers

    def __iter__(self) -> Iterator[str]:
        return iter(self._providers)


_DEFAULT_MANAGER: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = ProviderManager()
        from .builtin import register_builtin_providers

        register_builtin_providers(_DEFAULT_MANAGER)
    return _DEFAULT_MANAGER
