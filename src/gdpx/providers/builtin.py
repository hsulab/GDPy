"""Registration of provider implementations migrated to the new API."""


def register_builtin_providers(manager) -> None:
    from .ase import ASE_PROVIDER
    from .emt import EMT_PROVIDER
    from .deepmd import DEEPMD_PROVIDER
    from .lammps import LAMMPS_PROVIDER
    from .vasp import VASP_PROVIDER
    from .builtin_components import BUILTIN_PROVIDER

    manager.register(ASE_PROVIDER, replace=True)
    manager.register(EMT_PROVIDER, replace=True)
    manager.register(DEEPMD_PROVIDER, replace=True)
    manager.register(LAMMPS_PROVIDER, replace=True)
    manager.register(VASP_PROVIDER, replace=True)
    manager.register(BUILTIN_PROVIDER, replace=True)
    from .managed import managed_provider_fragments

    for provider in managed_provider_fragments():
        # Replace the legacy potential factory while retaining trainer and
        # executor capabilities contributed by the compatibility descriptor.
        current = manager.get(provider.name)
        retained = {
            kind: dict(implementations)
            for kind, implementations in current.capabilities.items()
            if kind not in provider.capabilities
        }
        retained.update(provider.capabilities)
        manager.register(
            type(provider)(provider.name, provider.version, retained),
            replace=True,
        )
    from .schedulers import scheduler_providers

    for provider in scheduler_providers():
        manager.register(provider, replace=True)
