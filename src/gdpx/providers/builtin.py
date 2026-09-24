"""Registration of provider implementations migrated to the new API."""


def register_builtin_providers(manager) -> None:
    from .ase import ASE_PROVIDER
    from .emt import EMT_PROVIDER
    from .deepmd import DEEPMD_JAX_PROVIDER, DEEPMD_JAX_X_PROVIDER, DEEPMD_PROVIDER
    from .lammps import LAMMPS_PROVIDER
    from .vasp import VASP_PROVIDER
    from .cp2k import CP2K_PROVIDER
    from .abacus import ABACUS_PROVIDER
    from .lasp import LASP_PROVIDER
    from .espresso import ESPRESSO_PROVIDER
    from .builtin_components import BUILTIN_PROVIDER
    from .jax import JAX_PROVIDER
    from .replica import REPLICA_PROVIDER
    from .bias import BIAS_PROVIDER
    from .classic import CLASSIC_PROVIDER
    from .dftd3 import DFTD3_PROVIDER
    from .dftd4 import DFTD4_PROVIDER
    from .eam import EAM_PROVIDER
    from .fairchem import FAIRCHEM_PROVIDER
    from .gp import GP_PROVIDER
    from .grid import GRID_PROVIDER
    from .mace import MACE_PROVIDER
    from .mattersim import MATTERSIM_PROVIDER
    from .nequip import NEQUIP_PROVIDER
    from .allegro import ALLEGRO_PROVIDER
    from .nnp import NNP_PROVIDER
    from .plumed import PLUMED_PROVIDER
    from .reann import BEANN_PROVIDER, REANN_PROVIDER
    from .reax import REAX_PROVIDER
    from .tace import TACE_PROVIDER
    from .xtb import XTB_PROVIDER

    providers = (
        ASE_PROVIDER, EMT_PROVIDER, DEEPMD_PROVIDER, DEEPMD_JAX_PROVIDER,
        DEEPMD_JAX_X_PROVIDER, LAMMPS_PROVIDER, VASP_PROVIDER, CP2K_PROVIDER,
        ABACUS_PROVIDER, LASP_PROVIDER, ESPRESSO_PROVIDER, BUILTIN_PROVIDER,
        BIAS_PROVIDER, CLASSIC_PROVIDER, DFTD3_PROVIDER, DFTD4_PROVIDER,
        EAM_PROVIDER, FAIRCHEM_PROVIDER, GP_PROVIDER, GRID_PROVIDER,
        MACE_PROVIDER, MATTERSIM_PROVIDER, NEQUIP_PROVIDER, ALLEGRO_PROVIDER,
        NNP_PROVIDER, PLUMED_PROVIDER, BEANN_PROVIDER, REANN_PROVIDER,
        REAX_PROVIDER, TACE_PROVIDER, XTB_PROVIDER, JAX_PROVIDER,
        REPLICA_PROVIDER,
    )
    for provider in providers:
        manager.register(provider, replace=True)
    from .schedulers import scheduler_providers, transport_providers

    for provider in (*scheduler_providers(), *transport_providers()):
        manager.register(provider, replace=True)
