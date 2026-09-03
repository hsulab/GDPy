"""Compatibility registry for the deprecated potential-manager API."""

from gdpx.core.registry import Registry


REGISTER = Registry("manager")

_IMPLEMENTATIONS = {
    "deepmd": ("gdpx.providers.deepmd", "DeepmdManager"),
    "deepmd_jax": ("gdpx.providers.deepmd", "DeepmdJaxManager"),
    "deepmd_jax_x": ("gdpx.providers.deepmd", "DeepmdJaxXManager"),
    "beann": ("gdpx.providers.reann", "BeannManager"),
    "reann": ("gdpx.providers.reann", "ReannManager"),
    "lasp": ("gdpx.providers.lasp.manager", "LaspManager"),
    "mace": ("gdpx.providers.mace", "MaceManager"),
    "nequip": ("gdpx.providers.nequip", "NequipManager"),
    "mattersim": ("gdpx.providers.mattersim", "MatterSimManager"),
    "tace": ("gdpx.providers.tace", "TaceManager"),
    "fairchem": ("gdpx.providers.fairchem", "FairChemManager"),
    "nnp": ("gdpx.providers.nnp", "NnAcsfManager"),
    "cp2k": ("gdpx.providers.cp2k.manager", "Cp2kManager"),
    "espresso": ("gdpx.providers.espresso.manager", "EspressoManager"),
    "vasp": ("gdpx.providers.vasp.manager", "VaspManager"),
    "ase": ("gdpx.providers.ase.manager", "AsePotManager"),
    "classic": ("gdpx.providers.classic", "ClassicManager"),
    "eam": ("gdpx.providers.eam", "EamManager"),
    "emt": ("gdpx.providers.emt.manager", "EmtManager"),
    "reax": ("gdpx.providers.reax", "ReaxManager"),
    "gp": ("gdpx.providers.gp", "GaussianProcessManager"),
    "grid": ("gdpx.providers.grid", "GridManager"),
    "mixer": ("gdpx.providers.mixer", "MixerManager"),
    "abacus": ("gdpx.providers.abacus.manager", "AbacusManager"),
    "xtb": ("gdpx.providers.xtb", "XtbManager"),
    "dftd3": ("gdpx.providers.dftd3", "Dftd3Manager"),
    "dftd4": ("gdpx.providers.dftd4", "Dftd4Manager"),
    "bias": ("gdpx.providers.bias", "BiasManager"),
    "plumed": ("gdpx.providers.plumed", "PlumedManager"),
}

for _name, (_module, _attribute) in _IMPLEMENTATIONS.items():
    REGISTER.register_lazy(_name, _module, _attribute)

del _name, _module, _attribute
