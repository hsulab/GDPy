"""Potential-manager registry with lazy implementation loading."""

from gdpx.core.registry import Registry


REGISTER = Registry("manager")

_IMPLEMENTATIONS = {
    "deepmd": ("gdpx.potential.deepmd", "DeepmdManager"),
    "deepmd_jax": ("gdpx.potential.deepmd", "DeepmdJaxManager"),
    "deepmd_jax_x": ("gdpx.potential.deepmd", "DeepmdJaxXManager"),
    "beann": ("gdpx.potential.reann.beann", "BeannManager"),
    "reann": ("gdpx.potential.reann.reann", "ReannManager"),
    "lasp": ("gdpx.potential.lasp", "LaspManager"),
    "mace": ("gdpx.potential.mace", "MaceManager"),
    "nequip": ("gdpx.potential.nequip", "NequipManager"),
    "mattersim": ("gdpx.potential.mattersim", "MatterSimManager"),
    "tace": ("gdpx.potential.tace", "TaceManager"),
    "fairchem": ("gdpx.potential.fairchem", "FairChemManager"),
    "nnp": ("gdpx.potential.nnp.manager", "NnAcsfManager"),
    "cp2k": ("gdpx.potential.cp2k", "Cp2kManager"),
    "espresso": ("gdpx.potential.espresso", "EspressoManager"),
    "vasp": ("gdpx.potential.vasp", "VaspManager"),
    "ase": ("gdpx.potential.asepot", "AsePotManager"),
    "classic": ("gdpx.potential.classic", "ClassicManager"),
    "eam": ("gdpx.potential.eam", "EamManager"),
    "emt": ("gdpx.potential.emt", "EmtManager"),
    "reax": ("gdpx.potential.reax", "ReaxManager"),
    "gp": ("gdpx.potential.gp", "GaussianProcessManager"),
    "grid": ("gdpx.potential.grid", "GridManager"),
    "mixer": ("gdpx.potential.mixer", "MixerManager"),
    "abacus": ("gdpx.potential.abacus", "AbacusManager"),
    "xtb": ("gdpx.potential.xtb", "XtbManager"),
    "dftd3": ("gdpx.potential.dftd3", "Dftd3Manager"),
    "dftd4": ("gdpx.potential.dftd4", "Dftd4Manager"),
    "bias": ("gdpx.potential.bias", "BiasManager"),
    "plumed": ("gdpx.potential.plumed.plumed", "PlumedManager"),
}

for _name, (_module, _attribute) in _IMPLEMENTATIONS.items():
    REGISTER.register_lazy(_name, _module, _attribute)

del _name, _module, _attribute
