# Potentials

The guides below describe the available potential configurations.
An interface listed here reflects the provider registry; individual guides
identify legacy or experimental implementations that are not yet runnable.
Allegro is a flavour of `nequip`.

Each page describes the `potential` configuration and its model parameters.
Replace placeholder model and input paths with your files. The potential
component’s `method` defaults to `default`; calculator-specific methods such
as `GFN2-xTB` belong inside `potential.parameters`.

See {doc}`../computations/index` for assembling a runtime, selecting an executor,
and configuring a calculation.

```{toctree}
:maxdepth: 1
:titlesonly:

emt
eam
reax
deepmd
mace
mattersim
tace
nequip
reann
fairchem
gp
nnp
xtb
vasp
cp2k
abacus
espresso
lasp
dftd3
dftd4
bias
plumed
```

