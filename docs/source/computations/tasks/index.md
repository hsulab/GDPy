# task examples

These examples separate the calculation task from machine-resource settings.
The first four use ASE’s bundled EMT calculator, so no model download or
external simulator is needed. Run them in the environment where GDPy is installed.

From the repository root, generate the input structures once:

```shell
python examples/compute/tasks/generate.py
```

This creates three Cu dimers, a strained Cu unit cell, a 32-atom Cu supercell,
and relaxed Al/Au surface endpoints for the NEB example. Use a fresh output
directory for each demo. The YAML files are in `examples/compute/tasks/`.

```{toctree}
:maxdepth: 1
:titlesonly:

single-point
relaxation
cell-relaxation
molecular-dynamics
transition-states
vibrations
```
