(bh-tio2-water-rotation)=

# water on titania: molecular orientations

## system

Search orientations of two neighboring waters on anatase (101). The system
contains a 192-atom slab and two waters, with 24 Å of vacuum and 48 fixed bottom
atoms. Each water has a shared positive tag; substrate atoms have tag 0.
A three-water input uses the same slab and adds a third neighboring molecule.

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/tio2_water2.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

## runtime

- {doc}`ReaxFF / xreac <../../../potentials/reax>` ({download}`xreac_tioh.yaml <../../../../../examples/global_optimisation/runtimes/xreac_tioh.yaml>`): minimisation with
  `fmax: 0.02` eV/Å and `steps: 500`. Fixed atoms: zbot 48.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

## methods

- {doc}`basin hopping <../../basin_hopping>`: 1 generation after initialization, with 2 chains of
  4 attempted steps using `rotate` at 500 K acceptance temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/basin_hopping/tio2_water2.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `basin_hopping` | `tio2_water2.yaml` | `../../runtimes/xreac_tioh.yaml` | `tmp/tio2-water-rotation-bh` |

Use `tio2_water3.yaml` for the three-water variant, with the same runtime.

## results

Inspect `candidates.db` for evaluated minima and `results/history/gen0001.png`
for chain decisions. In the recorded short test, both two-water seeds relaxed
to parallel-like waters without a water-to-water hydrogen bond. The three-water
variant also did not recover a hydrogen-bonded minimum. This does not rule out
other minima with different seeds, sites, or potentials.

Use `examples/global_optimisation/tio2_water_rotation.py` from the repository
root for independent seed relaxation and geometric diagnostics.
