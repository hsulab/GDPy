(bh-tio2-water-bond-change)=

# water on titania: dissociation and recombination

## system

Search molecular and dissociated water states on anatase (101), starting from
two seeds: intact waters and a dissociated configuration. The two-water system
contains 198 atoms, including 48 fixed slab atoms; a three-water variant is also
available. Fragment tags are reconciled after relaxation as bonds change.

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/tio2_water2_bond_change.yaml
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
  4 attempted steps using `rotate`, `bond_break`, `bond_form` at 500 K acceptance temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/basin_hopping/tio2_water2_bond_change.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `basin_hopping` | `tio2_water2_bond_change.yaml` | `../../runtimes/xreac_tioh.yaml` | `tmp/tio2-water-bond-change-bh` |

Use `tio2_water3_bond_change.yaml` for the three-water variant.

## results

Compare proposed bond changes with relaxed connectivity in `candidates.db`.
The archived two-water test retained molecular and dissociated minima; its
best refined dissociated structure was 0.129322 eV above the best molecular
structure. One trial required additional relaxation, and current seed-generation
rules differ from those used for the archived test. These results are a local
comparison, not an exhaustive stability search.

Use `examples/global_optimisation/tio2_water_bond_change.py` from the repository
root to regenerate and independently relax seeds; `--waters 3` selects the variant.
