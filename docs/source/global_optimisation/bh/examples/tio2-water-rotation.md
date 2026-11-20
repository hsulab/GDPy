(bh-tio2-water-rotation)=
(bh-tio2-water-bond-change)=

# water on titania

## system

Search orientations and dissociation of neighboring waters on anatase (101).
The 192-atom slab has 24 Å of vacuum and 48 fixed bottom atoms. Inputs with two
or three waters contain 198 or 201 atoms, respectively. Substrate atoms have
tag 0; each water initially shares a positive fragment tag.

```{literalinclude} ../../../../../examples/global_optimisation/basin_hopping/tio2_water2.yaml
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

Both {doc}`basin-hopping <../../basin_hopping>` variants run 1 generation after
initialization, with 2 chains of 4 attempted steps at 500 K acceptance temperature:

- **Molecular orientations:** rotate intact waters about oxygen by up to 75°.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/basin_hopping/tio2_water2.yaml).
- **Dissociation and recombination:** mix `rotate` (50%), `bond_break` (25%), and
  `bond_form` (25%), starting from molecular and dissociated seeds. Fragment
  tags are reconciled after relaxation as bonds change.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/basin_hopping/tio2_water2_bond_change.yaml).

Three-water inputs use the same runtime:
[rotations](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/basin_hopping/tio2_water3.yaml)
and [bond changes](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/basin_hopping/tio2_water3_bond_change.yaml).
Use a separate output directory for each input.

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `basin_hopping` | `tio2_water2.yaml` | `../runtimes/xreac_tioh.yaml` | `tmp/tio2-water-rotation-bh` |
| `basin_hopping` | `tio2_water2_bond_change.yaml` | `../runtimes/xreac_tioh.yaml` | `tmp/tio2-water-bond-change-bh` |

## results

Inspect minima and relaxed connectivity in `candidates.db`, and chain decisions
in `results/history/gen0001.png`. Short rotation tests found parallel-like
waters without a water-to-water hydrogen bond; this does not rule out other minima.

An archived bond-change test retained molecular and dissociated minima, with
the best refined dissociated structure 0.129322 eV higher. One trial needed
further relaxation, so this is a local comparison rather than an exhaustive
stability search.

The molecular and dissociated seed structures are included in
`examples/global_optimisation/assets/`; the commands above use them directly.
