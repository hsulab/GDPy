(bh-tio2-water-bond-change)=

# Water dissociation and recombination on TiO₂

Use {doc}`../../../explorations/operators/bond_change` to search molecular and
dissociated water adsorption with xreac's Monti2012 Ti/O/H parameters. The
two-water example has 198 atoms: 192 anatase (101) slab atoms and two neighboring
waters. The bottom 48 atoms are fixed. The three-water variant has 201 atoms
on the same slab. Install GDPy's `reax` extra to use the pinned xreac revision.

## Seeds and independent relaxation

The first seed contains molecular water. The second is generated with an O–H
breaking proposal pointed toward a neighboring surface oxygen. For this site,
`break_distance: 1.5` Å leaves the H near the surface oxygen while breaking its
original water bond; the operator's general default is 2.5 Å.

Both seed slabs receive the same reproducible 0.001 Å perturbation on mobile
slab atoms. Protonation activates torsions involving exactly collinear atoms
in the ideal slab, for which xreac rejects undefined derivatives. Fixed atoms
are unchanged. Connectivity uses O–H and Ti–H cutoffs of 1.25 Å; the Ti–H
override prevents nearby metal–hydrogen contacts from creating false cycles.

```shell
python examples/global_optimisation/tio2_water_bond_change.py \
  --output tmp/tio2-water-bond-change --steps 500 --fmax 0.02
```

The script writes initial and relaxed XYZ files, optimization trajectories,
logs, `summary.json`, and a two-frame `relaxed_pair.xyz`. It reports molecular
water, adsorbed OH, surface H and free H separately, plus Ti–O distances,
energies and maximum mobile force. Optimizer convergence is recorded explicitly;
an unconverged pair exits with a nonzero status. `--build-only` validates seed
geometry without xreac. `--waters 3` selects the three-water variant.
Use `--seed-output PATH.xyz` to regenerate the reusable seed file.

## Mixed basin hopping

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/tio2_water2_bond_change.yaml
:language: yaml
```

Use the existing xreac runtime:

```shell
gdp --runtime examples/global_optimisation/runtimes/xreac_tioh.yaml \
  -d tmp/tio2-water-bond-change-bh \
  explore examples/global_optimisation/explorations/basin_hopping/tio2_water2_bond_change.yaml
```

The search starts two chains with molecular and dissociated seeds and attempts
four hops per chain, mixing oxygen-pivot rotations and bond changes equally.
The rotation operator selects `H2O` and `HO` (Hill notation for hydroxyl).
The bond-change filters allow breaking water or surface O–H bonds and forming
OH + H or surface O + H bonds. Separate break/form operations can change other
bonds during relaxation; the proposed operation and actual final connectivity
are recorded separately in `candidates.db`.

Tags are reconciled after all relaxations. OH and molecular water have their
own positive tags; surface H retains an independent positive tag. Substrate
atoms remain tag 0. Changing the connectivity policy during checkpoint recovery
requires a new search directory. Replace the exploration input with
`tio2_water3_bond_change.yaml` for three waters.

## Independent validation result

With xreac revision `2d0f2bd`, Monti2012 and BFGS at fmax=0.02 eV/Å, both
two-water seeds converged and remained adsorbed:

| Initial state | Relaxed state | Total energy (eV) | Maximum mobile force (eV/Å) |
| --- | --- | --- | --- |
| Molecular | Two intact waters | -1364.707476 | 0.016961 |
| Dissociated | One water, one adsorbed OH, one surface H | -1364.324416 | 0.016163 |

The dissociated minimum is 0.383059 eV higher for the same total composition.
Its surface O–H distance is 0.949 Å and adsorbate O–H distance is 0.940 Å.
All 48 fixed coordinates remained unchanged. This is a comparison of local
minima with the specified potential, not an adsorption-energy calculation or
an exhaustive stability search. Three-water seed geometry and both formation
tests also passed; a three-water optimization was unnecessary because the
two-water case retained a converged dissociated state.

The mixed two-water BH smoke run evaluated eight trials (four breaks and four
rotations), accepting six and rejecting two, with no invalid proposals. Nine
of its ten stored endpoints met fmax=0.02 eV/Å. One trial reached the 500-step
limit at 0.0421 eV/Å; a separate BFGS refinement converged in three additional
steps. The database retains the original endpoint rather than replacing its
record with the refinement.

| Representative | Total energy (eV) | Relative to best molecular state (eV) |
| --- | --- | --- |
| Best BH molecular pair | -1364.708926 | 0 |
| Refined lower-energy dissociated state | -1364.579604 | 0.129322 |
| Independently relaxed alternate dissociated state | -1364.324416 | 0.384510 |

The two dissociated representatives protonate different surface oxygen sites.
All evaluated structures retained 48 fixed coordinates and consistent fragment
tags. Only converged endpoints were selected for XYZ export. Formation and
recombination are separately exercised by the controlled two- and three-water
tests; all bond-change proposals selected in this short random search were
breaks.
