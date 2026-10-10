(bh-tio2-water-rotation)=

# Two or three neighboring waters on anatase (101)

This example searches water orientations with {doc}`../../../explorations/operators/rotate`
and xreac's bundled `ffield.reax.TiOH.Monti2012` parameters. GDPy's `reax` extra
pins the xreac source revision prepared for v0.10.0, which includes these Ti/O/H
parameters. Install `pip install -e '.[reax]'` before running the example.

The two seed frames contain 192 TiO₂ slab atoms and two neighboring H₂O
molecules on a four-layer p(1×4) anatase (101) slab in a periodic cell with
24 Å of vacuum along z and 48 fixed bottom-layer atoms. Substrate atoms have tag 0; the waters have tags
1 and 2. The parallel starting structure is adapted from xreac's
[`validation/anatase/2water_initial.xyz`](https://github.com/hsulab/xreac/tree/2d0f2bda35056a1ac941a68d4df16670923ccfd5/validation/anatase).
Its Ti5c sites are 3.784 Å apart. In the hydrogen-bonded seed each water is
shifted 0.5 Å toward the other, and one water is rotated about its oxygen so
one hydrogen points toward the other oxygen. A 5° deviation from perfect
alignment avoids an undefined collinear torsion derivative.

## Compare the two starting motifs

From the repository root:

```shell
python examples/global_optimisation/tio2_water_rotation.py \
  --output tmp/tio2-water-rotation --steps 2000 --fmax 0.02
```

The script independently relaxes both seeds with ASE FIRE and reports total
energies, maximum mobile force, Ti–O distances, O–H distances, water-bisector
alignment, and water-to-water hydrogen-bond geometry in `summary.json`.
`--build-only` checks the initial structures without evaluating xreac.
`--resume` reuses a completed relaxation in an existing output directory.
Each subdirectory contains `initial.xyz`, `relaxed.xyz`, a trajectory and
an optimizer log. `relaxed_pair.xyz` holds both endpoints for viewing.

The geometric hydrogen-bond criterion is O···O ≤ 3.5 Å, H···O ≤ 2.5 Å and
O–H···O ≥ 150°. Parallel waters have bisector alignment within 20°.
These diagnostics describe the geometries; they do not impose restraints.
The script records whether both requested motifs survive relaxation, and
reports unconverged endpoints explicitly. Two plausible seeds can converge
to the same minimum. Total-energy differences compare the same composition;
adsorption energies require clean-slab and isolated-water reference energies.

For the three-water motif, use three consecutive Ti5c sites on the same slab:

```shell
python examples/global_optimisation/tio2_water_rotation.py --waters 3 \
  --optimizer bfgs --output tmp/tio2-water3-rotation --steps 500 --fmax 0.02
```

`tio2_water3_seeds.xyz` contains 201 atoms: the same 192-atom slab plus three
waters with tags 1, 2 and 3. The parallel seed adds a third neighbor to the
two-water row. The hydrogen-bonded seed shifts the outer waters inward by
0.8 Å and rotates the first two waters to form a chain with two donor–acceptor
contacts. This captures the three-neighbor motif in the reference image;
the screenshot supplies an arrangement, not numerical atomic coordinates.
The same adsorption and hydrogen-bond diagnostics cover all water pairs.

## Basin hopping with molecular rotations

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/tio2_water2.yaml
:language: yaml
```

```{literalinclude} ../../../../../examples/global_optimisation/runtimes/xreac_tioh.yaml
:language: yaml
```

```shell
gdp --runtime examples/global_optimisation/runtimes/xreac_tioh.yaml \
  -d tmp/tio2-water-rotation-bh \
  explore examples/global_optimisation/explorations/basin_hopping/tio2_water2.yaml
```

Replace the exploration input with `tio2_water3.yaml` and use a new output
directory to run the three-water search with the same rotation settings and
calculation runtime.

Two initial minima seed two chains with four attempted rotations each.
Each rotation keeps its oxygen pivot fixed during the proposal, preserves
the internal molecular geometry, and then relaxes all mobile atoms with
BFGS. This includes oxygen motion and possible water dissociation; original
tags identify the proposed groups, not a dynamic bond assignment.
The calculation runtime applies the fixed-bottom-layer constraint explicitly.
The `atoms` comparator keeps geometrically distinct candidates without
introducing orientation-insensitive duplicate filtering in this short demo.

Inspect `candidates.db` for every evaluated minimum, including rejected
trials, and `results/history/gen0001.png` for the chain history. Four hops
per chain demonstrate the operator and are not an exhaustive stability search.

## Validation result

With xreac source revision `2d0f2bd` and Monti2012, the supplied short search
completed all eight rotations, retaining ten evaluated minima including the
two initial relaxations. All minima had finite energies and forces, intact
Ti-bound waters, unchanged tags and unchanged fixed-atom coordinates. Their
maximum mobile force was at most 0.0197 eV/Å.

Both initial motifs and every relaxed rotation trial produced parallel-like
waters, with O···O distances of 3.754–3.767 Å and no water-to-water hydrogen
bond under the stated geometric criterion. Independent FIRE relaxations at
fmax=0.02 eV/Å also returned parallel-like pairs from both starting motifs.
This test did not establish a second hydrogen-bonded adsorption minimum;
other sites, orientations or potentials may give different outcomes.

For three waters, independent BFGS relaxations converged to a parallel trio
and a tilted arrangement 0.305 eV higher in total energy. All waters remained
intact and Ti-bound, but neither endpoint retained a water-to-water hydrogen
bond. The three-water search also completed eight rotations and stored ten
converged minima. The tilted initial minimum entered the database; all eight
relaxed rotation trials were parallel-like, without water-to-water hydrogen
bonds. Thus adding the third neighbor did not recover the requested hydrogen-
bonded minimum in this short test.

A stricter independent two-water FIRE refinement at fmax=0.005 eV/Å reached
that force tolerance for the seed that began hydrogen-bonded, still yielding
a parallel-like pair. The parallel seed did not reach the tighter tolerance
within 2000 additional steps (maximum mobile force 0.0239 eV/Å). Those tighter
endpoints are not a pair of converged minima for energy ranking.
