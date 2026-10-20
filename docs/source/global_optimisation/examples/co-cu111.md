(co-cu111-example)=

# CO adsorption on Cu(111)

This example searches positions and orientations of one CO molecule on an
orthogonal Cu(111)-p(2×2) slab. The right-angled surface cell is rectangular,
5.11 × 4.43 Å, not exactly square. Three layers contain twelve Cu atoms.
The bottom four Cu atoms are fixed during both MD and minimisation; the upper
two layers and CO are mobile.

The dedicated substrate `cu111_p2x2_3layer_sa.xyz` preserves the geometry of
`cu111_p2x2_3layer.xyz`, with substrate tags set to zero and all three PBC flags
enabled to match the population interface. Its 24 Å cell height leaves vacuum
between periodic slab images. The original shared substrate is unchanged.

## Simulated annealing with DPA4C

Four starting structures are built by inserting an intact CO molecule with
random orientation and centre of mass 1.5–3.5 Å above the top Cu layer.
The search retains up to four distinct minima and performs three generations
of four quenches. Each quench starts with fresh velocities and cools from 600
to 100 K over 1,000 steps at 0.5 fs (0.5 ps). If fewer than four parents survive
duplicate filtering, selection uses replacement.

```{literalinclude} ../../../../examples/global_optimisation/explorations/simulated_annealing/co_cu111.yaml
:language: yaml
```

The separate runtime uses the DeepMD ASE backend and an ASE Langevin thermostat:

```{literalinclude} ../../../../examples/global_optimisation/explorations/simulated_annealing/co_cu111_runtime.yaml
:language: yaml
```

Supply a DPA4C foundation model at `models/dpa4c-foundation.pt2`, or edit the
runtime model path. The model must cover Cu, C, and O and be exported for the
execution device. No weights are bundled or downloaded. See
{ref}`DPA4C setup <potential-deepmd>` for export and installation guidance.
Element coverage alone does not establish accuracy for CO adsorption.

Run from the repository root:

```shell
gdp -d run-co-cu111-sa \
  --runtime examples/global_optimisation/explorations/simulated_annealing/co_cu111_runtime.yaml \
  explore examples/global_optimisation/explorations/simulated_annealing/co_cu111.yaml
```

CO starts intact but is flexible during MD and minimisation. Fragment tags
identify the initial molecule; they do not constrain its bond or prevent
desorption or dissociation. Inspect the final structures to check their chemical
state. Candidates are ranked by total energy because their composition is fixed;
these energies are not adsorption energies relative to a clean slab and gas-phase
CO. This small, short search demonstrates the workflow, not convergence or model
accuracy. Results are written to `results/all_candidates.xyz` and `results/pop.png`.
