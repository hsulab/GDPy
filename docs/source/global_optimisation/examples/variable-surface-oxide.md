(ga-variable-surface-oxide-example)=

# Variable-composition surface oxide

This example searches CuₓOᵧ structures on the same two-layer
Cu(111)-p(2×2) substrate used by the fixed-composition surface-oxide example.
Both *x* and *y* may vary from 1 to 4. The candidates are relaxed with the
1-million-parameter MatterSim model and ranked in a grand-canonical form using
configured Cu and O chemical potentials.

## Variable composition

The random builder uses range-valued composition entries:

```yaml
composition:
  Cu: "1:4"
  O: "1:4"
```

The quoted ranges include both endpoints, so every initial or completion
candidate contains one to four added Cu atoms and one to four O atoms. Setting
`population.name: variable` makes the population manager compare and select
candidates across these different compositions.

The `exchange` mutation inserts or removes one independently tagged Cu or O
atom. Its `num_min_max` entries follow the order in `species` and enforce the
same inclusive 1–4 bounds. Rattle mutation supplies additional configurational
variation without changing composition. This compact example creates offspring
by mutation only; failed mutations are replaced by candidates from the random
builder.

The `cohesive_energy` target ranks candidates by
`E - N_Cu μ_Cu - N_O μ_O`. The eight substrate Cu atoms contribute the same
constant term to every candidate, while the variable overlayer composition
changes the score.

:::{note}
The chemical potentials in this example are illustrative values for exercising
the workflow. Chemical potentials define the composition being favoured and
must be calculated consistently with the selected potential and the intended
Cu/O reservoirs before interpreting a production search.
:::

## Substrate and input

The substrate is
`examples/global_optimisation/assets/cu111_p2x2_2layer.xyz`, with four Cu atoms
in each layer. The lower four atoms are fixed during relaxation, while the
upper substrate layer and the CuₓOᵧ overlayer can relax. The insertion region
covers the complete surface cell from z = 4.60 to 9.10 Å.

The complete configuration is available at
`examples/global_optimisation/cuxoy_cu111_mattersim.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/cuxoy_cu111_mattersim.yaml
:language: yaml
```

The default periodic and fragment-preservation settings apply. Substrate atoms
have tag 0, and every added Cu or O atom receives a distinct positive tag.
MatterSim is required because the search contains both Cu and O; see the
{ref}`potential-mattersim` guide for installation and model details.

## Run

After installing MatterSim, run from the repository root:

```shell
gdp -d ./run-cuxoy-cu111 explore \
    ./examples/global_optimisation/cuxoy_cu111_mattersim.yaml
```

The search is stored under `run-cuxoy-cu111/expedition-0`. When it completes,
`results/all_candidates.xyz` contains the relaxed candidates ordered by their
grand-canonical score, and `results/pop.png` plots that target by generation.

:::{note}
The two-layer slab, eight-candidate initial population, and single generation
make this a format demonstration. Use a thicker slab, converge the model and
relaxation settings, expand the population, and sample more generations for a
scientific surface phase search.
:::
