(global-optimisation-overview)=

# global optimisation

Global optimisation searches for low-energy atomic structures by repeatedly
building candidates, relaxing them, and using the results to guide the next
generation. GDPy provides two population-based strategies:

| Strategy | How it proposes new candidates |
| --- | --- |
| {doc}`Genetic algorithm <genetic-algorithm>` | Selects compatible parents for crossover, applies mutations, and fills any remaining generation slots with builders. |
| {doc}`Basin hopping <basin_hopping>` | Selects starting structures for independent chains, then proposes, relaxes, and accepts or rejects a trial at each chain step. |

Both use `method: global_optimisation` and the same {doc}`population configuration
<population>`. Generation 0 builds and evaluates the initial candidates. Later
generations select from the best distinct, eligible candidates in the search
history. The strategy determines how new candidates are produced; the runtime
determines how they are evaluated.

## Configuration at a glance

Search configuration follows the same three-section layout as MC: `method`,
`system`, and `strategy`. There is no `recipe` wrapper.

| Section | Purpose |
| --- | --- |
| `system` | Named builders, initialization, retained population size, generation size, comparison, and extinction rules. |
| `strategy` | Algorithm method, operators, search policies, objective, convergence, and archive behavior. |
| `random_seed` | Reproducible search random streams. |
| `runtime` | Potential and executor used to evaluate candidates; can instead be supplied in a separate file. |
| `scheduler` | Where the exploration loop runs; defaults to direct execution. |

`system.initial.total_size` is the number of initial candidates.
`system.retained_size` is the maximum number kept for selection and defaults
to `system.generation.total_size`. The generation size counts offspring for
GA and chains for BH. For BH, `strategy.steps_per_chain` counts attempted
proposals per chain per generation, including rejected and invalid proposals.

## Demo: choose where to define the runtime

These two layouts run the same Cu₈ basin-hopping search with ASE's EMT potential.
The search builds four initial structures, retains up to two distinct candidates,
and launches two chains of ten steps in generation 1. BH defaults to one hopping
generation when `strategy.convergence` is omitted. The EMT runtime relaxes initial
structures and valid trials with a force tolerance of 0.05 eV/Å.

### Separate `expo.yaml` and `runtime.yaml`

Save this exploration configuration as `expo.yaml`:

```{literalinclude} ../../../examples/global_optimisation/explorations/basin_hopping/cu8.yaml
:language: yaml
:caption: expo.yaml
```

Save the calculation settings separately as `runtime.yaml`:

```{literalinclude} ../../../examples/global_optimisation/runtimes/emt.yaml
:language: yaml
:caption: runtime.yaml
```

Run the search with `--runtime` **before** the `explore` subcommand:

```shell
gdp -d run-separate --runtime runtime.yaml explore expo.yaml
```

This layout lets several explorations share one runtime. To compare suitable
potentials, select another runtime file and use a new output directory. The
repository examples use this layout; see the {doc}`GA examples <examples/index>`
and {doc}`BH examples <bh/examples/index>`.

### Embed the runtime in `expo.yaml`

Alternatively, keep the exploration settings above and append this top-level
`runtime` section to `expo.yaml`. The contents of the separate runtime file
become the value of `runtime`, indented by two spaces:

```yaml
runtime:
  potential:
    provider: emt
  executor:
    provider: ase
    method: min
    parameters:
      fmax: 0.05
      steps: 1000
```

Now the single file contains both the search and its calculation settings:

```shell
gdp -d run-embedded explore expo.yaml
```

Use either layout with GA or BH. A minimization runtime supplies the local
relaxation used by both strategies. See {doc}`runtime configurations
<../computations/runtime>` for other potentials and calculation settings.
`runtime.scheduler` schedules candidate calculations, while the top-level
`scheduler` schedules the exploration loop itself.

## Switch to a genetic algorithm

Keep the shared `system` and chosen runtime layout. Replace the BH
`strategy` section with:

```yaml
strategy:
  method: genetic_algorithm
  convergence:
    generation: 1
  reproduction:
    size: 2
    mutation_probability: 0.5
  mutation:
    size: 0
  completion:
    builder_proportions:
      - builder: random
        proportion: 1.0
  operators:
    crossover:
      method: cut_and_splice
    mutation:
      method: rattle
```

This produces two offspring after initialization. Parent compatibility follows
the crossover operator automatically; no composition-selection switch is needed.
See {doc}`population` for the shared settings and {doc}`genetic-algorithm` for
reproduction, mutation, and completion behavior.

## Combine random candidates and seed structures

The {doc}`seeded Cu₈ basin-hopping example <bh/examples/seeded-cluster>` uses
two initial builders: `random_structure_improved` creates two candidates and
a `direct` builder reads two frames from a seed file. Exact builder allocations
combine them into one initial population, evaluated by the same runtime.

## Results and restart

Inspect `gdp.out` for progress, `candidates.db` for evaluated candidates, and
`results/` for strategy-specific reports and lineage figures. Rerun the same
command with the same configuration and output directory to resume interrupted
work. Use a new directory when changing the strategy, runtime, or search setup.
See {doc}`output` for the complete directory layout and restart conventions.
