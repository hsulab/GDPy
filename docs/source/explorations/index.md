(explorations)=

# gdp explore

Exploration proposes and selects candidate structures. Execution evaluates
those candidates through a complete runtime. This keeps Monte Carlo, genetic
algorithms, and other search policies independent of the potential and the
software used to run it.

Run a complete exploration configuration with:

```shell
gdp -d results explore exploration.yaml
```

See the method guides below for complete input examples and the shared
{ref}`sampling-operators` reference for MC, hybrid MC, and basin-hopping moves.

```yaml
potential:
  provider: deepmd
  parameters:
    model: [graph-0.pb, graph-1.pb]
    type_list: [Al, Cu, O]
executor:
  provider: lammps
  method: min
  parameters:
    fmax: 0.05
    steps: 400
    constraint: lowest 120
```

This runtime executes directly on the current machine by default. Add a
{ref}`scheduler configuration <scheduler-transport>` for queue or SSH execution.

The exploration layer owns proposal state, convergence, and selection. The
execution layer owns materialization, job submission, restart, and result
collection.

Global-optimisation inputs use `system` and `strategy` sections. The system
owns population construction; the strategy selects GA or BH and owns its
operators, generation policies, objective, convergence, and archive behavior.
Runtime and optional scheduler configuration remain separate execution concerns.
MC uses the same `method`/`system`/`strategy` layout; simulated annealing retains
its `recipe` wrapper.

```yaml
method: global_optimisation
runtime: {}
random_seed: 7
system:
  builders:
    random: {}
strategy:
  method: genetic_algorithm
```

Omitting the top-level `scheduler` also runs the exploration itself directly
on the current machine.

## Monte Carlo ensembles

See {doc}`Monte Carlo <mc>` for canonical, semi-grand-canonical, and grand-canonical EMT examples using single-point energies.
The guide covers displacement, identity-change, and insertion/removal moves,
chemical potentials, output inspection, restart behaviour, and current
limitations of the ensemble sampling rules.
For alternating MD and MC blocks, see the {doc}`Hybrid Monte Carlo guide
<hmc>`.

## global optimisation

Start with the {doc}`global optimisation overview <../global_optimisation/index>`
for the shared configuration and a runnable demo using either a separate
`runtime.yaml` or a runtime embedded in `expo.yaml`.

Use {doc}`genetic algorithms <../global_optimisation/genetic-algorithm>` or
{doc}`basin hopping <../global_optimisation/basin_hopping>` to search for low-energy
structures. These methods share {doc}`population configuration
<../global_optimisation/population>` and {doc}`output conventions
<../global_optimisation/output>`. The older {doc}`GA configuration discussion
<ga>` remains available as supplementary material.
