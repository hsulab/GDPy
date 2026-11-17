(global-optimisation-overview)=

# global optimisation

Global optimisation searches the configuration space of a chemical system for
low-energy structures. Starting from an initial population, it proposes new
candidates, evaluates them, and retains promising structures to guide the search.

The `system` defines the {doc}`population`, the `strategy` controls the search,
and the `runtime` evaluates candidates. These roles let different search methods
share the same structure builders and calculation settings.

## configuration

Set `method: global_optimisation` and configure the following sections:

| Setting | Purpose |
| --- | --- |
| `random_seed` | Seed for reproducible search choices |
| `system` | Builders, population sizes, structural comparison, and extinction checks |
| `strategy` | Search method, objective, operators, and stopping conditions |
| `runtime` | Potential and calculation settings, supplied here or in a separate file |
| `scheduler` | Where the exploration runs; local execution by default |
| `broadcast` | Optional setting variations that expand one input into independent searches |

Generation 0 builds and evaluates the initial structures. Later generations use
retained candidates to continue exploring. The {doc}`population` guide explains
initialization, retention, and diversity; each method's guide describes how it
produces new candidates.

## run a small search

This Cu₈ example uses basin hopping with ASE's EMT potential. It starts from
four structures, retains up to two distinct candidates, and runs two chains of
ten steps. It is a short demonstration, not a converged structure search.

Save the exploration settings as `expo.yaml`:

```{literalinclude} ../../../examples/global_optimisation/explorations/basin_hopping/cu8.yaml
:language: yaml
:caption: expo.yaml
```

Save the calculation settings as `runtime.yaml`:

```{literalinclude} ../../../examples/global_optimisation/runtimes/emt.yaml
:language: yaml
:caption: runtime.yaml
```

Run with `--runtime` before `explore`:

```shell
gdp -d run-cu8 --runtime runtime.yaml explore expo.yaml
```

A separate runtime can be reused across searches. You can also place its contents
under a top-level `runtime` key in `expo.yaml` and omit `--runtime`.
See {doc}`runtime configurations <../computations/runtime>` for calculation
options. The top-level `scheduler` controls the exploration; `runtime.scheduler`
controls candidate calculations.

For other systems and search methods, start with the
{doc}`examples catalogue <examples/index>`.

(exploration-output-layout)=

## results and restart

For a single exploration, outputs are written directly into the directory
specified by `-d`:

| Path | Contents |
| --- | --- |
| `gdp.out` | Progress and diagnostics |
| `candidates.db` | Evaluated structures and search metadata |
| `results/` | Method-specific reports and history figures |
| `tmp_folder/` | Intermediate calculations and checkpoints |
| `_meta/` | Exploration inputs, launch scripts, and scheduler records |

Rerun the same command with the same configuration and output directory to resume
interrupted work. Keep the scheduler provider and exploration count unchanged.
Use a fresh directory when changing the search settings, runtime, or sweep values.
Legacy `expedition-*` layouts require a new run.

## run multiple searches

A top-level `broadcast` mapping expands an input into independent searches.
For example, add this block to a suitable Cu–Ni search input to vary composition:

```{code-block} yaml
:caption: YAML

broadcast:
  system.builders.init.composition:
    - {Cu: 6, Ni: 2}
    - {Cu: 4, Ni: 4}
```

Keys identify settings by dotted paths; each listed value replaces that setting
in one search. Multiple keys generate all combinations of their values. Keep
coupled settings together, as with the complete composition dictionaries above.
See the {doc}`composition sweep example <examples/alloy-cluster>` for a complete
input and run command.

Each search gets its own directory, such as `expo.00` and `expo.01`, while
`_meta/` holds shared submission records. Direct scheduling runs the searches
sequentially; a queue scheduler submits them separately. Resume with the same
values and ordering.

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

examples/index
population
genetic-algorithm
basin_hopping
simulated-annealing
```
