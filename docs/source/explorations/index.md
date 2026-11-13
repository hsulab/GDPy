(explorations)=

# gdp explore

`gdp explore` explores the configuration space of a chemical system according
to user-defined rules. These rules determine how candidate structures are
generated, evaluated, and selected to guide further exploration.

## run an exploration

Run an exploration with a separate calculation runtime:

```shell
gdp -d results --runtime runtime.yaml explore exploration.yaml
```

Place `--runtime` before `explore`. Alternatively, define `runtime` inside
`exploration.yaml` and omit the command-line option:

```shell
gdp -d results explore exploration.yaml
```

The input selects a `method`, defines the structures and search settings, and
can set a `random_seed`. The method guides below provide complete inputs.
See {doc}`runtime configurations <../computations/runtime>` for potentials and
calculation settings.

Execution is local by default. An optional top-level `scheduler` controls where
the exploration runs; `runtime.scheduler` controls its candidate calculations.
See {ref}`scheduler configuration <scheduler-transport>` for queue and SSH setup.

## global optimisation

Global optimisation searches for stable structures by finding low-energy
regions of configuration space. See the
{doc}`global optimisation guide <../global_optimisation/index>` for configuration
and examples.

## reaction sampling

Reaction sampling explores how a chemical system transforms between states,
sampling configurations along reaction pathways and in transition regions.
See the {doc}`reaction sampling guide <../reaction_sampling/index>` for
configuration and examples.
