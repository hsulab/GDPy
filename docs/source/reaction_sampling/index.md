# reaction sampling

Sample structures along reaction coordinates to explore rare events and generate
candidates for active learning. {doc}`Umbrella sampling <umbrella-sampling>` uses
harmonic distance restraints to sample independent windows, with equilibration
before production.

These workflows run through `gdp explore`. Umbrella sampling uses
`method: umbrella_sampling`; it does not use the population-based
`method: global_optimisation` configuration.

```{toctree}
:maxdepth: 1
:titlesonly:
:hidden:

umbrella-sampling
```
