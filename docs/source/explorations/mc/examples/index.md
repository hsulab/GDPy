(mc-examples)=

# Examples

These examples cover canonical, semi-grand-canonical, and grand-canonical MC
with EMT, plus a short Cu(111) oxidation run with xreac. Read the
{ref}`Monte Carlo guide <monte-carlo>` for proposal settings, sampling
limitations, output files, and restart behaviour.

The existing grand-canonical inputs use the shared parser's plain `exchange`
method; these inputs remain available as legacy examples, but this method is
not part of the user-facing operator catalogue.
For the curated MC operator list and detailed-balance requirements, see
{ref}`sampling-operators`. The documentation catalogue does not change parser support
or the acceptance rule used by these inputs.

## Run the EMT examples

The complete inputs and initial structures are in `examples/monte_carlo/`.
Run from the **repository root**, since structure paths are relative to it:

```shell
gdp -d run-mc-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/canonical.yaml

gdp -d run-mc-semi-grand-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/semi-grand-canonical.yaml

gdp -d run-mc-grand-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/grand-canonical.yaml
```

All three use this sequential single-worker runtime:

```{literalinclude} ../../../../../examples/monte_carlo/emt.yaml
:language: yaml
```

`method: spc` evaluates the initial structure and every trial without relaxation
or dynamics. Changing it to `min` samples relaxed trial structures and creates
an optimisation workflow, rather than ordinary finite-temperature MC. For
population-based optimisation, see {doc}`../../../global_optimisation/basin_hopping`.

The {doc}`hybrid examples <../../hmc/examples/index>` alternate MD segments with MC proposals
and include their own initialization, MD, and MC runtimes.

## Choose an example

```{toctree}
:maxdepth: 1

canonical
semi-grand-canonical
grand-canonical
cu111-oxidation-xreac
```
