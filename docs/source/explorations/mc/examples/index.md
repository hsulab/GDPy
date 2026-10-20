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
gdp -d run-mc-canonical compute examples/monte_carlo/assets/cu32.xyz --runtime examples/monte_carlo/canonical.yaml

gdp -d run-mc-semi-grand-canonical compute examples/monte_carlo/assets/cu16ni16.xyz --runtime examples/monte_carlo/semi-grand-canonical.yaml

gdp -d run-mc-grand-canonical compute examples/monte_carlo/assets/aucu8.xyz --runtime examples/monte_carlo/grand-canonical.yaml
```

All three define one EMT potential and use the default ASE single-point
MC evaluator. Optional `executor.parameters.setup.executor` customizes its
provider, parameters, or scheduler; its method must remain `spc`.

The {doc}`hybrid examples <../../hmc/examples/index>` alternate MD segments with MC proposals
and list MD and conventional MC executors directly in `parameters.setup`.

## Choose an example

```{toctree}
:maxdepth: 1

canonical
semi-grand-canonical
grand-canonical
cu111-oxidation-xreac
```
