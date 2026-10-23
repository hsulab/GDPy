(hybrid-biased-monte-carlo)=

# Hybrid biased Monte Carlo

`hybrid_biased_monte_carlo` alternates MD segments with
{doc}`biased Monte Carlo <biased-mc>` blocks. It uses the same ordered setup
and restart machinery as {doc}`hybrid Monte Carlo <hmc>`, but its MC children
are `native/biased_monte_carlo`. MD remains molecular dynamics; each MC proposal
is minimised before the acceptance decision.

The setup list must contain at least one `md` stage and one
`biased_monte_carlo` stage. Repeated stages are supported. Each biased MC stage
configures its minimiser through `parameters.setup.evaluation`, including
positive `stop.steps` and `stop.fmax`. The evaluator defaults to ASE `min`.

```{literalinclude} ../../../../examples/monte_carlo/hybrid-biased-canonical.yaml
:language: yaml
:caption: examples/monte_carlo/hybrid-biased-canonical.yaml
```

Run from the repository root:

```shell
gdp -d run-hybrid-biased-mc compute examples/monte_carlo/assets/cu32.xyz --runtime examples/monte_carlo/hybrid-biased-canonical.yaml
```

Define the ensemble once on the parent. Supported ensembles are `nvt`, `uvt`,
and `suvt`; MD uses NVT at the inherited temperature for the latter two.
Every stage uses the same potential and modifiers. Parent `stop.steps` counts
cycles, MD `stop.steps` counts integration steps, and biased MC `stop.steps`
counts proposals. The minimiser's `evaluation.stop.steps` limits each relaxation.

The initial structure is minimised with the first biased MC stage's evaluator.
MD endpoints are accepted directly. A biased MC invocation minimises its incoming
structure before attempting moves, then applies its MC acceptance rules to
relaxed proposals. As with biased MC, this changes the sampled energy landscape;
it does not reproduce ordinary canonical sampling of the original surface.
There is no Hamiltonian acceptance test for the complete MD trajectory.

The {doc}`batching worker <batching>` schedules the underlying MD and minimisation
calculations. Each chain keeps `mc.out`, `mc.xyz`, checkpoints, and `steps/`;
biased MC stages retain their own proposal statistics. Pending calculations and
stage boundaries resume using the same mechanism as HMC.
