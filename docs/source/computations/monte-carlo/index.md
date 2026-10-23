(boltzmann-sampling-overview)=

# Monte Carlo

GDPy supports standard, biased, and hybrid Monte Carlo with `nvt`, `uvt`, and
`suvt` ensembles. Choose {doc}`operators <operators/index>` to define the proposed
changes and an {doc}`ensemble <ensembles>` to set the thermodynamic conditions.

## Schemes

```{figure} monte_carlo.png
:alt: "Monte Carlo workflow: evaluate the initial structure, optionally run a simulation, then perform MC steps consisting of proposal, evaluation, and Metropolis acceptance."
:width: 100%

The simulation stage is optional. Each MC step proposes a change, evaluates it,
and applies the Metropolis acceptance rule.
```

| Scheme | Simulation | Evaluation |
| --- | --- | --- |
| {doc}`Standard MC <mc>` (`monte_carlo`) | None | Single-point calculation (`spc`) |
| {doc}`Biased MC <biased-mc>` (`biased_monte_carlo`) | None | Minimisation (`min`) |
| {doc}`Hybrid MC <hmc>` (`hybrid_monte_carlo`) | Molecular dynamics (`md`) | Single-point calculation (`spc`) |
| {doc}`Hybrid biased MC <hybrid-biased-mc>` (`hybrid_biased_monte_carlo`) | Molecular dynamics (`md`) | Minimisation (`min`) |

The evaluation method is fixed by the MC scheme and applies to initialisation
and proposal evaluation. Hybrid schemes alternate simulation stages with MC
blocks; MD endpoints are accepted directly, while MC proposals undergo the
acceptance test. Biased MC compares relaxed structures, changing the sampled
energy landscape.

```{toctree}
:maxdepth: 3
:titlesonly:
:hidden:

ensembles <ensembles>
batching <batching>
standard monte carlo <mc>
biased monte carlo <biased-mc>
hybrid monte carlo <hmc>
hybrid biased monte carlo <hybrid-biased-mc>
sampling operators <operators/index>
```

Standard MC evaluates each trial without relaxation and accepts or rejects it
for the configured ensemble. {doc}`Basin hopping <../../global_optimisation/basin_hopping>` uses the same kinds of proposals but
relaxes trials to local minima and ranks them for structure search.

## Demo

This example uses EMT to sample a periodic Cu32 structure at 1200 K. Each
step displaces one atom by 0.1 Å while keeping the composition and cell fixed.
The run stops after 100 attempts.

Compute runtime:

```{literalinclude} ../../../../examples/monte_carlo/canonical.yaml
:language: yaml
:caption: examples/monte_carlo/canonical.yaml
```

MC uses ASE single-point energies by default; no evaluator block is needed.

Run from the repository root:

```shell
gdp -d run-mc-canonical compute examples/monte_carlo/assets/cu32.xyz --runtime examples/monte_carlo/canonical.yaml
```

The sampled trajectory is written to `run-mc-canonical/mc.xyz`, including
repeated states after rejection. This short demo illustrates the workflow;
see the {doc}`standard Monte Carlo guide <mc>` for sampling, output, and restart details.
