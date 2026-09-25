(boltzmann-sampling-overview)=

# Boltzmann sampling

Boltzmann sampling generates configurations with the probability prescribed
by a thermodynamic ensemble. GDPy supports canonical, semi-grand-canonical,
and grand-canonical Monte Carlo. The selected
{doc}`operators <operators/index>` determine which quantities can change.
{doc}`Hybrid Monte Carlo <hmc>` alternates molecular dynamics with MC moves.

Unlike {doc}`basin hopping <../global_optimisation/basin_hopping>`, Boltzmann
sampling may not relax every trial to a local minimum. It accepts or rejects
trials so that the sampled states retain the target ensemble distribution.
Basin hopping instead transforms the energy landscape to search for low-energy
structures.

## Demo

This example uses EMT to sample a periodic Cu32 structure at 1200 K. Each
step displaces one atom by 0.1 Å while keeping the composition and cell fixed.
The run stops after 100 attempts.

Exploration input:

```{literalinclude} ../../../examples/monte_carlo/canonical.yaml
:language: yaml
:caption: examples/monte_carlo/canonical.yaml
```

The runtime evaluates each trial with a single-point energy calculation:

```{literalinclude} ../../../examples/monte_carlo/emt.yaml
:language: yaml
:caption: examples/monte_carlo/emt.yaml
```

Run from the repository root:

```shell
gdp -d run-mc-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/canonical.yaml
```

The sampled trajectory is written to `run-mc-canonical/mc.xyz`, including
repeated states after rejection. This short demo illustrates the workflow;
see the {doc}`MC guide <mc>` for sampling, output, and restart details.
