(boltzmann-sampling-overview)=

# Monte Carlo

Boltzmann sampling generates configurations with the probability prescribed
by a thermodynamic ensemble. GDPy provides canonical, semi-grand-canonical,
and grand-canonical ensembles (`nvt`, `suvt`, and `uvt`) through conventional
and hybrid Monte Carlo. The selected
{doc}`operators <operators/index>` determine which quantities can change.
{doc}`Hybrid Monte Carlo <hmc>` alternates molecular dynamics with MC moves.
See {doc}`ensembles` for thermodynamic settings and inheritance.
{doc}`Biased Monte Carlo <biased-mc>` instead minimises proposals and permits
geometry filters for independent searches over relaxed structures.

```{toctree}
:maxdepth: 3
:titlesonly:
:hidden:

ensembles <ensembles>
conventional monte carlo <mc>
biased monte carlo <biased-mc>
hybrid monte carlo <hmc>
sampling operators <operators/index>
```

Unlike {doc}`basin hopping <../global_optimisation/basin_hopping>`, Boltzmann
sampling evaluates each trial without relaxation and accepts or rejects it for
the configured ensemble. Basin hopping uses the same kinds of proposals but
relaxes trials to local minima and ranks them for structure search.

## Demo

This example uses EMT to sample a periodic Cu32 structure at 1200 K. Each
step displaces one atom by 0.1 Å while keeping the composition and cell fixed.
The run stops after 100 attempts.

Compute runtime:

```{literalinclude} ../../../examples/monte_carlo/canonical.yaml
:language: yaml
:caption: examples/monte_carlo/canonical.yaml
```

MC uses ASE single-point energies by default; no evaluator block is needed.

Run from the repository root:

```shell
gdp -d run-mc-canonical compute examples/monte_carlo/assets/cu32.xyz --runtime examples/monte_carlo/canonical.yaml
```

The sampled trajectory is written to `run-mc-canonical/cand0/mc.xyz`, including
repeated states after rejection. This short demo illustrates the workflow;
see the {doc}`conventional Monte Carlo guide <mc>` for sampling, output, and restart details.
