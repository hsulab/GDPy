(ga-examples)=

# Examples

These examples demonstrate complete GA searches for different classes of
atomic systems. They use small populations and inexpensive potentials where
possible so the workflow, configuration, and outputs remain easy to inspect.
Increase the population size, number of generations, and calculation accuracy
for production searches.

Exploration inputs live under `examples/global_optimisation/explorations/`,
grouped by method. Shared calculation inputs live under `runtimes/` in the same
directory. Each guide shows both files. Select a runtime with
`gdp --runtime <runtime.yaml> explore <exploration.yaml>`; the exploration can
be reused with another suitable potential without editing its recipe. Use a
new output directory when changing runtimes, and preserve the slab constraints
when adapting surface examples.

All examples run directly on the current machine by default, so their input
files omit `scheduler`. See {ref}`scheduler-transport` to configure queue or
SSH execution.

The examples are grouped by their main physical system: **Cluster** covers
finite atomic and molecular aggregates, **Bulk** covers fully periodic
crystals, and **Interface** covers surfaces, supported clusters, and
adsorbates. Select a chemical system below to open its complete configuration
and guide.

| Main system | Specific system | Potential | Notes |
| --- | --- | --- | --- |
| Cluster | {ref}`Cu₁₃ <ga-cluster-example>` | EMT | |
| Cluster | {ref}`Cu₇Ni₆ <ga-alloy-cluster-example>` | EMT | Swap mutation |
| Cluster | {ref}`(H₂O)₄ <ga-water-cluster-example>` | MatterSim | |
| Bulk | {ref}`Cu₄ <ga-bulk-example>` | EMT | |
| Interface | {ref}`Cu₄O₄/Cu(111) <ga-surface-oxide-example>` | MatterSim | |
| Interface | {ref}`CuₓOᵧ/Cu(111) <ga-variable-surface-oxide-example>` | MatterSim | Variable composition |
| Interface | {ref}`Cu₄/α-Al₂O₃(0001) <ga-supported-nanoparticle-example>` | MatterSim | |
| Interface | {ref}`CO–Cu₄/α-Al₂O₃(0001) <ga-supported-cluster-adsorbate-example>` | MatterSim | Random generation |
| Interface | {ref}`CO–Cu₄/α-Al₂O₃(0001) <ga-adsorbate-insertion-example>` | MatterSim | Site insertion |

```{toctree}
:maxdepth: 1
:hidden:

cluster
alloy-cluster
water-cluster
bulk
surface-oxide
variable-surface-oxide
supported-nanoparticle
supported-cluster-adsorbate
adsorbate-insertion
```
