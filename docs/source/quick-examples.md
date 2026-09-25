(quick-examples)=

# quick examples

Use this page to choose a complete example, then follow the link to its input,
run command, and notes. Unless an example says otherwise, run commands from the
GDPy repository root and use a new output directory for each run.

The examples use lightweight or readily available potentials to demonstrate
each workflow. You can replace the example potential with any compatible
potential that supports the required elements and calculation task. Update the
runtime configuration for the chosen potential and reconsider model-dependent
settings, such as chemical potentials, force tolerances, and relaxation
budgets.

These tables index the complete compute and exploration examples in this
documentation. Smaller configuration snippets, such as individual potential,
builder, selector, and scheduler configurations, remain in their respective
reference guides.

## Compute

The introductory demo and the first four task examples use ASE's bundled EMT
potential and require no model download. Generate the shared task structures as
described in the {doc}`task examples overview <computations/tasks/index>`.

| Example | System | Focus |
| --- | --- | --- |
| <a href="computations/compute.html#compute-copper-dimers-example">Cu<sub>2</sub></a> | molecule | Complete `gdp compute` lifecycle |
| [Cu<sub>2</sub>](computations/tasks/single-point.md) | molecule | Energy and force evaluation without moving atoms |
| [Cu<sub>2</sub>](computations/tasks/relaxation.md) | molecule | Atomic relaxation with a fixed cell |
| {doc}`Cu <computations/tasks/cell-relaxation>` | bulk | Joint optimization of positions and cell |
| [Cu<sub>32</sub>](computations/tasks/molecular-dynamics.md) | bulk | Short NVT trajectory with a Berendsen thermostat |
| {ref}`Au/Al <compute-neb-example>` | surface | Transition path through the reactor-worker API |
| {ref}`User system <compute-dimer-example>` | user defined | Local saddle-point search configuration |
| {ref}`User system <compute-vibrations-example>` | user defined | Native finite-difference frequency tasks |

## Monte Carlo

| Example | System | Focus |
| --- | --- | --- |
| [Cu<sub>32</sub>](explorations/mc/examples/canonical.md) | bulk | Canonical displacements |
| [Cu<sub>16</sub>Ni<sub>16</sub>](explorations/mc/examples/semi-grand-canonical.md) | bulk | Semi-grand-canonical identity changes |
| [Cu<sub>n</sub>Au](explorations/mc/examples/grand-canonical.md) | bulk | Grand-canonical insertion and removal |
| {doc}`O/Cu(111) <explorations/mc/examples/cu111-oxidation-xreac>` | surface | Grand-canonical O exchange |

See the {ref}`Monte Carlo guide <monte-carlo>` for proposal rules, outputs,
restart behavior, and sampling limitations.

## Hybrid Monte Carlo

| Example | System | Focus |
| --- | --- | --- |
| [Cu<sub>32</sub>](explorations/hmc/examples/canonical.md) | bulk | NVT MD followed by Cu displacements |
| [Cu<sub>16</sub>Ni<sub>16</sub>](explorations/hmc/examples/semi-grand-canonical.md) | bulk | NVT MD followed by identity changes |
| {doc}`O/Cu(111) <explorations/hmc/examples/cu111-oxidation-xreac>` | surface | NVT MD followed by O exchange |

See the {ref}`Hybrid Monte Carlo guide <hybrid-monte-carlo>` for cycle settings,
runtime roles, outputs, and restart behavior.

## Genetic algorithm

| Example | System | Focus |
| --- | --- | --- |
| [Cu<sub>8</sub>](global_optimisation/examples/cluster.md) | cluster | Minimal atomic-cluster search |
| [Cu<sub>7</sub>Ni<sub>6</sub>](global_optimisation/examples/alloy-cluster.md) | cluster | Chemical ordering with swap mutation |
| [(H<sub>2</sub>O)<sub>4</sub>](global_optimisation/examples/water-cluster.md) | cluster | Molecular-fragment search |
| [Cu<sub>4</sub>](global_optimisation/examples/bulk.md) | bulk | Periodic crystal search |
| [Cu<sub>4</sub>O<sub>4</sub>/Cu(111)](global_optimisation/examples/surface-oxide.md) | surface | Fixed-composition surface reconstruction |
| [Cu<sub>x</sub>O<sub>y</sub>/Cu(111)](global_optimisation/examples/variable-surface-oxide.md) | surface | Variable composition |
| [Cu<sub>4</sub>/α-Al<sub>2</sub>O<sub>3</sub>(0001)](global_optimisation/examples/supported-nanoparticle.md) | cluster/surface | Supported nanoparticle search |
| [CO–Cu<sub>4</sub>/α-Al<sub>2</sub>O<sub>3</sub>(0001)](global_optimisation/examples/supported-cluster-adsorbate.md) | molecule/surface | Random and site-insertion builders |

See the {ref}`genetic-algorithm guide <genetic-algorithm>` for the search
configuration shared by these examples.

## Basin hopping

| Example | System | Focus |
| --- | --- | --- |
| [Cu<sub>8</sub>](global_optimisation/bh/examples/cluster.md) | cluster | Basic search and trajectory export |
| [Cu<sub>8</sub>](global_optimisation/bh/examples/seeded-cluster.md) | cluster | Random structures and seed-file frames |
| [Cu<sub>6</sub>Ni<sub>2</sub> and Cu<sub>4</sub>Ni<sub>4</sub>](global_optimisation/bh/examples/compositions.md) | cluster | Two compositions in one allocation |
| [Cu<sub>6</sub>Ni<sub>x</sub>](global_optimisation/bh/examples/variable-composition.md) | cluster | Ni insertion and removal |
| [Cu<sub>4</sub>O<sub>4</sub>](global_optimisation/bh/examples/cuox_tace.md) | cluster | Oxide-cluster search |
| [Cu<sub>4</sub>O<sub>4</sub>](global_optimisation/bh/examples/cuox_thanos.md) | cluster | O-O extinction and chain restarts |

See the {doc}`basin-hopping guide <global_optimisation/basin_hopping>` for
acceptance, batching, checkpoints, and lineage output.
