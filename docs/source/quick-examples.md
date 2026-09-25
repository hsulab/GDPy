(quick-examples)=

# quick examples

Use this page to choose a complete example, then follow the link to its input,
run command, and notes. Unless an example says otherwise, run commands from the
GDPy repository root and use a new output directory for each run.

These tables index the complete compute and exploration examples in this
documentation. Smaller configuration snippets, such as individual potential,
builder, selector, and scheduler configurations, remain in their respective
reference guides.

## Compute

The introductory demo and the first four task examples use ASE's bundled EMT
potential and require no model download. Generate the shared task structures as
described in the {doc}`task examples overview <computations/tasks/index>`.

| Example | System | What it demonstrates |
| --- | --- | --- |
| {ref}`Copper dimers <compute-copper-dimers-example>` | Cu dimers, EMT | Complete `gdp compute` lifecycle |
| {doc}`Single-point energy and forces <computations/tasks/single-point>` | Cu dimers, EMT | Energy and force evaluation without moving atoms |
| {doc}`Fixed-cell relaxation <computations/tasks/relaxation>` | Cu dimers, EMT | Atomic relaxation with a fixed cell |
| {doc}`Cell relaxation <computations/tasks/cell-relaxation>` | Strained bulk Cu, EMT | Joint optimization of positions and cell |
| {doc}`Molecular dynamics <computations/tasks/molecular-dynamics>` | Cu supercell, EMT | Short NVT trajectory with a Berendsen thermostat |
| {ref}`NEB surface-diffusion path <compute-neb-example>` | Au on Al, EMT | Transition path through the reactor-worker API |
| {ref}`Dimer transition-state search <compute-dimer-example>` | User-supplied system, CP2K | Local saddle-point search configuration |
| {ref}`Vibrational analysis <compute-vibrations-example>` | Relaxed structure, VASP or CP2K | Native finite-difference frequency tasks |

## Monte Carlo

| Example | Ensemble and system | Potential |
| --- | --- | --- |
| {doc}`Canonical Cu displacements <explorations/mc/examples/canonical>` | NVT, periodic Cu32 | EMT |
| {doc}`Semi-grand-canonical Cu/Ni identity changes <explorations/mc/examples/semi-grand-canonical>` | Fixed-size Cu16Ni16 alloy | EMT |
| {doc}`Grand-canonical Cu insertion and removal <explorations/mc/examples/grand-canonical>` | Variable Cu count in a periodic box | EMT |
| {doc}`Cu(111) oxidation <explorations/mc/examples/cu111-oxidation-xreac>` | Grand-canonical O exchange on Cu(111) | xreac / ReaxFF |

See the {ref}`Monte Carlo guide <monte-carlo>` for proposal rules, outputs,
restart behavior, and sampling limitations.

## Hybrid Monte Carlo

| Example | MD/MC cycle | Potential |
| --- | --- | --- |
| {doc}`Canonical Cu displacements and MD <explorations/hmc/examples/canonical>` | NVT MD followed by Cu displacements | EMT |
| {doc}`Semi-grand-canonical Cu/Ni changes and MD <explorations/hmc/examples/semi-grand-canonical>` | NVT MD followed by identity changes | EMT |
| {doc}`Cu(111) oxidation with MD <explorations/hmc/examples/cu111-oxidation-xreac>` | NVT MD followed by O exchange | xreac / ReaxFF |

See the {ref}`Hybrid Monte Carlo guide <hybrid-monte-carlo>` for cycle settings,
runtime roles, outputs, and restart behavior.

## Genetic algorithm

| Example | Main system | Potential | Focus |
| --- | --- | --- | --- |
| {ref}`Cu8 cluster <ga-cluster-example>` | Cluster | EMT | Minimal atomic-cluster search |
| {ref}`Cu7Ni6 alloy cluster <ga-alloy-cluster-example>` | Cluster | EMT | Chemical ordering with swap mutation |
| {ref}`(H2O)4 molecular cluster <ga-water-cluster-example>` | Cluster | xreac / ReaxFF | Molecular-fragment search |
| {ref}`Cu4 bulk crystal <ga-bulk-example>` | Bulk | EMT | Periodic crystal search |
| {ref}`Cu4O4/Cu(111) surface oxide <ga-surface-oxide-example>` | Interface | MatterSim | Fixed-composition surface reconstruction |
| {ref}`CuxOy/Cu(111) surface oxide <ga-variable-surface-oxide-example>` | Interface | MatterSim | Variable composition |
| {ref}`Cu4 on alpha-Al2O3(0001) <ga-supported-nanoparticle-example>` | Interface | MatterSim | Supported nanoparticle search |
| {ref}`CO-Cu4 on alpha-Al2O3(0001) <ga-supported-cluster-adsorbate-example>` | Interface | MatterSim | Random and site-insertion builders |

See the {ref}`genetic-algorithm guide <genetic-algorithm>` for the search
configuration shared by these examples.

## Basin hopping

| Example | Main system | Potential | Focus |
| --- | --- | --- | --- |
| {ref}`Cu8 cluster <bh-cu8-example>` | Cluster | EMT | Basic search and trajectory export |
| {ref}`Seeded Cu8 cluster <bh-seeded-cu8-example>` | Cluster | EMT | Random structures and seed-file frames |
| {ref}`Cu6Ni2 and Cu4Ni4 <bh-composition-broadcast-example>` | Cluster | EMT | Two compositions in one allocation |
| {ref}`Variable-composition Cu6Nix <bh-variable-composition-example>` | Cluster | EMT | Ni insertion and removal |
| {ref}`Cu4O4 with TACE <bh-cuox-tace-example>` | Cluster | TACE | Oxide-cluster search |
| {ref}`Cu4O4 with extinction <bh-cuox-thanos-example>` | Cluster | MatterSim | O-O extinction and chain restarts |

See the {doc}`basin-hopping guide <global_optimisation/basin_hopping>` for
acceptance, batching, checkpoints, and lineage output.
