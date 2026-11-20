(global-optimisation-examples)=
(ga-examples)=
(bh-examples)=

# examples

Choose a physical system, then a search method. Each system guide introduces
its geometry and shared setup once, followed by the available genetic algorithm
(GA), basin-hopping (BH), or simulated annealing (SA) recipes. Small populations
and short searches keep these demonstrations easy to inspect; increase search
and calculation budgets for production work.

(go-running-examples)=

## running the examples

Each example lists its method folder, exploration input, runtime, and output
directory. From the repository root, enter the method folder and run:

```{code-block} shell
:caption: Shell

cd examples/global_optimisation/<method>
gdp -d <output-directory> --runtime <runtime-file> explore <exploration-file>
```

Replace the placeholders with the values in the example's run table. Keep
`--runtime` before `explore`. Shared runtimes are under `../runtimes/`; a
local `runtime.yaml` must be created as described in the example.

Paths are relative to the method folder: structure inputs use `../assets/`,
and supplied model paths resolve to the repository's `models/` directory.
Outputs are written relative to the same working directory. Rerun unchanged to
resume; use a fresh output directory when changing the input or runtime.

Activate an environment with the required potential dependencies before running.
EMT needs no model download; other runtimes may need model files or additional
packages. Set `OMP_NUM_THREADS=1` when reproducing the single-thread CPU runs
and timings described in the examples.

Local execution is the default. See {ref}`scheduler-transport` for queue and SSH
setup, and {ref}`exploration-output-layout` for outputs and restart conventions.
The short search and relaxation budgets demonstrate the workflows; increase
them and check convergence before interpreting the results scientifically.

## choose a system

| Main system | Specific system | Methods | Potential / variants |
| --- | --- | --- | --- |
| Cluster | {ref}`Cu₈ <cu8-example>` | GA, BH, SA | EMT; {ref}`seeded BH <bh-seeded-cu8-example>` |
| Cluster | {ref}`Cu–Ni <ga-alloy-cluster-example>` | GA, BH | EMT; chemical ordering, composition sweep, Ni insertion/removal |
| Cluster | Cu₄O₄ | BH | {ref}`TACE <bh-cuox-tace-example>`; {ref}`MatterSim with O–O extinction <bh-cuox-thanos-example>` |
| Cluster | {ref}`(H₂O)₄ <ga-water-cluster-example>` | GA | xreac / ReaxFF |
| Bulk | {ref}`Cu₄ <ga-bulk-example>` | GA | EMT |
| Interface | {ref}`Copper oxide/Cu(111) <ga-surface-oxide-example>` | GA | MatterSim; fixed and variable composition |
| Interface | {ref}`Cu₄/α-Al₂O₃(0001) <ga-supported-nanoparticle-example>` | GA, BH, SA | User-selected; {ref}`short DPA4C-mini SA <sa-supported-nanoparticle-example>` |
| Interface | {ref}`CO–Cu₄/α-Al₂O₃(0001) <ga-supported-cluster-adsorbate-example>` | GA | MatterSim; random and site-insertion builders |
| Interface | {ref}`Water/TiO₂(101) <bh-tio2-water-rotation>` | BH | xreac; rotations and {ref}`bond changes <bh-tio2-water-bond-change>` |
| Interface | {ref}`CO/Cu(111) <co-cu111-example>` | SA | DPA4C |

```{toctree}
:maxdepth: 1
:hidden:

cluster
alloy-cluster
../bh/examples/cuox_tace
water-cluster
bulk
surface-oxide
supported-nanoparticle
supported-cluster-adsorbate
../bh/examples/tio2-water-rotation
co-cu111
```
