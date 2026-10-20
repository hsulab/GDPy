(global-optimisation-examples)=
(ga-examples)=
(bh-examples)=

# Examples

Choose a physical system, then a search method. Each system guide introduces
its geometry and shared setup once, followed by the available genetic algorithm
(GA), basin-hopping (BH), or simulated annealing (SA) recipes. Small populations
and short searches keep these demonstrations easy to inspect; increase search
and calculation budgets for production work.

Exploration inputs live under `examples/global_optimisation/explorations/`,
grouped by method. Each guide provides runtime settings and run commands.
Select a runtime with
`gdp --runtime <runtime.yaml> explore <exploration.yaml>`, placing `--runtime`
before `explore`. Use a new output directory when changing the runtime or search
settings. SA needs an MD cooling runtime; GA and BH use local relaxation.

All examples use direct scheduling by default. See {ref}`scheduler-transport`
for queue and SSH execution, and {ref}`exploration-output-layout` for outputs
and restart metadata. EMT needs no model download; other potentials may require
optional dependencies and model files. Short relaxation budgets may not reach
the force tolerance. Follow each surface guide's constraints when preparing a
surface study.

| Main system | Specific system | Methods | Potential / variants |
| --- | --- | --- | --- |
| Cluster | {ref}`Cu₈ <cu8-example>` | GA, BH, SA | EMT; {ref}`seeded BH <bh-seeded-cu8-example>` |
| Cluster | {ref}`Cu₇Ni₆ <ga-alloy-cluster-example>` | GA | EMT; swap mutation |
| Cluster | {ref}`Cu₆Ni₂ and Cu₄Ni₄ <bh-composition-broadcast-example>` | BH | EMT; composition sweep |
| Cluster | {ref}`Cu₆Niₓ <bh-variable-composition-example>` | BH | EMT; Ni insertion/removal |
| Cluster | Cu₄O₄ | BH | {ref}`TACE <bh-cuox-tace-example>`; {ref}`MatterSim with O–O extinction <bh-cuox-thanos-example>` |
| Cluster | {ref}`(H₂O)₄ <ga-water-cluster-example>` | GA | xreac / ReaxFF |
| Bulk | {ref}`Cu₄ <ga-bulk-example>` | GA | EMT |
| Interface | {ref}`Cu₄O₄/Cu(111) <ga-surface-oxide-example>` | GA | MatterSim |
| Interface | {ref}`CuₓOᵧ/Cu(111) <ga-variable-surface-oxide-example>` | GA | MatterSim; variable composition |
| Interface | {ref}`Cu₄/α-Al₂O₃(0001) <ga-supported-nanoparticle-example>` | GA, BH | User-selected |
| Interface | {ref}`CO–Cu₄/α-Al₂O₃(0001) <ga-supported-cluster-adsorbate-example>` | GA | MatterSim; random and site-insertion builders |
| Interface | {ref}`Water/TiO₂(101) <bh-tio2-water-rotation>` | BH | xreac; rotations and {ref}`bond changes <bh-tio2-water-bond-change>` |
| Interface | {ref}`CO/Cu(111) <co-cu111-example>` | SA | DPA4C |

```{toctree}
:maxdepth: 1
:hidden:

cluster
../bh/examples/seeded-cluster
alloy-cluster
../bh/examples/compositions
../bh/examples/variable-composition
../bh/examples/cuox_tace
../bh/examples/cuox_thanos
water-cluster
bulk
surface-oxide
variable-surface-oxide
supported-nanoparticle
supported-cluster-adsorbate
../bh/examples/tio2-water-rotation
../bh/examples/tio2-water-bond-change
co-cu111
```
