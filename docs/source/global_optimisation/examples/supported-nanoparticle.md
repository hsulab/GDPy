(ga-supported-nanoparticle-example)=

# Cu<sub>4</sub>/α-Al<sub>2</sub>O<sub>3</sub>(0001) supported cluster

Search for low-energy Cu₄ clusters on α-Al₂O₃(0001), the basal surface
corresponding to `(111)` in the rhombohedral setting. Both the genetic
algorithm (GA) and basin-hopping (BH) recipes below generate random Cu
clusters above the same substrate. Supply a runtime with a potential suitable
for Cu, Al, O, and their interfaces; the exploration inputs are independent
of the model. See the {doc}`potential guides <../../potentials/index>` for
installation, model files, and runtime configurations.

## Substrate and search region

`examples/global_optimisation/assets/alpha_alumina111_ortho.xyz` contains a
180-atom Al₇₂O₁₀₈ slab in a 14.28 × 16.49 × 34.0 Å orthogonal cell. A 3×2
repeat of the orthogonal surface unit gives nearly square rectangular periodic
boundaries and 12 Al atoms in the exposed top layer. The nine atomic planes
span z = 2.00–8.01 Å.

Cu atoms are generated inside a sphere of radius 1.8 Å centred at
(7.1385, 8.2428, 10.0) Å above the surface. Its top is at z = 11.8 Å,
leaving about 22.2 Å of upper vacuum. The sphere controls initial generation;
subsequent search moves can leave it.

Keep ASE tag 0 on all substrate atoms. The builder gives each Cu atom a
distinct positive tag so GA operators can distinguish the support from the
searchable cluster.

## Local relaxation runtime

Create `runtime.yaml` using a potential configuration from its provider guide
and the following executor block. This fixes the bottom 60 substrate atoms
while allowing the upper six planes and Cu atoms to relax:

```yaml
executor:
  provider: ase
  method: min
  parameters:
    setup:
      constraint: '`zbot 60`'
    stop:
      fmax: 0.05
      steps: 20
```

The 20-step limit keeps this demonstration short and may not reach the force
tolerance. Increase it for converged local minima. Use a separate output
directory whenever you change the model or search settings.

For reference, these models were tested on the 184-atom supported structure.
Times are median energy-and-force evaluations on one CPU thread after warmup,
excluding loading and compilation; they are not complete relaxation times.

| Model | Tested CPU path | Time per evaluation |
| --- | --- | ---: |
| {ref}`MACE-OMAT-0 Small <potential-mace>` | Compiled, float32 | 373.9 ms |
| {ref}`TACE-OMat24-7M <potential-tace>` | Compiled, scalar code generation | 6,104.7 ms |
| {ref}`DPA4-mini <potential-deepmd>` | CPU-exported `.pt2` | 601.1 ms |
| {ref}`DPA4C-mini <potential-deepmd>` | CPU-exported `.pt2`, scalar code generation | 29.2 ms |

Prophet-OAME-MBD took **156.8 s per energy-and-force evaluation** (median of
two warmed samples, 140.7–172.8 s) in a shorter one-thread CPU check on the
same structure after one 205.2 s warmup. This is an estimate rather than the ten-sample median protocol
used in the table. See {ref}`the Prophet timing details <prophet-cpu-timing>`.

See the {ref}`potential comparison <potential-cpu-inference-comparison>` for
the test protocol and model-specific setup.

## Genetic algorithm

The GA uses periodic cut-and-splice crossover and rattle mutation. Each Cu
atom has its own tag, so default fragment preservation keeps the atoms
independently movable.

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu4_alumina111.yaml
:language: yaml
```

Run from the repository root with your runtime:

```shell
gdp -d ./run-cu4-alumina111-ga --runtime ./runtime.yaml explore \
    ./examples/global_optimisation/explorations/genetic_algorithm/cu4_alumina111.yaml
```

`results/all_candidates.xyz` contains relaxed structures ordered by GA score;
`results/pop.png` shows their energies by generation.

(bh-supported-nanoparticle-example)=

## Basin hopping

The BH recipe relaxes four random supported structures, then starts two chains
with eight attempted moves each. Hopping proposals select only Cu atoms. The
500 K acceptance temperature controls the search, rather than an MD simulation.

```{literalinclude} ../../../../examples/global_optimisation/explorations/basin_hopping/cu4_alumina111.yaml
:language: yaml
```

Use the same local-relaxation runtime:

```shell
gdp -d ./run-cu4-alumina111-bh --runtime ./runtime.yaml explore \
    ./examples/global_optimisation/explorations/basin_hopping/cu4_alumina111.yaml
```

Inspect `candidates.db` for relaxed structures and acceptance metadata,
`results/history/gen0001.png` for the search history, and
`tmp_folder/gen1/rounds/events.jsonl` for committed rounds. Repeating the
command with the same output directory resumes the search.

### Prophet runtime

Prophet-OAME-MBD is the currently released standard Prophet energy/force
checkpoint. Upstream does not release a smaller standard checkpoint or report
a speed comparison between Prophet models, so no fastest-model ranking is
available. Download `prophet-oame-mbd.pt` from
[Kairos Materials' model repository](https://huggingface.co/kairosmaterial/prophet)
to `./models/prophet-oame-mbd.pt`, and install GDPy's `prophet` extra as described
in {doc}`../../potentials/prophet`.

This runtime fixes the bottom 60 substrate atoms and uses the same short
relaxation budget as the example above:

```{literalinclude} ../../../../examples/global_optimisation/runtimes/prophet_oame_mbd.yaml
:language: yaml
```

```shell
mamba run -n gdp3 gdp -d ./run-cu4-alumina111-bh-prophet \
    --runtime ./examples/global_optimisation/runtimes/prophet_oame_mbd.yaml explore \
    ./examples/global_optimisation/explorations/basin_hopping/cu4_alumina111.yaml
```

For accelerated inference on an NVIDIA GPU, install Prophet's optional CUDA
kernels and change the potential parameters to `device: cuda`,
`use_kernel: true`, and `use_compile: true`; see
{ref}`the Prophet acceleration instructions <prophet-supported-cluster>`.
Compilation adds first-evaluation overhead. The supplied CPU configuration
also runs on machines without CUDA. The CPU estimate above indicates that
this checkpoint is costly for this example on the measured hardware.

The compact populations and short search budgets demonstrate the workflow.
Converge the slab thickness, surface area, vacuum, relaxation budget, and search
length, and validate your potential for the Cu/alumina interface before
interpreting the resulting energy ordering.
