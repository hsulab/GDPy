# Global optimisation examples

Choose an exploration and a runtime independently. Run commands from the GDPy
repository root so substrate paths resolve correctly.

- `explorations/genetic_algorithm/`: system construction, populations, genetic operators, and stopping criteria.
- `explorations/basin_hopping/`: system construction, populations, hopping moves, and stopping criteria.
- `runtimes/`: potential, executor, and relaxation settings.
- `assets/`: shared substrate and seed structures.
- `verify_cu4o4_bh_thanos.py`: verification for the extinction demonstration.

Exploration YAML files contain `method: global_optimisation`, `population`,
`strategy`, shared search settings, and optionally `broadcast`; runtime YAML files contain
`schema_version`, `potential`, and `executor`. The existing `--runtime` option
joins them at execution time, so no YAML includes or copied exploration files
are needed. Put `--runtime` before the `explore` subcommand.

## Reuse one search with different runtimes

For the same Cu₄O₄ basin-hopping search with TACE:

```shell
OMP_NUM_THREADS=1 gdp -d ./run-cu4o4-tace \
  --runtime examples/global_optimisation/runtimes/tace.yaml \
  explore examples/global_optimisation/explorations/basin_hopping/cu4o4.yaml
```

Or with MatterSim:

```shell
OMP_NUM_THREADS=1 gdp -d ./run-cu4o4-mattersim \
  --runtime examples/global_optimisation/runtimes/mattersim.yaml \
  explore examples/global_optimisation/explorations/basin_hopping/cu4o4.yaml
```

Install optional models with `python -m pip install -e '.[tace,mattersim]'`.
Use a **new output directory when changing runtimes**; restarting an existing
search is for the same exploration and runtime. MatterSim and TACE both use a 20-step relaxation limit for quick demos.

## Suggested pairings

Examples share baseline systems across methods while covering different chemistry
and search capabilities. The GA and BH `cu8.yaml` recipes use the same Cu₈
composition, builder settings, random seed, initial population size, and EMT
runtime. Their method-specific operators and search budgets differ; these are
comparable workflow demonstrations, not equal-cost performance benchmarks.

Additional examples cover alloys, molecular clusters, bulk, supported clusters,
and variable composition. Reuse the same system and runtime when demonstrating
another method on an existing case; add a new system when it demonstrates a
distinct capability.

Paths below are relative to this directory. Other chemically appropriate
runtimes can be substituted without changing the exploration.

| Exploration | Runtime |
| --- | --- |
| `explorations/genetic_algorithm/cu8.yaml` | `runtimes/emt.yaml` |
| `explorations/genetic_algorithm/cu7ni6.yaml` | `runtimes/emt.yaml` |
| `explorations/genetic_algorithm/cu4_bulk.yaml` | `runtimes/emt.yaml` |
| `explorations/genetic_algorithm/water4.yaml` | `runtimes/mattersim.yaml` |
| `explorations/genetic_algorithm/cu4o4_cu111.yaml` | `runtimes/mattersim.yaml` |
| `explorations/genetic_algorithm/cuxoy_cu111.yaml` | `runtimes/mattersim.yaml` |
| `explorations/genetic_algorithm/cu4_alumina111.yaml` | `runtimes/mattersim.yaml` |
| `explorations/genetic_algorithm/cu4_co_alumina111.yaml` | `runtimes/mattersim.yaml` |
| `explorations/basin_hopping/cu8.yaml` | `runtimes/emt.yaml` |
| `explorations/basin_hopping/cu8_seeded.yaml` | `runtimes/emt.yaml` |
| `explorations/basin_hopping/cu_ni_compositions.yaml` | `runtimes/emt.yaml` |
| `explorations/basin_hopping/cu6_nix.yaml` | `runtimes/emt.yaml` |
| `explorations/basin_hopping/cu4o4.yaml` | `runtimes/tace.yaml` |
| `explorations/basin_hopping/cu4o4_thanos.yaml` | `runtimes/mattersim.yaml` |

The CO–Cu₄/alumina example combines `random` and `site_insertion` initialization
in one population (four candidates each). Its family tree labels both initial
groups so their relaxed energies and descendants can be compared.

There is one runtime per potential: `emt.yaml` allows up to 1,000 relaxation
steps; `mattersim.yaml` and `tace.yaml` allow up to 20. All use `fmax: 0.05`
eV/Å and may stop earlier when the force tolerance is met. The 20-step limits
keep demos short and do not guarantee force-converged minima.

These shared runtimes have no atom constraints: all atoms, including substrate
atoms, can relax. For a surface study, customize `executor.parameters.constraint`
as appropriate, for example `lowest 60` for the alumina slabs or `lowest 4` for
the Cu(111) slab. Increase the relaxation limit for production calculations.

EMT examples require no model download. MatterSim uses its 1M checkpoint;
TACE uses OAM-7M on CPU. Both download their model on first use. Choose a
potential that supports the system's elements; EMT is not an oxide or water
runtime. The Thanos verifier checks observed extinction/restart events, which
are not guaranteed to occur with a different potential.

## Random and seeded initialization

`explorations/basin_hopping/cu8_seeded.yaml` combines two random Cu₈ candidates
with two structures read from `assets/cu8_seeds.xyz`. The `seeds` builder uses
`method: direct` and `indices: [0, 1]`; its initial allocation matches the two
selected frames. All four candidates are relaxed before chain-start selection.

```shell
gdp -d ./run-cu8-seeded-bh-emt \
  --runtime examples/global_optimisation/runtimes/emt.yaml explore \
  examples/global_optimisation/explorations/basin_hopping/cu8_seeded.yaml
```

The seed geometries are unrelaxed, slightly perturbed cube and square-antiprism
clusters with a 20 Å periodic cell and positive per-atom tags. Replace the file
and frame indices to use your own seeds, adjusting `initial.total_size` and
`initial.builder_allocations` to match. Use a new output directory when changing
seeds. The candidate database records each initial candidate's builder name.

## Different compositions in one job

Broadcast a whole composition mapping to run independent Cu₆Ni₂ and Cu₄Ni₄
basin-hopping searches with EMT:

```yaml
method: global_optimisation
strategy:
  method: basin_hopping
broadcast:
  population.builders.random.composition:
    - {Cu: 6, Ni: 2}
    - {Cu: 4, Ni: 4}
# See the complete configuration in cu_ni_compositions.yaml.
```

Run the complete example from the repository root:

```shell
OMP_NUM_THREADS=1 gdp -d run-cu-ni-compositions \
  --runtime examples/global_optimisation/runtimes/emt.yaml \
  explore examples/global_optimisation/explorations/basin_hopping/cu_ni_compositions.yaml
```

Each search uses seed 7, two initial candidates, and two hopping rounds. Both
species participate in the move operator. The searches run sequentially in the
same process with independent calculation workers and checkpoints.

For one Slurm allocation containing both searches:

```shell
sbatch examples/global_optimisation/two_explorations.slurm
```

Activate your GDPy environment first and adapt account/partition settings for
your cluster. The supplied job requests one CPU and 30 minutes. Exploration and
calculation scheduling are direct, so no additional queue jobs are submitted.

```text
run-cu-ni-compositions/
├── _meta/
│   ├── _scheduler.json       # provider, layout, and job records
│   └── ...resolved inputs and scripts...
├── gdp.out
├── expo.00/                 # Cu6Ni2
│   └── ...candidates.db, results, and checkpoints...
└── expo.01/                 # Cu4Ni4
    └── ...candidates.db, results, and checkpoints...
```

Broadcast keys are dotted paths into top-level search settings; numeric components
index lists, such as `strategy.operators.0.temperature`. Each alternative replaces the whole
value at that path. Composition dictionaries therefore remain paired: splitting
Cu and Ni counts into separate broadcast fields would generate every combination.

Multiple fields form a Cartesian product in YAML field/value order, with the
rightmost field varying fastest. For example, adding `random_seed: [7, 17]` after
the composition field produces Cu6Ni2/7, Cu6Ni2/17, Cu4Ni4/7, Cu4Ni4/17.
Ordinary lists in the search configuration retain their usual meaning. Parent paths must
exist, and broadcast paths cannot overlap.

Rerun the same command to resume. Use a fresh output directory when changing
broadcast values, their order, recipes, or runtimes. Generated inputs contain
resolved recipes without broadcast settings, so spawned jobs run one search.

## Variable composition within one search

`cu6_nix.yaml` keeps six Cu atoms and exchanges Ni atoms during basin hopping:

```shell
OMP_NUM_THREADS=1 gdp -d run-cu6-nix \
  --runtime examples/global_optimisation/runtimes/emt.yaml \
  explore examples/global_optimisation/explorations/basin_hopping/cu6_nix.yaml
```

The builder samples one to three Ni atoms initially. An `exchange` operator
attempts insertion/removal of Ni with chemical potential −0.5 eV, mixed with
ordinary Cu/Ni displacement moves. The population uses formation-energy ranking
with the same Ni reference. These are illustrative chemical potentials for a
short EMT demonstration. The initial composition range is not a constraint on
later moves: Ni can disappear or exceed three atoms, while Cu remains fixed.

This single exploration writes directly into `run-cu6-nix/`. Inspect
`candidates.db` for initial and trial compositions, acceptance flags, and parent
IDs, and `tmp_folder/gen1/rounds/events.jsonl` for committed chain history.
See the basin-hopping documentation's **Variable-composition Cu₆Niₓ clusters**
guide for a script that identifies accepted and rejected composition changes.
