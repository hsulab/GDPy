# Global optimisation examples

Choose an exploration and a runtime independently. Run commands from the GDPy
repository root so substrate paths resolve correctly.

- `explorations/genetic_algorithm/`: system construction, populations, genetic operators, and stopping criteria.
- `explorations/basin_hopping/`: system construction, populations, hopping moves, and stopping criteria.
- `runtimes/`: potential, executor, and relaxation settings.
- `assets/`: shared substrate structures.
- `verify_cu4o4_bh_thanos.py`: verification for the extinction demonstration.

Exploration YAML files contain `method`, `recipe`, and optionally `broadcast`; runtime YAML files contain
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

Paths below are relative to this directory. Other chemically appropriate
runtimes can be substituted without changing the exploration.

| Exploration | Runtime |
| --- | --- |
| `explorations/genetic_algorithm/cu13.yaml` | `runtimes/emt.yaml` |
| `explorations/genetic_algorithm/cu7ni6.yaml` | `runtimes/emt.yaml` |
| `explorations/genetic_algorithm/cu4_bulk.yaml` | `runtimes/emt.yaml` |
| `explorations/genetic_algorithm/water4.yaml` | `runtimes/mattersim.yaml` |
| `explorations/genetic_algorithm/cu4o4_cu111.yaml` | `runtimes/mattersim.yaml` |
| `explorations/genetic_algorithm/cuxoy_cu111.yaml` | `runtimes/mattersim.yaml` |
| `explorations/genetic_algorithm/cu4_alumina111.yaml` | `runtimes/mattersim.yaml` |
| `explorations/genetic_algorithm/cu4_co_alumina111.yaml` | `runtimes/mattersim.yaml` |
| `explorations/genetic_algorithm/cu4_co_alumina111_adsorbate_insertion.yaml` | `runtimes/mattersim.yaml` |
| `explorations/basin_hopping/cu8.yaml` | `runtimes/emt.yaml` |
| `explorations/basin_hopping/cu_ni_compositions.yaml` | `runtimes/emt.yaml` |
| `explorations/basin_hopping/cu4o4.yaml` | `runtimes/tace.yaml` |
| `explorations/basin_hopping/cu4o4_thanos.yaml` | `runtimes/mattersim.yaml` |

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

## Different compositions in one job

Broadcast a whole composition mapping to run independent Cu₆Ni₂ and Cu₄Ni₄
basin-hopping searches with EMT:

```yaml
method: basin_hopping
broadcast:
  population.builders.random.composition:
    - {Cu: 6, Ni: 2}
    - {Cu: 4, Ni: 4}
recipe:
  # See the complete recipe in cu_ni_compositions.yaml.
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

Broadcast keys are dotted paths relative to `recipe`; numeric components index
lists, such as `operators.0.temperature`. Each alternative replaces the whole
value at that path. Composition dictionaries therefore remain paired: splitting
Cu and Ni counts into separate broadcast fields would generate every combination.

Multiple fields form a Cartesian product in YAML field/value order, with the
rightmost field varying fastest. For example, adding `random_seed: [7, 17]` after
the composition field produces Cu6Ni2/7, Cu6Ni2/17, Cu4Ni4/7, Cu4Ni4/17.
Ordinary lists inside `recipe` retain their usual meaning. Parent paths must
exist, and broadcast paths cannot overlap.

Rerun the same command to resume. Use a fresh output directory when changing
broadcast values, their order, recipes, or runtimes. Generated inputs contain
resolved recipes without broadcast settings, so spawned jobs run one search.
