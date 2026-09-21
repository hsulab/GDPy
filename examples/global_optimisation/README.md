# Global optimisation examples

Choose an exploration and a runtime independently. Run commands from the GDPy
repository root so substrate paths resolve correctly.

- `explorations/genetic_algorithm/`: system construction, populations, genetic operators, and stopping criteria.
- `explorations/basin_hopping/`: system construction, populations, hopping moves, and stopping criteria.
- `runtimes/`: potential, executor, relaxation settings, and calculation constraints.
- `assets/`: shared substrate structures.
- `verify_cu4o4_bh_thanos.py`: verification for the extinction demonstration.

Exploration YAML files contain `method` and `recipe`; runtime YAML files contain
`schema_version`, `potential`, and `executor`. The existing `--runtime` option
joins them at execution time, so no YAML includes or copied exploration files
are needed. Put `--runtime` before the `explore` subcommand.

## Reuse one search with different runtimes

For the same Cu₄O₄ basin-hopping search with TACE:

```shell
OMP_NUM_THREADS=1 gdp -d ./run-cu4o4-tace \
  --runtime examples/global_optimisation/runtimes/tace_min_300.yaml \
  explore examples/global_optimisation/explorations/basin_hopping/cu4o4.yaml
```

Or with MatterSim:

```shell
OMP_NUM_THREADS=1 gdp -d ./run-cu4o4-mattersim \
  --runtime examples/global_optimisation/runtimes/mattersim_min_150.yaml \
  explore examples/global_optimisation/explorations/basin_hopping/cu4o4.yaml
```

Install optional models with `python -m pip install -e '.[tace,mattersim]'`.
Use a **new output directory when changing runtimes**; restarting an existing
search is for the same exploration and runtime. These two runtimes preserve
the original examples' different relaxation step limits; use matching executor
settings for a controlled potential comparison.

## Suggested pairings

Paths below are relative to this directory. Other chemically appropriate
runtimes can be substituted without changing the exploration.

| Exploration | Runtime |
| --- | --- |
| `explorations/genetic_algorithm/cu13.yaml` | `runtimes/emt_min_100.yaml` |
| `explorations/genetic_algorithm/cu7ni6.yaml` | `runtimes/emt_min_100.yaml` |
| `explorations/genetic_algorithm/cu4_bulk.yaml` | `runtimes/emt_min_100.yaml` |
| `explorations/genetic_algorithm/water4.yaml` | `runtimes/mattersim_min_100.yaml` |
| `explorations/genetic_algorithm/cu4o4_cu111.yaml` | `runtimes/mattersim_cu111_min_100.yaml` |
| `explorations/genetic_algorithm/cuxoy_cu111.yaml` | `runtimes/mattersim_cu111_min_100.yaml` |
| `explorations/genetic_algorithm/cu4_alumina111.yaml` | `runtimes/mattersim_alumina_min_20.yaml` |
| `explorations/genetic_algorithm/cu4_co_alumina111.yaml` | `runtimes/mattersim_alumina_min_100.yaml` |
| `explorations/genetic_algorithm/cu4_co_alumina111_adsorbate_insertion.yaml` | `runtimes/mattersim_alumina_min_100.yaml` |
| `explorations/basin_hopping/cu8.yaml` | `runtimes/emt_min_300.yaml` |
| `explorations/basin_hopping/cu4o4.yaml` | `runtimes/tace_min_300.yaml` |
| `explorations/basin_hopping/cu4o4_thanos.yaml` | `runtimes/mattersim_min_150.yaml` |

Runtime names end with the maximum relaxation step count. All use `fmax: 0.05`
eV/Å. The `alumina` variants fix the lowest 60 atoms; the `cu111` variant fixes
the lowest 4. Retain these constraints when adapting a slab runtime to another
potential. The 20-step alumina runtime is a short demonstration and may not
converge every structure.

EMT examples require no model download. MatterSim uses its 1M checkpoint;
TACE uses OAM-7M on CPU. Both download their model on first use. Choose a
potential that supports the system's elements; EMT is not an oxide or water
runtime. The Thanos verifier checks observed extinction/restart events, which
are not guaranteed to occur with a different potential.
