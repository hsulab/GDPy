(monte-carlo)=

# monte carlo (mc)

## Overview

MC is a conventional method to explore the configuration space.

This implementation remains an exploration method. Its worker determines
whether trial configurations undergo single-point evaluation, minimization,
or dynamics. The minimization example below is a search workflow, not a claim
of equilibrium canonical sampling. Some biased moves use heuristic acceptance
rules. Ensemble validation and migration to an MC executor are separate work.

Population-based basin hopping is a separate {doc}`../global_optimisation/basin_hopping` method.
Both methods reuse moves and acceptance rules from `gdpx.exploration.sampling`; BH does
not inherit from MC. Moves use reversible in-place edits rather than copying
the whole structure on every attempt.

## Example

The related commands are

```shell
# - explore configuration space defined by `config.yaml`
#   results will be written to the `results` folder
#   a log file will be written to `results/gdp.out` as well
$ gdp -d exp -r runtime.yaml explore ./config.yaml

# - after MC is converged i.e. reaches the maximum number of steps,
#   the MC trajectory is stored at `results/mc.xyz`
```

In the `recipe.operators` section,

Every MC operator has parameters of `temperature`, `pressure`, and `region`. In general,
these three parameters should be consistent among different operators used in the simulation.
Otherwise, the simulation may not converge the structure to the phyiscal equilibrium.

To increase the acceptance, `convalent_ratio` is often set to check if the new structure has
too small or too large distances. The two values are the minimum and the maximum coefficients,
which will be multipied by the covalent bond distance.

See {ref}`region-definitions` for more information about defining a region.

- move:

  > Move a particle to a random position with maximum `max_disp` displacement.

- swap:

  > Swap the positions of two particles from two different types.

- rattle:

  > Perturb several particles in one proposal. Each eligible particle is selected
  > independently with `rattle_prop` (default `0.4`). Each displacement component
  > is uniform between `-rattle_strength` and `+rattle_strength` Angstrom
  > (default `0.8`), following the GA rattle convention rather than a Gaussian.
  > Tagged molecular particles translate rigidly without rotation. Eligibility
  > follows the same `particles` and `region` rules as `move`.

- exchange:

  > Exchange particles with an imaginary reservoir by inserting or removing. This
  > changes the number of atoms in the system as it samples the grand canonical
  > ensemble.

:::{note}
In general, operators should have the same region. Otherwise, the simulation is
not converged to an equilibrium.
:::

In the `recipe.convergence` section,

- steps: Number of MC steps.

Since MC usually takes ~5000 steps, the `dump_period` determines what MC step
will be saved. For example, if `dump_period = 2`, step 0, 2, 4 ... will be saved.
These saved structures and trajectories can be used for MLIP training.

The input file shown below explores the oxidation of Cu(111) surface. The MC operators
only apply to atoms in the surface region including Cu and O.

```yaml
method: monte_carlo
recipe:
  random_seed: 1112
  builder:
    method: read_stru
    fname: ./fcc-s111p44.xyz
  operators:
    - method: exchange
      region:
        method: lattice
        origin: [0, 0, 8.0]
        cell: [10.17, 0, 0, 0, 8.81, 0, 0, 0, 6.0]
      covalent_ratio: [0.8, 2.0]
      particles: [O]
      chempots: [-5.75]
      temperature: 800
      probability: 0.5
    - method: move
      particles: [Cu, O]
      region:
        method: lattice
        origin: [0, 0, 8.0]
        cell: [10.17, 0, 0, 0, 8.81, 0, 0, 0, 6.0]
      covalent_ratio: [0.8, 2.0]
      max_disp: 2.0
      temperature: 800
      probability: 0.5
  convergence:
    steps: 5
  dump_period: 1
```

Use a single-worker runtime for sequential Monte Carlo moves:

```yaml
potential:
  provider: deepmd
  parameters:
    command: lmp -in in.lammps 2>&1 > lmp.out
    type_list: [Cu, O]
    model:
      - ./graph.pb
executor:
  provider: lammps
  method: min
  parameters:
    ignore_convergence: false
    fmax: 0.05
    steps: 400
options:
  worker: single
```

## Rattle example

For collective rattle moves in either MC or BH, use this operator entry:

```yaml
operators:
  - method: rattle
    particles: [Cu]
    rattle_strength: 0.2
    rattle_prop: 0.4
    temperature: 500.0
    probability: 1.0
```

Rattle retries empty selections and invalid geometries up to
`max_random_attempts`. Exhausted proposals are skipped. It uses the existing
distance settings and energy-based MC acceptance rule.
Use distinct atom tags for independent atomic particles;
atoms sharing a tag form one particle. Position edits are reversible and do not
copy the complete structure.

## Checkpoints and restart

Restart with the same configuration and output directory. MC stores versioned
JSON metadata and uncompressed NumPy `.npz` arrays, without pickle. Arrays,
constraints, cached results, operator configuration, and random state are
preserved without copying the entire structure before serialization.

`ckpt_period` controls ordinary checkpoint frequency. Only the latest and
previous committed checkpoints are retained; a damaged latest snapshot falls
back to the previous one. A queued move also retains its pending state until
its resolution is committed, forcing a checkpoint even between ordinary
checkpoint steps. Hybrid MC commits this state after its complete procedure.
Older pickle checkpoints are not loaded; use a new output directory for those
runs.

## Application

1. {ref}`ref-acs-catal-2022-xu`
