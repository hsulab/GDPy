# Biased Monte Carlo

`native/biased_monte_carlo` runs an independent chain for each input structure
with `gdp compute`. It minimises the initial structure and every valid proposal,
then applies the operator's acceptance rule to the minimised endpoint energies.
Accepted endpoints become the next chain state; rejection retains the previous
state. This is a biased search over relaxed structures, not equilibrium sampling
of the original potential. Reaching a minimiser's step limit does not guarantee
a local minimum.

Unlike population {doc}`basin hopping <../global_optimisation/basin_hopping>`,
this method has no population selection or generations. Conventional
{doc}`Monte Carlo <mc>` requires single-point energies and rejects enabled
geometry filters. Biased MC permits those filters and requires fixed-cell
minimisation. It cannot be used as an {doc}`HMC <hmc>` child.

## EMT example

```{literalinclude} ../../../examples/monte_carlo/biased.yaml
:language: yaml
```

From the repository root:

```shell
gdp -d run-biased-mc compute examples/monte_carlo/assets/cu32.xyz --runtime examples/monte_carlo/biased.yaml
```

EMT is bundled with ASE; no model file is required. This short example illustrates
the input and output workflow.

## Minimisation settings

Configure the evaluator under `parameters.setup.executor`. Its defaults are
`provider: ase` and `method: min`; both fields may be omitted. Only `min` is
accepted: `spc`, `md`, and `cmin` are rejected.

Choose the minimisation algorithm in the evaluator's
`parameters.setup.optimizer`:

```yaml
setup:
  executor:
    parameters:
      setup:
        optimizer:
          name: bfgs
          parameters:
            maxstep: 0.2
      stop:
        steps: 1000
        fmax: 0.05
```

This fragment belongs under the biased-MC executor's `parameters`. The ASE
adapter currently exposes `bfgs` for `min`, which is also its default. `maxstep`
is in Å. Available algorithms and their parameters depend on the evaluator
provider. See {doc}`optimizer configuration <../computations/tasks/relaxation>`
for the ordinary minimisation input.

Minimisation `stop.steps` must be an explicitly supplied positive integer, and
`stop.fmax` a finite positive force tolerance in eV/Å. The outer `stop.steps`
counts MC proposals, excluding initial minimisation. A minimiser that reaches
its iteration limit supplies its final endpoint for acceptance even if `fmax`
is not satisfied. Calculation failures remain failures; they are not accepted
as completed trials.

Set constraints in the evaluator's `parameters.setup.constraint`, using the
normal {doc}`constraint syntax <../groups-and-constraints>`. Input constraints
are retained when no override is supplied. Particle tags survive minimisation.
The evaluator inherits the outer potential and modifiers and cannot override
them. Its optional `scheduler` defaults to direct execution independently of
the parent scheduler. A different evaluator provider must support the shared
potential.

## Ensembles and filters

Use the shared `parameters.ensemble` mapping with `nvt`, `uvt`, or `suvt`.
Temperature and chemical potentials supply the existing operator acceptance
rules; they do not restore equilibrium sampling after minimisation. The same
operator/ensemble restrictions apply as in {doc}`ensembles`: species changes
require SUVT or UVT, and particle exchange requires UVT.

Biased MC uses the operators' existing filter defaults: distance checking is
enabled unless `skip_distance_check: true` is supplied. `covalent_ratio` controls
the distance limits where supported. `max_random_attempts` limits attempts to
construct a valid proposal; the example sets it to one. Once those attempts
are exhausted, the invalid proposal consumes an MC step without minimisation
and retains the previous state. See {doc}`operators/index` for operator-specific
filter behavior. Basin-hopping filters remain available independently.

## Output and restart

- `candN/mc.xyz`: initial relaxed endpoint and saved chain states, including
  repeated states after rejection or invalid proposals.
- `candN/mc_attempts.xyz`: initial input and valid proposed structures before
  minimisation.
- `candN/opstat.txt`: proposal outcomes and endpoint energies.
- `candN/calculations/step.NNNN/`: evaluator calculations and relaxation
  trajectories; step zero is initial minimisation. Intermediate calculation
  directories follow the MC trajectory cadence and may be pruned.
- `results/end_frames.xyz`: collected final chain endpoints.

Checkpoints preserve accepted structures, particle tags, constraints, and random
state. Repeating the same command resumes pending minimisations and completed
steps without replaying acceptance decisions. Use a fresh output directory when
changing the configuration.
