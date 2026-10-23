(hybrid-monte-carlo)=

# Hybrid Monte Carlo

`hybrid_monte_carlo` alternates worker calculations and blocks of MC proposals.
The examples combine short NVT molecular-dynamics segments with Cu
displacements or Cu/Ni identity changes using EMT, and oxygen exchange on
Cu(111) using xreac. Their short trajectories and small cells illustrate the
workflow rather than equilibration.

GDPy accepts the final structure of each MD segment directly. Only the
subsequent MC proposals undergo a Metropolis test using potential-energy
changes. There is no accept/reject test of the complete MD trajectory based on
total Hamiltonian change, so this implementation is a mixed MD/MC workflow,
not a Hamiltonian Monte Carlo sampler. The {ref}`MC sampling limitations
<monte-carlo>` also apply to its MC operators.

See the {ref}`MC operator reference <sampling-operators>` for available moves and
their configuration, particle selection, and acceptance rules.
Declare `parameters.ensemble` once on the HMC parent. Children inherit it,
along with the outer potential and modifiers. HMC supports `nvt`, `uvt`, and
`suvt`. For `uvt` and `suvt`, MD runs NVT at the inherited temperature and
current composition; MC handles particle exchange or species changes. Child
ensemble declarations, regulator targets, and temperature/pressure ramps are
rejected. Thermostat algorithms and numerical settings remain in MD's regulator.

## Examples

This complete canonical input alternates 20 NVT MD steps and five Cu displacement
proposals per cycle, for five cycles at 1200 K. It uses ASE EMT and the Cu32
structure in `examples/monte_carlo/assets/cu32.xyz`.

```{literalinclude} ../../../../examples/monte_carlo/hybrid-canonical.yaml
:language: yaml
:caption: examples/monte_carlo/hybrid-canonical.yaml
```

Run from the repository root:

```shell
gdp -d run-mc-hybrid-canonical compute examples/monte_carlo/assets/cu32.xyz --runtime examples/monte_carlo/hybrid-canonical.yaml
python examples/monte_carlo/inspect_run.py run-mc-hybrid-canonical/cand0
```

The example pages explain {doc}`canonical displacement <hmc/examples/canonical>`,
{doc}`semi-grand-canonical identity changes <hmc/examples/semi-grand-canonical>`,
and {doc}`Cu(111) oxidation with xreac <hmc/examples/cu111-oxidation-xreac>`.

```{toctree}
:maxdepth: 2
:hidden:

hmc/examples/index
```

## Configure the cycle

Run `gdp compute INPUT --runtime hybrid.yaml` with `executor.provider: native`
and `executor.method: hybrid_monte_carlo`. Each input starts an independent chain.

`executor.parameters.setup` is an ordered **list of executor definitions**.
Each child has `provider`, `method`, `parameters`, and an optional `scheduler`.
Include at least one MD child and one `native/monte_carlo` child. Repeated stages
are supported. Potential, modifiers, complete runtimes, and dispatch settings
cannot be overridden in children.

- MD owns its integration settings. Its `parameters.stop.steps` counts MD steps.
- Standard MC owns its operators, evaluator, and proposal count.
  Its `parameters.stop.steps` counts MC proposals. Evaluation defaults to ASE
  single-point calculation; use its optional `setup.evaluation` to override it.
- The parent `parameters.stop.steps` counts full passes through the setup list.
  Parent early stopping is checked after a complete cycle. An early-stopped
  child completes its invocation and passes its endpoint to the next child.

MD endpoints are accepted directly. Structures, cell, constraints, and momenta
pass between children. MD and MC use the same inherited
ensemble temperature. Each child scheduler defaults to
`direct`; the parent scheduler controls orchestration. The potential backend
is resolved once and must support all children.

The parent's `random_seed`, child seeds, cycle, and child index determine
reproducible invocation seeds. Repeated children use separate random streams
and working directories. Existing nonzero momenta carry into subsequent MD
segments unless the MD child explicitly requests fresh velocities.

## Outputs and restart

For a five-cycle run with `output.trajectory.period: 1`, each `candN` contains:

- `mc.out`: cycle progress and inline MD/MC evaluation diagnostics, appended on
  resume. The terminal and top-level `gdp.out` retain batch status and summaries.
- `mc.xyz`: six frames, the initial state and five complete-cycle endpoints.
- `calculations/step.0000/`: initial single-point evaluation using the first
  MC child's evaluator.
- `calculations/step.0001/procedure.0000/`: the first MD invocation.
- `calculations/step.0001/procedure.0001/cand0/`: the first MC invocation,
  including its own `mc.xyz`, `mc_attempts.xyz`, and `opstat.txt` decision log.
  Its `calculations/step.NNNN/` directories hold individual energy evaluations.

For the example, each MC invocation records five decisions in `opstat.txt`.
There is no parent-level `mcmoves.log`. Child output settings control child
trajectories independently of the parent's cycle trajectory.

Composite checkpoints commit child boundaries even between requested checkpoint
periods, so pending calculations resume without replaying completed children.
`output.checkpoint.keep` bounds retained snapshots. Initial and final states
are always included, including with sparse trajectory output. Collected chain
endpoints appear in `results/end_frames.xyz`.

Resume with the same command, directory, and inputs. Changed inputs or old
`gdpx`/`setup.cycle` runs require a new directory. Move the ensemble to parent `parameters.ensemble`, keep operators in the
standard MC child, and list children directly in `setup`.
