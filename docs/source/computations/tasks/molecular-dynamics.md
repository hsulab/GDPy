# molecular dynamics

Use `md` to propagate atomic positions and velocities. This short demo uses a periodic copper supercell and a Berendsen NVT thermostat.

Generate the input files as described in {doc}`index`, then use
`examples/compute/tasks/molecular-dynamics.yaml`:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: md
  parameters:
    random_seed: 7
    setup:
      ensemble: nvt
      timestep: 1.0
      velocities:
        initialize: if_missing
        seed: 7
      regulator:
        name: berendsen
        targets:
          temperature: 300
        parameters:
          Tdamp: 100.0
    output:
      trajectory:
        period: 10
    stop:
      steps: 100
```

From the repository root:

```shell
gdp -d md-demo -r examples/compute/tasks/molecular-dynamics.yaml compute examples/compute/tasks/md.xyz
```

The regulator temperature is in K; `timestep` and `Tdamp` are in fs. This
example runs 100 fs and saves every ten steps. `velocities.seed` controls
velocity initialization; the parameters-level `random_seed` controls the
executor’s random generator. `initialize: if_missing` retains existing nonzero
velocities.

To run the same calculation independently at several temperatures, add an
explicit executor broadcast. Broadcast keys are paths relative to
`executor.parameters`:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: md
  parameters:
    setup:
      ensemble: nvt
      timestep: 1.0
      regulator:
        name: berendsen
        targets:
          temperature: 300
        parameters: {}
    output:
      trajectory:
        period: 10
    stop:
      steps: 100
  broadcast:
    setup.regulator.targets.temperature: [300, 600, 900]
```

This creates three workers in `w0`, `w1`, and `w2`. Lists that are not named
under `broadcast` remain ordinary executor parameters.

This is a short execution demo, not an equilibrated production trajectory.
`md-demo/results/end_frames.xyz` contains the final frame; per-calculation
`traj.xyz` files contain the saved trajectory.

## Other ensembles

For NVE, set `setup.ensemble: nve`, omit the regulator, and put the initial
temperature under `setup.velocities`; gdpx uses velocity Verlet.

The registered ASE NPT configuration has this shape:

```yaml
executor:
  provider: ase
  method: md
  parameters:
    setup:
      ensemble: npt
      timestep: 1.0
      velocities:
        initialize: if_missing
        seed: 7
      regulator:
        name: berendsen
        targets:
          temperature: 300
          pressure: 1.0
        parameters:
          Tdamp: 100.0
          Pdamp: 1000.0
          compressibility: 0.000001
    output:
      trajectory:
        period: 10
    stop:
      steps: 100
```

The pressure target is in bar. NPT requires a stress-capable potential and a
suitable periodic bulk structure. The current ASE Berendsen adapter has a known
compressibility conversion defect: it multiplies the supplied value by itself
before converting from inverse bar. The configuration above documents its
interface, but should not be used for quantitative NPT work until that adapter
is corrected. The runnable demo on this page uses NVT.

To change machine resources, add a scheduler as described in
{doc}`../schedulers`.
