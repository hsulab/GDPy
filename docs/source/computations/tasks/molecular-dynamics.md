# molecular dynamics

Use `md` to propagate atomic positions and velocities. This short demo uses a periodic copper supercell and a Berendsen NVT thermostat.

See {doc}`ensembles <../molecular-dynamics/ensembles>` for NVE, NVT, and NPT
settings and the distinction between thermodynamic targets and regulators.

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:

ensembles <../molecular-dynamics/ensembles>
```

Generate the input files as described in {doc}`index`, then use
`examples/compute/tasks/molecular-dynamics.yaml`:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: md
  parameters:
    ensemble:
      method: nvt
      temperature: 300
    random_seed: 7
    setup:
      timestep: 1.0
      velocities:
        initialize: if_missing
        seed: 7
      regulator:
        name: berendsen
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

The ensemble temperature is in K; `timestep` and `Tdamp` are in fs. This
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
    ensemble:
      method: nvt
      temperature: 300
    setup:
      timestep: 1.0
      regulator:
        name: berendsen
        parameters: {}
    output:
      trajectory:
        period: 10
    stop:
      steps: 100
  broadcast:
    ensemble.temperature: [300, 600, 900]
```

This creates three workers in `w0`, `w1`, and `w2`. Lists that are not named
under `broadcast` remain ordinary executor parameters.

This is a short execution demo, not an equilibrated production trajectory.
`md-demo/results/end_frames.xyz` contains the final frame; per-calculation
`traj.xyz` files contain the saved trajectory.

To change machine resources, add a scheduler as described in
{doc}`../schedulers`.
