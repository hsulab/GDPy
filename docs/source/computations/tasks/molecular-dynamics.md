# molecular dynamics

Use `md` to propagate atomic positions and velocities. This short demo uses a periodic copper supercell and a Berendsen NVT thermostat.

Generate the input files as described in {doc}`index`, then use
`examples/compute/tasks/molecular-dynamics.yaml`:

```yaml
potential:
  provider: emt
  parameters: {}
executor:
  provider: ase
  method: md
  parameters:
    ensemble: nvt
    temp: 300
    timestep: 1.0
    steps: 100
    dump_period: 10
    velocity_seed: 7
    random_seed: 7
    controller:
      name: berendsen
      params:
        Tdamp: 100.0
```

From the repository root:

```shell
gdp -d md-demo -r examples/compute/tasks/molecular-dynamics.yaml compute examples/compute/tasks/md.xyz
```

`temp` is in K, `timestep` and `Tdamp` are in fs. This example runs 100 fs
and saves every ten steps. `velocity_seed` controls velocity initialization;
`random_seed` controls the executor’s random generator. Existing input
velocities are retained unless `ignore_atoms_velocities: true` is set.

This is a short execution demo, not an equilibrated production trajectory.
`md-demo/results/end_frames.xyz` contains the final frame; per-calculation
`traj.xyz` files contain the saved trajectory.

## Other ensembles

For NVE, set `ensemble: nve` and remove `controller`; gdpx uses velocity Verlet.
`temp` then controls initial velocities, not a thermostat target.

The registered ASE NPT configuration has this shape:

```yaml
executor:
  provider: ase
  method: md
  parameters:
    ensemble: npt
    temp: 300
    press: 1.0
    timestep: 1.0
    steps: 100
    dump_period: 10
    velocity_seed: 7
    controller:
      name: berendsen
      params:
        Tdamp: 100.0
        Pdamp: 1000.0
        compressibility: 0.000001
```

`press` is in bar. NPT requires a stress-capable potential and a suitable
periodic bulk structure. The current ASE Berendsen adapter has a known
compressibility conversion defect: it multiplies the supplied value by itself
before converting from inverse bar. The configuration above documents its
interface, but should not be used for quantitative NPT work until that adapter
is corrected. The runnable demo on this page uses NVT.

To change machine resources, add a scheduler as described in
{doc}`../schedulers`.
