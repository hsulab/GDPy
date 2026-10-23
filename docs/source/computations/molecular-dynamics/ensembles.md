# Ensembles

MD defines its ensemble under `executor.parameters.ensemble`. The regulator
selects the thermostat or barostat algorithm; its numerical parameters belong
under `setup.regulator.parameters`.

| Method | Required ensemble fields | MD behavior |
| --- | --- | --- |
| `nve` | None | Integrate without a thermostat or barostat |
| `nvt` | `temperature` | Control temperature at fixed particle count and cell |
| `npt` | `temperature`, `pressure` | Control temperature and pressure with a changing cell |

Temperature is in K and must be finite and positive. Pressure is in bar and
must be finite. MD does not support `uvt` or `suvt` by itself; see
{doc}`Monte Carlo ensembles <../monte-carlo/ensembles>` for particle and
species exchange, including HMC inheritance.

The examples below are executor fragments for a runtime with one potential.

## NVE

```yaml
executor:
  provider: ase
  method: md
  parameters:
    ensemble:
      method: nve
    setup:
      timestep: 1.0
      velocities:
        temperature: 300
        initialize: if_missing
    stop:
      steps: 100
```

Omit the regulator for NVE. `setup.velocities.temperature` sets the initial
velocity distribution; it is not a thermostat target. Existing nonzero
velocities are retained with `initialize: if_missing`. ASE uses velocity Verlet.

## NVT

```yaml
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
        name: langevin
        parameters:
          friction: 0.01
    stop:
      steps: 100
```

Specify temperature once in the ensemble block. The regulator contains only
its algorithm name and parameters, without a separate `targets` mapping.
Supported regulator algorithms depend on the executor provider. The
{doc}`MD demo <../tasks/molecular-dynamics>` uses ASE Berendsen dynamics.

## NPT

The registered ASE NPT configuration has this shape:

```yaml
executor:
  provider: ase
  method: md
  parameters:
    ensemble:
      method: npt
      temperature: 300
      pressure: 1.0
    setup:
      timestep: 1.0
      velocities:
        initialize: if_missing
        seed: 7
      regulator:
        name: berendsen
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
is corrected. The linked MD demo uses NVT.


## Temperature sweeps

Use `executor.broadcast` with `ensemble.temperature` to run independent
trajectories at several temperatures. See {doc}`../runtime-broadcast` for
complete examples. For cooling trajectories managed by structure search,
see {doc}`simulated annealing <../../global_optimisation/simulated-annealing>`.
