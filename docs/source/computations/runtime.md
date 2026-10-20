# runtime configurations

A runtime combines a `potential` with an `executor`. It may also define
`modifiers`, a `scheduler`, and a worker `dispatch` policy. The potential
supplies energies and forces; the executor selects the calculation and the
software interface used to run it.

Start with one complete runtime:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: min
  parameters:
    stop:
      fmax: 0.05
      steps: 300
```

Choose the guide that matches how the runtime will be used:

| Configuration | Use it for |
| --- | --- |
| {doc}`runtime-single` | One potential/executor pairing, including modifiers and provider compatibility. |
| {doc}`runtime-dispatch` | Batching structures and selecting batch or single-worker execution. |
| {doc}`runtime-broadcast` | Independent calculations created by varying executor parameters. |
| {doc}`runtime-chain` | Ordered workflow stages such as equilibration followed by production MD. |
| {doc}`schedulers` | Local execution, queue schedulers, SSH, and machine resources. |

```{toctree}
:hidden:
:maxdepth: 1

runtime-single.md
runtime-dispatch.md
runtime-broadcast.md
runtime-chain.md
ni111-water-restraint.md
```

In standalone runtime YAML, these fields are top-level; do not wrap them in a
`runtime` key. User-authored files omit `schema_version` and use the current
schema automatically. Saved runtime snapshots and compute plans record a
version so incompatible persisted data can be rejected.

Use the shared {doc}`units <../units>` unless a parameter explicitly documents
another unit.

Executor parameters are organized consistently across tasks:

- `random_seed` controls all stochastic behavior unless a subsection overrides it.
- `setup` selects the algorithm and initial calculation state.
- `output` controls trajectories and restart checkpoints.
- `stop` contains step limits and convergence criteria.

Molecular dynamics may add `setup.velocities` and `setup.regulator`; fixed-cell
and cell minimization may add `setup.optimizer`. Flat executor parameters remain
accepted for compatibility, but they cannot be mixed with the structured form.


## Ensemble settings

MD, conventional MC, and HMC use the same `executor.parameters.ensemble` mapping:

```yaml
ensemble:
  method: nvt
  temperature: 300
```

| Method | Required thermodynamic fields | Supported executors |
| --- | --- | --- |
| `nve` | None | MD |
| `nvt` | `temperature` | MD, MC, HMC |
| `npt` | `temperature`, `pressure` | MD |
| `uvt` | `temperature`, `chemical_potentials` | MC, HMC |
| `suvt` | `temperature`, `chemical_potentials` | MC, HMC |

Temperature is in K, pressure in bar, and chemical potentials in eV per particle.
Temperatures must be finite and positive; pressure and chemical potentials must
be finite. `chemical_potentials` maps particle/species names to values. `suvt`
preserves total particle count while allowing species changes.

Keep thermostat/barostat algorithms in `setup.regulator`, with `name` and
algorithm `parameters` only. For `nve`, initial velocity temperature may be set
in `setup.velocities.temperature`; it is not a thermostat target.

HMC declares the ensemble once on the parent. Its children inherit the applicable
settings and cannot redeclare them. MD segments use NVT for `uvt` and `suvt`
parents, while MC handles particle/species exchange. A broadcast at
`ensemble.temperature` changes the target for every child together.
