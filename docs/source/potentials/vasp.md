(potential-vasp)=

# vasp

The `vasp` provider configures VASP electronic settings and pseudopotentials.

## requirements

Provide a VASP executable, pseudopotential directory, and INCAR. The ASE interface additionally requires `vasp_interactive`.

## backends

| Potential backend | Executor | Ionic motion |
| --- | --- | --- |
| `vasp` | `vasp` | VASP performs minimization or MD. |
| `interactive` | `ase` | ASE controls minimization or MD through an interactive VASP calculator. |

Each executor defaults to the backend shown above. The examples specify
`potential.backend` explicitly; it may be omitted. Backend names are scoped
to the potential provider. The Python package remains `vasp_interactive`.

Additional potential parameters are passed to the corresponding calculator;
`vdw_path` sets the van der Waals kernel directory when needed.

## configurations

### vasp + vasp

```yaml
potential:
  provider: vasp
  backend: vasp  # Optional; default for the vasp executor.
  parameters:
    command: mpirun -n 32 vasp_std
    pp_path: /path/to/potentials
    incar: ./INCAR
    kpts: [1, 1, 1]
executor:
  provider: vasp
  method: min
  parameters:
    stop:
      fmax: 0.05
      steps: 100
```

### Shared inputs on restart

The first run writes ordinary VASP inputs. When GDP creates the first checkpoint,
`0000.run` retains the real `POTCAR`, `KPOINTS`, and `ase-sort.dat`. Subsequent
restarts generate a regular `INCAR`, copy the preceding `CONTCAR` to a regular
`POSCAR`, and use relative links for the unchanged inputs:

```text
POTCAR                -> 0000.run/POTCAR
KPOINTS               -> 0000.run/KPOINTS
ase-sort.dat          -> 0000.run/ase-sort.dat
0001.run/POTCAR       -> ../0000.run/POTCAR
0001.run/KPOINTS      -> ../0000.run/KPOINTS
0001.run/ase-sort.dat -> ../0000.run/ase-sort.dat
```

All four usual VASP input names remain in the working directory, so a manual
VASP invocation uses the same inputs as GDP. Restarts initialize ASE and call
its INCAR writer directly; they do not call its full input writer or regenerate
POTCAR, KPOINTS, or the sorting file. Calculations using KSPACING retain their
existing convention of having no KPOINTS file.

Before reusing files, GDP checks pseudopotential contents, k-point input,
species ordering, and ASE's sort/resort mapping against the first run.
Incompatible settings require a new calculation directory. When creating a
checkpoint, GDP adjusts the links for the new directory depth and converts
identical regular copies in older checkpoints into links; differing historical
inputs are retained.

Keep `0000.run` with the calculation: it owns the shared inputs and must not be
deleted while other runs reference it. Move or archive the whole calculation
directory to preserve the relative links. GDP does not prune numbered VASP
checkpoints. WAVECAR and CHGCAR remain separate writable restart files.

### Optional graceful stopping under Slurm

For the native VASP `min` executor, configure the OUTCAR watcher explicitly:

```yaml
executor:
  provider: vasp
  method: min
  parameters:
    setup:
      graceful_stop:
        enabled: false
        margin_seconds: 7200
        poll_interval_seconds: 30
        checkpoint_margin_seconds: 300
        safety_factor: 1.5
        window: 5
        min_samples: 2
    stop:
      fmax: 0.05
      steps: 100
```

The block is optional and `enabled` defaults to `false`. When enabled, GDP's
background thread reads only newly appended OUTCAR data, logging each completed
geometry's full SCF-cycle wall time from `LOOP+`. It ignores the first cycle
of each VASP invocation and keeps the latest `window` subsequent timings.

Until `min_samples` timings are available, `margin_seconds` is the fallback
reserve before the Slurm deadline. After that, the reserve is:

```text
max(safety_factor * recent_mean,
    recent_maximum,
    safety_factor * elapsed_since_last_cycle_observed)
+ checkpoint_margin_seconds + poll_interval_seconds
```

When remaining time is at or below this reserve, GDP writes `LSTOP = .TRUE.`
to STOPCAR. This lets VASP finish an ionic step and write its checkpoint.
The elapsed-time term increases the reserve when the current cycle runs longer
than the recent history. The first cycle is excluded because cold-start SCF
costs can be much larger than later cycles. `LOOP:` electronic-iteration timings
are not used as full-cycle samples.

All defaults are shown above. Durations must be finite and positive,
`safety_factor` must be at least 1, and the positive integer `min_samples`
must not exceed `window`. Set `lwave` and `lcharg` in the potential parameters
to save the electronic state. Choose a fallback and write allowance appropriate
for the system; runtime estimates cannot guarantee completion of an unexpectedly
slow SCF or checkpoint write.

Slurm job metadata supplies the allocation deadline. An enabled run fails
before launching VASP if the deadline is unavailable or less than the fallback
margin remains. Missing or unreadable OUTCAR data uses the fallback reserve;
file replacement or truncation resets the timing history. The watcher stops
when the VASP subprocess exits and removes only its own unchanged STOPCAR.
There are no environment-variable tuning switches.

### interactive + ase

This backend additionally requires the `vasp_interactive` Python package.

```yaml
potential:
  provider: vasp
  backend: interactive  # Optional; default for the ase executor.
  parameters:
    command: mpirun -n 32 vasp_std
    pp_path: /path/to/potentials
    incar: ./INCAR
    kpts: [1, 1, 1]
executor:
  provider: ase
  method: min
  parameters:
    stop:
      fmax: 0.05
      steps: 100
```

### interactive + ase with dft-d3

Add external DFT-D3 through a runtime modifier:

```yaml
potential:
  provider: vasp
  backend: interactive  # Optional; default for the ase executor.
  parameters:
    command: mpirun -n 32 vasp_std
    pp_path: /path/to/potentials
    incar: ./INCAR
modifiers:
  - provider: dftd3
    backend: ase  # Optional; default for the dftd3 modifier.
    parameters:
      method: PBE
      damping: d3bj
executor:
  provider: ase
  method: min
  parameters:
    stop:
      fmax: 0.05
      steps: 100
```

This requires `dftd3.ase`. The modifier adds its energy and forces to the VASP
results; VASP is paused while modifiers run. Do not also enable the same
dispersion correction inside VASP.

Migration: replace `parameters.interface: vasp_interactive` with
`potential.backend: interactive`. The `vasp_interactive_disp` backend and
`parameters.dispersion` section are removed; move dispersion parameters into
the `dftd3` modifier above, omitting the old `type: dftd3` field.
