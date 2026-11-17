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

### Optional graceful stopping under Slurm

For the native VASP `min` executor, configure graceful stopping explicitly:

```yaml
executor:
  provider: vasp
  method: min
  parameters:
    setup:
      graceful_stop:
        enabled: false
        margin_seconds: 7200
    stop:
      fmax: 0.05
      steps: 100
```

`setup.graceful_stop.enabled` defaults to `false`. Set it to `true` to request an ionic-step
stop by writing `LSTOP = .TRUE.` into STOPCAR before the Slurm allocation ends.
The positive, finite margin in seconds defaults to 7200 and must allow the
current ionic step and checkpoint writes to finish. This is a fixed margin,
not an estimate from SCF timings. Enable `lwave` and `lcharg` in the potential
parameters to save the electronic state.

Slurm's job metadata supplies the actual allocation deadline. An enabled run
fails before launching VASP if the deadline is unavailable or less than the
margin remains. The old `GDPX_VASP_STOP_MARGIN_SECONDS` environment switch is
not read; all tuning is through the executor input.

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
