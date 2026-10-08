(potential-mace)=

# mace

The `mace` provider loads local MACE checkpoints.

## Requirements

From the repository root:

```shell
python -m pip install -e '.[mace]'
```

The extra installs `mace-torch` and PyTorch for inference and training,
including `mace_run_train`. See {doc}`../installation` for CPU/GPU setup,
verification, and dependency conflicts with other providers.

LAMMPS requires a separate binary with the MACE pair style and a compatible
exported model.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | MACE Python calculator. |
| `lammps` | `lammps` | Yes | LAMMPS with an exported MACE model. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### ase + ase

```yaml
potential:
  provider: mace
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model: ./mace.model
    type_list: [H, O]
    precision: float32
    estimate_uncertainty: false
executor:
  provider: ase
  method: spc
```

### lammps + lammps

```yaml
potential:
  provider: mace
  backend: lammps  # Optional; default for the lammps executor.
  parameters:
    model: ./mace-lammps.pt
    type_list: [H, O]
    command: lmp
executor:
  provider: lammps
  method: spc
```

### Parameter notes

`model` accepts an existing path or list of paths. The ASE adapter passes
`precision` (default `float32`) as `default_dtype`, selecting CUDA if available
and CPU otherwise. Multiple models with `estimate_uncertainty: true` enable
a committee; otherwise only the first is used. This adapter expects local
files rather than foundation-model names.

The LAMMPS interface requires exactly one exported model; `command` defaults
to `lmp`.

See {doc}`../trainers/mace` for training.

### MACE-OMAT-0 Small runtime

Supply your own checkpoint at `./models/mace-omat-0-small.model`, or update
`potential.parameters.model` in this reusable local-relaxation runtime:

```{literalinclude} ../../../examples/global_optimisation/runtimes/mace_omat_small.yaml
:language: yaml
```

Use it with any exploration whose elements and chemistry are supported by the
model. Add system-specific constraints to your own runtime copy when needed.
This runtime uses GDPy's standard ASE path. The compiled timings below were
measured with the upstream calculator; GDPy's current MACE adapter does not
forward `compile_mode`.

## Foundation-model CPU inference

Local checkpoints from the
[official MACE foundation registry](https://github.com/ACEsuit/mace-foundations)
were tested on the {ref}`shared 184-atom CPU fixture <potential-cpu-inference-comparison>`
with float32, one thread, and dispersion and specialized acceleration disabled.
Each timing excludes loading and compilation. All outputs were finite.

| Checkpoint | Standard CPU | `compile_mode="default"` |
| --- | ---: | ---: |
| MP-0a Small | 895.0 ms | — |
| MP-0b Small | 665.0 ms | — |
| MP-0b2 Small | 503.3 ms | **286.2 ms** |
| OMAT-0 Small | 732.9 ms | **373.9 ms** |
| OMAT-0 Medium | 1,570.0 ms | 1,117.4 ms |

Compilation was tested directly with the upstream `MACECalculator`, using
`device="cpu"`, `default_dtype="float32"`, and `compile_mode="default"`.
MP-0a Small and MP-0b Small were tested on their standard paths only.
MP-0b2 Small was the fastest of these tested configurations. Compilation made
OMAT-0 Small about twice as fast as its standard path and three times faster
than compiled OMAT-0 Medium.
