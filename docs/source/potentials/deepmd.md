(potential-deepmd)=

# deepmd

The `deepmd` provider loads Deep Potential models.

## Requirements

From the repository root, choose the extra matching your model, for example:

```shell
python -m pip install -e '.[deepmd3-torch]'
```

| Installation option | DeepMD 2 | DeepMD 3 |
| --- | --- | --- |
| Installed separately | `deepmd2` | `deepmd3` |
| PyTorch | Not supported | `deepmd3-torch` |
| TensorFlow CPU | `deepmd2-cpu` | `deepmd3-cpu` |
| TensorFlow GPU, existing CUDA/cuDNN | `deepmd2-gpu` | `deepmd3-gpu` |
| TensorFlow GPU, install CUDA 12 runtime | `deepmd2-cu12` | `deepmd3-cu12` |

Every extra includes `dpdata`. Use separate environments for DeepMD 2 and 3;
combine `deepmd3-torch` with a DeepMD 3 TensorFlow extra to install support for
both frameworks.
Both versions use `provider: deepmd` in configuration.

See {doc}`../installation` for the recommended GPU installation and
{ref}`gpu-verification` for checks. LAMMPS requires a separate binary with the
DeepMD pair style and a compatible exported model.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | DeepMD Python calculator. |
| `lammps` | `lammps` | Yes | LAMMPS with the DeepMD pair style. |
| `lammps` | `ase` | No | LAMMPS evaluates energies and forces; ASE drives the calculation. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### ase + ase

```yaml
potential:
  provider: deepmd
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model: ./graph.pb
    type_list: [H, O]
    estimate_uncertainty: false
executor:
  provider: ase
  method: spc
```

### lammps + lammps

```yaml
potential:
  provider: deepmd
  backend: lammps  # Optional; default for the lammps executor.
  parameters:
    model: ./graph.pb
    type_list: [H, O]
    command: lmp
executor:
  provider: lammps
  method: spc
```

### lammps + ase

```yaml
potential:
  provider: deepmd
  backend: lammps  # Required; overrides the default backend for the ase executor.
  parameters:
    model: ./graph.pb
    type_list: [H, O]
    command: lmp
executor:
  provider: ase
  method: spc
```

### ase + ase / dpa4c

DPA4C models use the DeepMD 3 PyTorch-Exportable backend. Export a trained
checkpoint to a local `.pt2` model before passing it to **gdp**:

```shell
dp --pt-expt freeze -c model.ckpt.pt -o frozen_model --lower-kind graph
```

Save the following configuration as `dpa4c.yaml`:

```yaml
potential:
  provider: deepmd
  backend: ase
  parameters:
    model: ./frozen_model.pt2
executor:
  provider: ase
  method: spc
```

Then evaluate any ASE-readable structure file:

```shell
gdp -d dpa4c-spc -r dpa4c.yaml compute structures.xyz
```

The ASE backend reads the element mapping from the model. The `.pt2` archive is
specific to the device type used during export, so export it for the CPU or GPU
that will run the calculation. **gdp** currently requires a local model path;
DeepMD preset names such as `dpa4c-nano-v20260901` are not resolved
automatically.

(deepmd-dpa4c-export-troubleshooting)=

#### Export troubleshooting

:::{note}
Use Python 3.12 for DPA4C export rather than Python 3.10. Check the active
environment with `python --version` before running `dp --pt-expt freeze`.

If CPU export still fails on macOS ARM with an undeclared variable in generated
C++ code, scalar code generation provides a workaround without changing the
checkpoint weights:

```shell
DEVICE=cpu OMP_NUM_THREADS=1 python -c \
    'import torch._inductor.config as cfg; cfg.cpp.simdlen = 1; from deepmd.main import main; main()' \
    --pt-expt freeze -c /path/to/model.ckpt.pt \
    -o ./frozen_model --lower-kind graph
```

This generated-code failure was also observed with Python 3.12 and PyTorch
2.11, so upgrading Python alone may not resolve it.
:::

(bh-supported-nanoparticle-dpa4-example)=

### User-provided DPA4-mini and DPA4C-mini

With the `deepmd3-torch` extra and DeepMD-kit 3.2 or newer, export your own
pretrained checkpoint for the device that will run inference. These examples
neither bundle nor download weights. For CPU:

```shell
mkdir -p models
DEVICE=cpu OMP_NUM_THREADS=1 dp --pt freeze \
    -c /path/to/DPA4-Mini-OMat24-v20260805.pt -o ./models/dpa4-mini
DEVICE=cpu OMP_NUM_THREADS=1 dp --pt-expt freeze \
    -c /path/to/DPA4C-Mini-OMat24-v20260819.pt \
    -o ./models/dpa4c-mini --lower-kind graph
```

DPA4 uses `--pt`; DPA4C uses `--pt-expt`. Alternatively, provide an existing
`.pt2` export and update `potential.parameters.model` in your runtime. See
{ref}`deepmd-dpa4c-export-troubleshooting` if DPA4C export fails.

The following runtimes pair these local models with short ASE
relaxations for the {ref}`supported-cluster exploration <bh-supported-nanoparticle-example>`.
Both checkpoints must support Cu, Al, and O.

The runtimes can also be used for other systems supported by the models. Add
system-specific constraints to your own runtime copy when needed.

```{literalinclude} ../../../examples/global_optimisation/runtimes/dpa4_mini.yaml
:language: yaml
```

```{literalinclude} ../../../examples/global_optimisation/runtimes/dpa4c_mini.yaml
:language: yaml
```

In the {ref}`single-thread CPU comparison <potential-cpu-inference-comparison>`,
DPA4-mini measured **601.1 ms** per energy-and-force evaluation; DPA4C-mini
measured **29.2 ms** with the scalar CPU export workaround. The latter was
about 20.6 times faster on this fixture.

### Parameter notes

`model` accepts one existing checkpoint or a list. `models` is also accepted
as an alias when `model` is absent. `type_list` defines the model’s element
mapping and must agree with its training configuration. Optional `head` is
passed to the ASE DeepMD calculator for models that support it.

For ASE, multiple models with `estimate_uncertainty: true` create a committee;
otherwise only the first is evaluated. For the LAMMPS interface, optional
`command` supplies the executable (default `lmp`).

Checkpoint formats depend on the installed DeepMD backend; the `.pb` example
is not a format requirement for every backend. See {doc}`../trainers/deepmd`
for training configuration.
