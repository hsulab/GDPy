(potential-deepmd)=

# deepmd

The `deepmd` provider loads Deep Potential models.

## Installation

Choose a major version and install one extra from the repository root in the
same environment as gdpx. Every extra includes `dpdata` for training data
conversion; none is required by the base gdpx installation.

| Installation | DeepMD 2 | DeepMD 3 |
| --- | --- | --- |
| Backend installed separately | `.[deepmd2]` | `.[deepmd3]` |
| TensorFlow CPU | `.[deepmd2-cpu]` | `.[deepmd3-cpu]` |
| TensorFlow GPU, existing CUDA/cuDNN | `.[deepmd2-gpu]` | `.[deepmd3-gpu]` |
| TensorFlow GPU, pip installs CUDA 12 runtime dependencies | `.[deepmd2-cu12]` | `.[deepmd3-cu12]` |

### DeepMD 2

```shell
# TensorFlow CPU:
python -m pip install -e '.[deepmd2-cpu]'
# Or NVIDIA GPU with CUDA 12 runtime dependencies:
python -m pip install -e '.[deepmd2-cu12]'
```

The `deepmd2` and `deepmd2-cpu` extras constrain DeepMD to `>=2,<3`.
The GPU extras use `>=2.2.11,<3`, following the CUDA 12 installation options
in the [DeepMD 2.2.11 guide](https://docs.deepmodeling.com/projects/deepmd/en/v2.2.11/getting-started/install.html).

### DeepMD 3

```shell
# TensorFlow CPU:
python -m pip install -e '.[deepmd3-cpu]'
# Or NVIDIA GPU with CUDA 12 runtime dependencies:
python -m pip install -e '.[deepmd3-cu12]'
```

All `deepmd3` extras constrain DeepMD to `>=3,<4`. For an existing backend
installation, use `.[deepmd2]` or `.[deepmd3]` to select just the major version.
Use separate environments for DeepMD 2 and 3; their version constraints cannot
be combined. Both versions use `provider: deepmd` in gdpx configuration.

The original `deepmd` extra remains available with `>=2,<4`.
`deepmd-gpu` and `deepmd-cu12` retain the same dependencies as `deepmd3-gpu`
and `deepmd3-cu12`, respectively.

### NVIDIA GPU setup and verification

Use the `-gpu` option when compatible CUDA and cuDNN libraries are already
available, for example through HPC modules. Use `-cu12` to also install the
CUDA 12 runtime dependencies through pip. Both require an NVIDIA GPU and
compatible driver, which must be installed separately. CUDA extras do not
enable NVIDIA GPU execution on macOS.

Check that TensorFlow can see the GPU in the environment where the job runs:

```shell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

An empty list means TensorFlow cannot see a usable GPU. On a cluster, run this
check inside a GPU allocation with the required environment modules loaded.

### PyTorch backend

For PyTorch models with DeepMD 3:

```shell
python -m pip install -e '.[deepmd3]' 'deepmd-kit[torch]>=3,<4'
```

For a specific CUDA build, install the appropriate PyTorch wheel following the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then
install `.[deepmd3]` in that environment. The TensorFlow GPU extras above do
not select the PyTorch CUDA build. Check GPU availability with:

```shell
python -c "import torch; print(torch.cuda.is_available())"
```

See the [DeepMD installation guide](https://docs.deepmodeling.org/projects/deepmd/en/latest/getting-started/install.html)
for GPU and platform-specific installation options. LAMMPS execution requires
a binary with the DeepMD pair style and a model compatible with that build;
the gdpx extra does not install LAMMPS.

## Configuration

```yaml
potential:
  provider: deepmd
  parameters:
    model: ./graph.pb
    type_list: [H, O]
    estimate_uncertainty: false
```

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
