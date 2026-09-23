(potential-mace)=

# mace

The `mace` provider loads local MACE checkpoints.

## Installation

From the repository root, install the optional MACE dependencies:

```shell
python -m pip install -e '.[mace]'
```

The extra installs `mace-torch` and PyTorch, including the ASE calculator and
MACE training code. `mace-torch` is the distribution recommended by the
[MACE installation guide](https://mace-docs.readthedocs.io/en/latest/guide/installation.html).

For CPU execution on Linux or Windows, install the CPU PyTorch build first:

```shell
python -m pip install 'torch>=1.12,!=2.4.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[mace]'
```

For NVIDIA GPU execution, first install the appropriate CUDA-enabled PyTorch
build using the [PyTorch installation selector](https://pytorch.org/get-started/locally/),
then install `.[mace]`. The same extra serves CPU and GPU environments.
Check the installation with:

```shell
python -c "from mace.calculators import MACECalculator; import torch; print('GPU available:', torch.cuda.is_available())"
mace_run_train --help
```

For LAMMPS, supply a compatible exported model and a binary with the MACE pair
style; the extra does not install LAMMPS.

## Configuration

```yaml
potential:
  provider: mace
  parameters:
    model: ./mace.model
    type_list: [H, O]
    precision: float32
    estimate_uncertainty: false
```

`model` accepts an existing path or list of paths. The ASE adapter passes
`precision` (default `float32`) as `default_dtype`, selecting CUDA if available
and CPU otherwise. Multiple models with `estimate_uncertainty: true` enable
a committee; otherwise only the first is used. This adapter expects local
files rather than foundation-model names.

The LAMMPS interface requires exactly one exported model; `command` defaults
to `lmp`.

See {doc}`../trainers/mace` for training.
