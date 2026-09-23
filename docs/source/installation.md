# installation

## Requirements

Use Python 3.10 or newer. Required packages, optional extras, and their version
constraints are declared in
[`pyproject.toml`](https://github.com/hsulab/GDPy/blob/main/pyproject.toml).
Pip installs the dependencies for the extras you select.

## Create an environment

Use Conda to create an isolated Python environment, then use pip to install
GDPy and its dependencies:

```shell
conda create -n gdpx python=3.12 pip
conda activate gdpx
```

## Install from source

Clone the repository and enter its root directory:

```shell
git clone https://github.com/hsulab/GDPy.git
cd GDPy
```

For the base package, use an editable installation so source changes take
effect without reinstalling:

```shell
python -m pip install -e .
```

For GPU potential packages, use the installation command below instead.
Add `-e` to that command if you also want an editable installation.

(gpu-install-on-cpu-node)=

## Recommended GPU installation

For DeepMD 3 (PyTorch and TensorFlow), TACE, and MatterSim, run this from the
repository root on either a CPU or GPU Linux node:

```shell
python -m pip install '.[deepmd3-torch,deepmd3-cu12,tace,mattersim]' 'deepmd-kit==3.2.0' 'torch==2.11.0+cu128' --extra-index-url https://download.pytorch.org/whl/cu128
```

We use DeepMD's explicit backend extras to establish the PyTorch and TensorFlow
GPU dependencies. TACE and MatterSim reuse the same compatible PyTorch and
shared libraries. Resolving them together with the pinned CUDA wheel reduces
version mismatches from separate installations. Their declared shared
dependency constraints overlap; the complete environment has not yet been
validated end to end.

This selects the CUDA-enabled packages even when no GPU is visible during
installation. Git is required for TACE. GPU execution requires a compatible
NVIDIA driver on the compute node; `CONDA_OVERRIDE_CUDA` is unnecessary.

## Other potential packages

:::{warning}
Some provider versions require incompatible dependencies. For example,
[`mace-torch` 0.3.16](https://pypi.org/pypi/mace-torch/0.3.16/json) requires
`e3nn==0.4.4`, while DeepMD 3.2.0's PyTorch backend requires `e3nn>=0.5.9`
and MatterSim 1.2.3 requires `e3nn>=0.5.0`. Install these MACE versions in a
separate environment rather than adding `mace` to the command above.
:::

Install individual providers with `python -m pip install '.[mace]'`, replacing
`mace` with the extra you need. See the provider pages for other backends and
training setup: {doc}`potentials/deepmd`, {doc}`potentials/mace`,
{doc}`potentials/mattersim`, {doc}`potentials/reann`, and {doc}`potentials/tace`.
