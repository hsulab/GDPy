# installation

## Requirements

Must:

- Python 3.10 or newer
- matplotlib 3.5.0
- numpy 1.21.2
- scipy 1.7.3
- scikit-learn 1.0.1
- [ase] 3.27 or newer
- dscribe 1.2.1
- joblib 1.1.0
- [tinydb] 4.7.0
- pyyaml 6.0
- networkx 2.6.3
- [omegaconf] 2.3.0
- h5py 3.7.0

% - e3nn 0.5.0

Optional:

- jax 0.2.27
- pytorch 1.10.1
- sella 2.0.2
- plumed 2.7.3

## From Source, Conda or Pip

```shell
# Create a python environment

# Install the latest RELEASED version from anaconda
$ conda install gdpx -c conda-forge

# or from pypi
$ pip install gdpx

# Install the latest development version
# 1. download the MAIN branch
$ git clone https://github.com/hsulab/GDPy.git gdpx
#    or the DEV branch
$ git clone -b dev https://github.com/hsulab/GDPy.git gdpx

# 2. Use pip to install the an editable version to
#    the current environment
$ cd gdpx
$ pip install -e ./

# 3. Update the source code
$ cd gdpx
$ git fetch
$ git pull
```

## Optional potential packages

From the repository root, install only the potential packages you need:

```shell
python -m pip install -e '.[deepmd]'
python -m pip install -e '.[mace]'
python -m pip install -e '.[mattersim]'
python -m pip install -e '.[reann]'
python -m pip install -e '.[tace]'
# Or combine extras in the same environment:
python -m pip install -e '.[deepmd,tace,mattersim]'
```

Select DeepMD 2 or 3 explicitly with `.[deepmd2]` or `.[deepmd3]`.
Both include `dpdata` for training data conversion. For a complete TensorFlow
installation, choose one of the following in separate environments:

```shell
# DeepMD 2: CPU or NVIDIA GPU with CUDA 12 runtime dependencies
python -m pip install -e '.[deepmd2-cpu]'
python -m pip install -e '.[deepmd2-cu12]'

# DeepMD 3: CPU or NVIDIA GPU with CUDA 12 runtime dependencies
python -m pip install -e '.[deepmd3-cpu]'
python -m pip install -e '.[deepmd3-cu12]'
```

If compatible CUDA and cuDNN libraries are already installed, use
`.[deepmd2-gpu]` or `.[deepmd3-gpu]` instead. GPU options require a compatible
NVIDIA driver. The original `deepmd` extra allows either major version;
`deepmd-gpu` and `deepmd-cu12` select DeepMD 3. See {doc}`potentials/deepmd`
for version constraints, backend selection, and GPU verification.

The `mace` extra installs `mace-torch` and PyTorch for inference and training.
See {doc}`potentials/mace` for CPU/GPU setup and {doc}`trainers/mace` for the
installed training command.

The TACE extra uses a tested GitHub commit and requires Git. See
{doc}`potentials/tace` and {doc}`potentials/mattersim` for model selection and
runtime configuration.

The `reann` extra installs PyTorch and `opt_einsum`. Supply an exported
TorchScript potential (`PES.pt`) for inference. For training, obtain and set up
the upstream REANN code separately. See {doc}`potentials/reann` for manual
training setup, CPU and GPU installation commands, and verification.

[ase]: https://wiki.fysik.dtu.dk/ase
[omegaconf]: https://omegaconf.readthedocs.io
[tinydb]: https://tinydb.readthedocs.io
