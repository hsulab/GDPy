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
python -m pip install -e '.[mattersim]'
python -m pip install -e '.[tace]'
# Or install both in the same environment:
python -m pip install -e '.[tace,mattersim]'
```

The TACE extra uses a tested GitHub commit and requires Git. See
{doc}`potentials/tace` and {doc}`potentials/mattersim` for model selection and
runtime configuration.

[ase]: https://wiki.fysik.dtu.dk/ase
[omegaconf]: https://omegaconf.readthedocs.io
[tinydb]: https://tinydb.readthedocs.io
