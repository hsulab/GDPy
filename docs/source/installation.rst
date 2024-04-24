Installation
============

Requirements
------------

Must:

- Python 3.9
- matplotlib 3.5.0
- numpy 1.21.2
- scipy 1.7.3
- scikit-learn 1.0.1
- ase_ 3.22.1
- dscribe 1.2.1
- joblib 1.1.0
- tinydb_ 4.7.0
- pyyaml 6.0
- networkx 2.6.3
- omegaconf_ 2.3.0
- h5py 3.7.0

.. - e3nn 0.5.0

.. _ase: https://wiki.fysik.dtu.dk/ase
.. _tinydb: https://tinydb.readthedocs.io
.. _omegaconf: https://omegaconf.readthedocs.io

Optional:

- jax 0.2.27
- pytorch 1.10.1
- sella 2.0.2
- plumed 2.7.3

From Source, Conda or Pip
-------------------------

.. code-block:: shell

    # Create a python environment
    
    # Install the latest RELEASED version from anaconda
    $ conda install gdpx -c conda-forge
    
    # or from pypi
    $ pip install gdpx
    
    # Install the latest development version
    # 1. download the MAIN branch
    $ git clone https://github.com/hsulab/GDPy.git
    #    or the DEV branch
    $ git clone -b dev https://github.com/hsulab/GDPy.git
    
    # 2. Use pip to install the an editable version to 
    #    the current environment
    $ cd GDPy
    $ pip install -e ./
    
    # 3. Update the source code
    $ cd GDPy
    $ git fetch
    $ git pull
    