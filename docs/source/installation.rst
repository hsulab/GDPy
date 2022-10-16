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

.. - e3nn 0.5.0

.. _ase: https://wiki.fysik.dtu.dk/ase
.. _tinydb: https://tinydb.readthedocs.io

Optional:

- jax 0.2.27
- pytorch 1.10.1
- sella 2.0.2
- plumed 2.7.3

From Source
-----------

Use conda to install necessary packages and add executable **gdp** to PATH.

.. code-block:: shell

    # install packages
    $ conda install ase dscribe joblib networkx tinydb pyyaml -c conda-forge
    # download repository
    $ git clone https://github.com/hsulab/GDPy.git
    # add package to python path
    $ export PYTHONPATH=$PYTHONPATH:_YOUR-GDPY-PATH_
    # add executable to path
    $ export PATH=$PATH:_YOUR-GDPY-PATH_/bin/gdp

From Conda
----------

Coming...

From Pip
--------

Coming...