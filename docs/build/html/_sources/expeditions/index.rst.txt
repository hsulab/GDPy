Expeditions
===========

This section introduces expeditions implemented in GDPy. In general, each expedition
includes four stages, namely create, collect, select, and label. As usual, we offer 
a unified input file to access different explorations by setting the **method** 
and its corresponding **create** procedure.

The code structure is

.. code-block:: python

    for exp in explorations:
        for sys in systems:
            create()   # run simulation 
            collect()  # collect results
            select()   # select structures
            label()    # label candidates

The input file structure is 

.. code-block:: yaml

    database: ./set
    method: MD # options are md, evo, ads, rxn
    systems:
        ... # a dict of system definitions
    explorations:
        ... # a dict of expedition definitions
        # for each exploration
        exp:
            systems: [...]
            create:
                ...
            collect:
                ...
            select:
                ...
        ...

The related command is 

.. code-block:: shell

    # gdp -h for more info
    $ gdp -h

    # run explorations defined in exp.yaml with potential as pot.yaml and
    #   with selected candidates labelled by ref.yaml
    $ gdp -p ./pot.yaml -r ./ref.yaml explore ./exp.yaml

Define a System
---------------




List of Expeditions
-------------------

.. toctree::
   :maxdepth: 2

   md.rst
   population.rst
   adsorbate.rst
   reaction.rst
