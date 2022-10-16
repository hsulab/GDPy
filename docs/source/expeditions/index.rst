Expeditions
===========
This section introduces expeditions implemented in GDPy. In general, each expedition
includes four stages, namely create, collect, select, and label. As usual, we offer 
a unified input file to access different explorations by setting the **method** 
and its corresponding **create** procedure. Expeditions can be combined with any 
potential formulation that can be recognised in `pot.yaml` (see :ref:`potential examples`). 

See Expedition_Examples_ in the GDPy repository for prepared input files.

.. _Expedition_Examples: https://github.com/hsulab/GDPy/tree/main/examples/expedition

The related command is 

.. code-block:: shell

    # gdp -h for more info
    $ gdp -h

    # run explorations defined in exp.yaml with potential as pot.yaml and
    #   with selected candidates labelled by ref.yaml
    $ gdp -p ./pot.yaml -r ./ref.yaml explore ./exp.yaml

.. important:: 
    If a scheduler is defined in `pot.yaml`, the `create` step would be submitted to
    the queue and the expedition would stop. When running the `explore` command 
    again, the expedition would check whether the current step (e.g. `create`) is 
    finished and move on to the next if the current is finished.

.. important:: 
    If `-r ref.yaml` is not provided, the expedition would stop at the `select` 
    step and prompt `Reference worker is not set properly.`. Thus, the selected 
    candidates would not be labelled by reference (e.g. DFT).

The code structure is

.. code-block:: python

    for exp in explorations:
        for sys in systems:
            create()   # run simulation 
            collect()  # collect results
            select()   # select structures
            label()    # label candidates

The input file (`exp.yaml`) structure is 

.. code-block:: yaml

    database: ./set # where labelled structures are stored
    method: md # options are md, evo, ads, rxn
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

Define a System
---------------
In the **systems** section, we need to specify what particular systems would be 
explored. There are several parameters need to define:

- prefix:      Use to distinguish systems when creating directories if needed.
- composition: System chemical composition.
- kpts:        System-depandent for DFT calculation.
- constraint:  Fix some atoms during the exploration.

More importantly, there are two ways to create initial structures.

The first one using structures from a file. The example below reads structures with 8
Cu atoms as initial structures to explore.

.. code-block:: yaml

    sysName:
        prefix: sysPrefix
        structure: ./frames.xyz
        composition: {"Cu": 8}
        kpts: 30 # kspacing
        constraint: "1:4"

The second one defining a generator to create some. The example below creates
2 (`size`) structures that have 9 water molecules randomly placed above a Pt substrate.

.. code-block:: yaml

    sysName:
        prefix: sysPrefix
        generator:
            method: random
            type: surface
            composition: {"H2O": 9}
            covalent_ratio: 0.8
            substrate: ./substrates.xyz
            surfdis: [1.5, 6.5]
        size: 2
        composition: {"H": 9, "O": 18, "Pt": 16}
        kpts: [2,2,1] # kmesh
        constraint: "1:4"

List of Expeditions
-------------------
The **method** should be set to access different expedition strategies (e.g. 
md, evo, ads and rxn).

.. toctree::
   :maxdepth: 2

   md.rst
   evo.rst
   ads.rst
   rxn.rst
