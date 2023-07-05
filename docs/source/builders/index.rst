Builders
========

Builders are several classes that generate structures. They can be defined in two 
categories as Builder and Modifier.

Related Commands
----------------

.. code-block:: shell

    # - build structures based on `config.yaml`
    #   results would be written to the `results` directory
    $ gdp -d ./results build ./config.yaml 

    # - build structures based on `config.yaml`
    #   some builders (modifiers) require substrates as input
    #   it can be set in `config.yaml` directly or as a command argument
    $ gdp -d ./results build ./config.yaml --substrates ./sub.xyz

    # - build 10 structures based on `config.yaml`
    #   `number` can be used for some random-based builders (modifiers) 
    #   otherwise, only **1** structure is randomly built.
    $ gdp -d ./results build ./config.yaml --substrates ./sub.xyz --number 10

.. FixedNumberBuilders that returns a fixed number of structures based on input parameters.
.. 
..     direct, molecule, graph
.. 
.. UserDefinedNumberBuilders
.. 
..     random, perturbator
.. 
.. Modifiers that must have substrates as input
.. 
..     repeat, cleave, perturb

List of Builders
----------------

.. toctree::
    :maxdepth: 2

    dimer.rst
    random.rst
    graph.rst

Related Components
------------------

.. toctree::
    :maxdepth: 2

    region.rst

