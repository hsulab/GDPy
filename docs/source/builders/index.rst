Builders
========

Builders are several classes that generate structures. They can be defined in two 
categories as Builder and Modifier.

Related Commands
----------------

.. code-block:: shell

    # - build 10 structures based on `config.yaml`
    #   results would be written to the `results` directory
    $ gdp -d ./results build ./config.yaml --number 10

List of Builders
----------------

.. toctree::
    :maxdepth: 2

    dimer.rst
    random.rst

Related Components
------------------

.. toctree::
    :maxdepth: 2

    region.rst

