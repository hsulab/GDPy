Getting Started
===============
This section introduces the schema-v2 **potential**, **executor**, and
**scheduler** components used to calculate structures with GDPy.

The related commands are 

.. code-block:: shell

    # gdp -h for more info
    $ gdp -h

    # --- run simulations on local nodes or submitted to job queues
    $ gdp -r ./runtime.yaml compute ./structures.xyz

    # - if -d option is used, results would be written to the folder `./results`
    $ gdp -d ./results -r ./runtime.yaml compute ./structures.xyz

An example input file (``runtime.yaml``) is organised as follows:

.. code-block:: yaml

    schema_version: 2
    potential:
        provider: deepmd
        parameters:
            model: ./graph.pb
    executor:
        provider: ase
        method: md
        parameters:
            ensemble: nvt
            temp: 600
            timestep: 1.0
            steps: 100
    scheduler:
        provider: local
        parameters: {}

Units
-----
We use the following units through all input files:

Time ``fs``, Length ``AA``, Energy ``eV``, Force ``eV/AA``.

Potential
---------
Potential providers describe a model independently of the software that will
execute it. Materialization connects the two at runtime.

The example below shows how to define a **deepmd** potential using the **ase** backend 
in a **yaml** file: 

.. code-block:: yaml

    schema_version: 2
    potential:
        provider: deepmd
        method: default
        parameters:
            model: ./graph.pb

See :ref:`Potential Examples` section for more details. 


Executor
--------
The **executor** specifies the concrete calculation. Its provider determines
the software, while ``method`` selects single point, minimization, molecular
dynamics, dimer, or a path method such as NEB.

The example below shows how to define an **executor** in a **yaml** file:

.. code-block:: yaml

    executor:
        provider: ase
        method: md
        parameters:
            ensemble: nvt
            temp: 600 # temperature, Kelvin
            timestep: 1.0 # fs
            steps: 100

Scheduler
---------
With **potential** and **executor** defined, we can run simulations on local
machines (directly in the command line). However, simulations, under most 
circumstances, would be really heavy even by MLIPs (imagine a 10 ns molecular 
dynamics). The simulations would ideally be dispatched to high performace clusters
(HPCs).

The example below shows how to define a **scheduler** in a **yaml** file:

.. code-block:: yaml

    scheduler:
        provider: slurm
        parameters:
            partition: k2-hipri
            ntasks: 1
            time: "0:10:00"
            environs: "conda activate py37\n"

Runtime
-------

A runtime is the complete executable unit: one potential, one executor,
optional modifiers, and one scheduler. Use explicit lists for independent
runtimes and explicit nested lists for runtime chains; GDPy does not infer a
Cartesian product between components.

.. note:: 

    If **scheduler** is omitted, GDPy uses ``LocalScheduler``.
