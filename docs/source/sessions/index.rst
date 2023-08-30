.. _sessions:

Sessions
========

To make the structure exploration and the model training much more flexible, **gdp** 
organises several steps of a workflow into a `session`, which is a graph flow that 
is made up of `variables` and `operations`. In general, `variables` are some simple 
components that execute for little time while `operations` require a lot of time to 
get the final results. One can organise a custom workflow using the combination of 
`variables` and `operations` through a simple YAML configuration file.

Each variable uses some parameters and sometimes other variables as inputs and 
creates a working component. Each `operation` accepts `variables`, `operations` 
and custom parameters as inputs and forwards calculation results.


Related Commands
----------------

.. code-block:: shell

    # - run a session defined in the file `./config.yaml`
    $ gdp session ./config.yaml

    # - run a session defined in the file `./config.yaml`
    #   fill in the placeholders in config by `--feed`
    $ gdp session ./config.yaml --feed temperatures=50,150,300


Minimal Configuration
---------------------

Every `session` configuration needs three sections, namely, `variables`, `operations`, 
and `sessions`. In the configuration below, one defines a workflow that runs molecular 
dynamics simulations of some structures.

For each variable, one must first set a `type` parameter. 
`driver`, `potter` (potential), and `scheduler` are simple variables that can be 
defined by several parameters, which are similiar to the definition in :ref:`Potential Examples`. 
The `computer` variable requires three variables as the input. One can **${vx:potter}** to 
point to the required variable. **vx** means the input is in the `variables` section and 
`potter` is just the variable name. At the first glance, this way of definition 
is a little complicated than ones uses in `gdp compute`. However, if several different 
`computer` are required in one workflow, `driver` and `scheduler` variables can be 
reusable.

The definition for `operations` is similar. One can further use **${op:read}** to 
point to a defined operation. **op** means the input is in the `operations` section 
and **read** is the operation name. Here, **read** operation reads structures from a file,
which is a wrapper of the `ase.io.read` function. Then **scan** operation takes 
the output of **read** (structures) to run simulations defined in the **computer** 
variable.

`sessions` sets the entry point of the workflow. Here, **scan** is the name of the 
operation, and ALL results by variables and operations will be saved the directory 
`_scan`. When starting this worfklow, **gdp** checks what inputs the **scan** operation 
needs and executes those inputs, which forms a flow of operations as 

.. code-block:: shell

    **scan** <- **read**
             <- **computer** <- **potter**
                             <- **driver**
                             <- **scheduler_loc**

.. code-block:: yaml

    variables:
      computer:
        type: computer
        potter: ${vx:potter}
        driver: ${vx:driver}
        scheduler: ${vx:scheduler_loc}
      potter:
        type: potter
        name: deepmd
        params:
          backend: lammps
          command: lmp -in in.lammps 2>&1 > lmp.out
          type_list: [H, O]
          model:
            - ./graph-0.pb
            - ./graph-1.pb
      driver:
        type: driver
        task: md
        init:
          md_style: nvt
          timestep: 0.5
          temp: [150, 300, 600] # ${placeholders.temperatures}
          dump_period: 50
        run:
          steps: 10000
      scheduler_loc:
        type: scheduler
        backend: local
    operations:
      read:
        type: read_stru
        fname: ./candidates.xyz # ${placeholders.structure}
      scan:
        type: compute
        builder: ${op:read}
        worker: ${vx:_computer}
        batchsize: 256
    sessions:
      _scan: scan


Variables
---------



Operations
----------

See :ref:`operations`.

.. +--------+---------+
.. | build  | modify  |
.. +--------+---------+
.. | select | drive   |
.. +--------+---------+
