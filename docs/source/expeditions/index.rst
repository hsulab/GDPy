Expeditions
===========

This section demonstrates several advanced methods to explore the configuration space, 
which are made up of basic components. In general, an expedition is made up of three 
components, namely, **builder**, **worker**, and some specific parameters. 

.. |Expedition| image:: ../../images/expedition.svg
    :width: 800
    :align: middle

Also, an expedition progresses iteratively. The figure below demonstrates how we decouple 
the working ingrediants and manage the communication between the expedition and 
the job queue (e.g. SLURM) if minimisations were not directly performed in the command line. 

In every iteration, the expedition will build several new structures (either from the scratch or 
based on previously explored structures), and then evolves these structures into more 
physically reasonable ones by minimisation or molecular dynamics. This procedure produces 
a large number of trajectories. Applying some selections, we can extract local minima 
of interest and some structures from trajectories, which help the MLIP learns a comprehensive configuration space.

   |Expedition|

Related Commands
----------------

.. code-block:: shell

   # - explore configuration space defined by `config.yaml` 
   #   results will be written to the `results` folder
   #   a log file will be written to `results/gdp.out` as well
   $ gdp -d exp explore ./config.yaml


List of Expeditions
-------------------

.. toctree::
   :maxdepth: 2

   mc.rst
   ga.rst


