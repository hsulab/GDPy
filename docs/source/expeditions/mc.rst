.. _Monte Carlo:

Monte Carlo (MC)
================

Overview
--------

MC is a conventional method to explore the configuration space.

Example
-------

The related commands are 

.. code-block:: shell

    # - explore configuration space defined by `config.yaml` 
    #   results will be written to the `results` folder
    #   a log file will be written to `results/gdp.out` as well
    $ gdp -d exp -p worker.yaml explore ./config.yaml

    # - after MC is converged i.e. reaches the maximum number of steps,
    #   the MC trajectory is stored at `results/mc.xyz`

In the `operators` section,

Every MC operator has parameters of `temperature`, `pressure`, and `region`. In general, 
these three parameters should be consistent among different operators used in the simulation. 
Otherwise, the simulation may not converge the structure to the phyiscal equilibrium.

To increase the acceptance, `convalent_ratio` is often set to check if the new structure has 
too small or too large distances. The two values are the minimum and the maximum coefficients, 
which will be multipied by the covalent bond distance.

See :ref:`region definitions` for more information about defining a region.

- move:

    Move a particle to a random position with maximum `max_disp` displacement.

- swap:

    Swap the positions of two particles from two different types.

- exchange:

    Exchange particles with an imaginary reservoir by inserting or removing. This 
    changes the number of atoms in the system as it samples the grand canonical 
    ensemble.

.. note::

    In general, operators should have the same region. Otherwise, the simulation is 
    not converged to an equilibrium.

  
In the `convergence` section,

* steps: Number of MC steps.

Since MC usually takes ~5000 steps, the `dump_period` determines what MC step 
will be saved. For example, if `dump_period = 2`, step 0, 2, 4 ... will be saved. 
These saved structures and trajectories can be used for MLIP training.

The input file shown below explores the oxidation of Cu(111) surface. The MC operators 
only apply to atoms in the surface region including Cu and O.

.. code-block:: yaml
    
  method: monte_carlo
  random_seed: 1112
  builder:
    method: read
    fname: ./fcc-s111p44.xyz
  operators:
    - method: exchange
      region:
        method: lattice
        origin: [0, 0, 8.0]
        cell: [10.17, 0, 0, 0, 8.81, 0, 0, 0, 6.0]
      covalent_ratio: [0.8, 2.0]
      reservoir:
        mu: -5.75
        species: O
      temperature: 800
      prob: 0.5
    - method: move
      particles: [Cu, O]
      region:
        method: lattice
        origin: [0, 0, 8.0]
        cell: [10.17, 0, 0, 0, 8.81, 0, 0, 0, 6.0]
      covalent_ratio: [0.8, 2.0]
      max_disp: 2.0
      temperature: 800
      prob: 0.5
  convergence:
    steps: 5
  dump_period: 1

For the `worker.yaml`, the parameter **`use_single`** must be **true** as

.. code-block:: yaml

  use_single: true
  potter:
    name: deepmd
    params:
      backend: lammps
      command: lmp -in in.lammps 2>&1 > lmp.out
      type_list: [Cu, O]
      model:
        - ./graph.pb
  driver:
    backend: lammps
    task: min
    ignore_convergence: false
    run:
      fmax: 0.05
      steps: 400


Application
-----------

.. include:: ../references.rst

#. |ACSCatal2022Xu|