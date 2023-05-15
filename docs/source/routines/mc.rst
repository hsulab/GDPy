.. _Monte Carlo:

Monte Carlo (MC)
================

The related commands are 

.. code-block:: shell

    # - run 10000 steps MC
    #   results would be written to current directory
    $ gdp -p ./pot.yaml task ./mc.yaml --run 10000

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

The input file is organised as 

.. code-block:: shell

    task: mc
    directory: ./results
    random_seed: 1112
    system:
      method: direct
      frames: ./init.xyz
    operators:
      - name: exchange
        temperature: 800
        region:
          method: sphere
          origin: [80, 80, 80]
          radius: 50
        reservoir:
          species: O
          mu: -5.75
        prob: 0.5
      - name: swap
        temperature: 800
        region:
          method: sphere
          origin: [80, 80, 80]
          radius: 50
        particles: ["Pt", "O"]
        covalent_ratio: [0.5, 2.0]
        prob: 0.5
      - name: move
        temperature: 800
        region:
          method: sphere
          origin: [80, 80, 80]
          radius: 50
        particles: ["O"]
        max_disp: 2.0
        prob: 0.5
    drivers:
      init:
        backend: lammps
        task: min
        run:
          steps: 10000
          fmax:  0.1
      post:
        backend: lammps
        task: min
        run:
          steps: 10000
          fmax:  0.1

.. note::

    In general, operators should have the same region. Otherwise, the simulation is 
    not converged to an equilibrium.
