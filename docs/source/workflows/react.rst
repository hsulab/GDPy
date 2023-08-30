Run Massive NEB Calculations
============================

We can run reaction calculations with the operation `react` and the variable `reactor`. 
In the workflow below, we use Nudged Elastic Band (NEB) to calculate the reaction pathway 
between several structure pairs.

For a typical NEB calculation, it has the following steps:

#. Minimise the initial state (IS) and the final state (FS).
#. Align atoms in IS and FS and make sure the atom order is consistent.
#. Run NEB to converge the minimum energy path (MEP).

To run several NEB calculations at the same time, we need 

#. Minimse all intermediates of interest (operations: read_stru to select_converged).
#. Run NEB calculations on selected structure pairs. (operations: pair -> react)

Node Definitions
----------------

The `reactor` variable is very similiar to `computer` that we use for minimisations and 
molecular dynamics. It requires a `potter` and a `driver` as well. The `potter` can be 
any potential interfaced with **gdp**. Currently, the `driver` only suports NEB.

The `init` section defines some parameters related to NEB.

- mic: 

  Whether use the minimum image convention. If used, the smallest displacement between 
  the IS and the FS will be found according to the periodic boundary.

- nimages: 

  Number of images that define the pathway. This includes the IS and the FS that are 
  two fixed points.

- optimiser:

  Algorithm to optimise the structures. `mdmin` is recommanded. Sometimes `bfgs` is 
  more effective but less efficient as it needs to update the Hessian matrix. 
  Matrix diagonalisation for Hessian is very time-consuming, O(N^3), where N is the 
  matrix size, and it takes much more time to solve the Hessian than evaluate the forces 
  for a large structure or a large number of images.

The `run` section is the same as the one in `computer`. `fmax` is the force convergence 
tolerance and `steps` is the maximum optimisation steps. Sometimes one would like NOT to 
minimise the IS and the FS that are pre-minimised by DFT, provided that the MLIP is not 
good enough. `steps` can be set to -1 that means a single-point calculation will be 
performed.

.. code-block:: yaml

  reactor:
    type: reactor
    potter: ${vx:dpmd}
    driver:
      backend: ase
      init:
        optimiser: mdmin
        mic: true
        nimages: 11
      run:
        fmax: 0.05
        steps: 50
        constraint: "lowest 16"

The `locate` variable is used to get the last frame of the minimisation. If there are
3 intermediates minimised with 50 steps, the input for the selection is an AtomsArray 
with a shape of (3, 50). The `axis` defines on what dimension the selection is performed. 
The `indices` defines the indices to select on the selected dimension. In the example below, 
the selection is performed on the second dimension (axis: 1, the shape 50) and the last structure 
is selected (indices: -1).

.. code-block:: yaml

  locator:
    type: selector
    selection:
      - method: locate
        axis: 1
        indices: "-1"

The `pair` operation constructs the structure pairs for NEB. Here, the `custom` method 
is used. If the input `structures` has 3 structures, it will prepare two pairs, 
one between the first and the second structure, and another is between the second 
and the third structure. Therefore, two pathways will be calculated by NEB later.
  
.. code-block:: yaml

  pair:
    type: pair_stru
    structures: ${op:select_converged}
    method: custom
    pairs: # NOTE: index starts from 0!!!
      - [0, 1]
      - [1, 2]

Session Configuration
---------------------

.. code-block:: yaml

  variables:
    reactor:
      type: reactor
      potter: ${vx:dpmd}
      driver:
        backend: ase
        init:
          optimiser: mdmin
          mic: true
          nimages: 11
        run:
          fmax: 0.05
          steps: 50
          constraint: "lowest 16"
    dpmd_min:
      type: computer
      potter: ${vx:dpmd}
      driver:
        task: min
        run:
          fmax: 0.05 # eV/Ang
          steps: -1 # steps: 400
          constraint: "lowest 16"
    dpmd:
      type: potter
      name: deepmd
      params:
        backend: ase
        type_list: ["Al", "Cu", "O"]
        model:
          - ./graph.pb
    scheduler_loc:
      type: scheduler
    locator:
      type: selector
      selection:
        - method: locate
          axis: 1
          indices: "-1"
  operations:
    read_stru:
      type: read_stru
      fname: ./intermediates.xyz
    run_dpmin:
      type: compute
      builder: ${op:read_stru}
      worker: ${vx:dpmd_min}
      batchsize: 512
    extract_min:
      type: extract
      compute: ${op:run_dpmin}
    select_converged:
      type: select
      structures: ${op:extract_min}
      selector: ${vx:locator}
    pair:
      type: pair_stru
      structures: ${op:select_converged}
      method: custom
      pairs: # NOTE: index starts from 0!!!
        - [0, 1]
        - [1, 2]
    react:
      type: react
      structures: ${op:pair}
      reactor: ${vx:reactor}
  sessions:
    _rxn: react
