.. _graph builders:

Graph
=====


insert
------

.. code-block:: yaml

    # config.yaml
    method: graph_insert
    species: CO
    spectators: [C, O]
    sites:
        - cn: 1
          group:
              - "symbol Cu"
              - "region cube 0. 0. 0. -100. -100. 6. 100. 100. 8."
          radius: 3
          ads:
              mode: "atop"
              distance: 2.0
        - cn: 2
          group:
              - "symbol Cu"
              - "region cube 0. 0. 0. -100. -100. 6. 100. 100. 8."
          radius: 3
          ads:
              mode: "atop"
              distance: 2.0
        - cn: 3
          group:
              - "symbol Cu"
              - "region cube 0. 0. 0. -100. -100. 6. 100. 100. 8."
          radius: 3
          ads:
              mode: "atop"
              distance: 2.0
    graph: 
        pbc_grid: [2, 2, 0]
        graph_radius: 2
        neigh_params:
            covalent_ratio: 1.1
            skin: 0.25

remove
------

.. code-block:: yaml

    # config.yaml
    method: graph_remove
    species: O
    graph: 
        pbc_grid: [2, 2, 0]
        graph_radius: 2
        neigh_params:
            covalent_ratio: 1.1
            skin: 0.25
    spectators: [O]
    target_group:
        - "symbol O"
        - "region surface_lattice 0.0 0.0 8.0 9.8431 0.0 0.0 0.0 10.5534 0.0 0.0 0.0 8.0"

exchange
--------

.. code-block:: yaml

    # config.yaml
    method: graph_exchange
    species: Zn
    target: Cr
    graph:
        pbc_grid: [2, 2, 0]
        graph_radius: 2
        neigh_params:
            # AssertionError: Single atoms group into one adsorbate.
            # Try reducing the covalent radii. if it sets 1.1.
            covalent_ratio: 1.0
            skin: 0.25
    spectators: [Zn, Cr]
    target_group:
        - "symbol Zn Cr"
        - "region surface_lattice 0.0 0.0 8.0 9.8431 0.0 0.0 0.0 10.5534 0.0 0.0 0.0 8.0"
