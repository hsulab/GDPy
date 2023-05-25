.. _random builders:

RandomBuilders
==============

This section is about builders that generate random structures under certain geometric 
restraints.

The builder parameters:

* composition:

    A dictionary of the chemical composition to insert. This can be atoms, molecules, 
    or a mixture of them. For example, `{"Cu": 13}`, `{"H2O": 3}` and `{"Cu": 13, "H2O": 3}`.

* region:

    Define the region where random atoms are put. See :ref:`Region Definitions` for 
    more details.

* covalent_ratio:

    The geometric restraints. The minimum and maximum multipliers for the covalent distance 
    between atoms.

* random_seed:

    Random seed for the random generator. Use the same seed to reproduce results.

.. note:: 

    Sometimes the builder will fail to generate new structures due to geometric 
    restraints. Simply reduce the first element of `covalent_ratio` that allows 
    structures with very close atomic distances. If this still does not work, set 
    `test_too_far` to false, which allows sparse atomic positions to generate.

The YAML input file has the format of 

Cluster
-------

Use `cell` to define the box where the generated cluster is in even though it is 
not periodic. Here, a smaller region is set to put random atoms, which makes atoms 
more concentrated.

.. code-block:: yaml

    # - Genreate a cluster with 13 Cu and 3 H2O.
    builder:
      composition: 
        Cu: 13
        H2O: 3
      cell: [30., 0., 0., 0., 30., 0., 0., 0., 30.]
      region:
        method: "lattice"
        origin: [10., 10., 10.,]
        cell: [10., 0., 0., 0., 10., 0., 0., 0., 10.]
      covalent_ratio: [0.6, 2.0]
      test_too_far: false
      random_seed: 1112

Surface
-------

Generate random atoms on a substrate. This is useful to explore reconstructed (amorphous) 
surfaces or supported nanoparticles. The region defines the lattice vector in the x-axis 
and the y-axis but a cut in z-axis that has a range from 7.5 to 13.5 (7.5+6.0).

.. code-block:: yaml

    # - Genreate a surface with 8 Cu and 3 O.
    builder:
      substrate: ./assets/slab.xyz
      composition: 
        Cu: 8
        O: 3
      region:
        method: "lattice"
        origin: [0., 0., 7.5]
        cell: [5.85, 0.0, 0.0, 0.0, 4.40, 0.0, 0.0, 0.0, 6.0]
      covalent_ratio: [0.4, 2.0]
      test_dist_to_slab: false
      test_too_far: false
      random_seed: 1112

Bulk
----

Bulks have random lattice parameters. Use `cell_bounds` to set the range of 
angles and lengths.

.. code-block:: yaml

    # - Genreate a bulk with 4 Cu and 2 O.
    builder:
      composition:
        Cu: 4
        O: 2
      cell_bounds:
        phi: [35, 145]
        chi: [35, 145]
        psi: [35, 145]
        a: [3, 50]
        b: [3, 50]
        c: [3, 50]
