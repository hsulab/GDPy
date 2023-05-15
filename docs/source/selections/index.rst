Selections
==========

This section gives more details how to run basic selections with different selectors
using a unified input file.

The related commands are 

.. code-block:: shell

    # gdp -h for more info
    $ gdp -h

    # - results would be written to current directory
    $ gdp select ./selection.yaml -s structures.xyz

    # - if -d option is used, results would be written to it
    $ gdp -d results select ./selection.yaml -s structures.xyz

    # - accelerate the selection using 8 parallel processes,
    #   which is useful for descriptor-based selection as it requires
    #   massive computations
    $ gdp -nj 8 -d results select ./selection.yaml -s structures.xyz

The selection configuration (`./selection.yaml`) is organised as

.. code-block:: yaml

    selection:
      - method: property
        ... # define property and sparsification
        number: [512, 0.5]
      - method: descriptor
        ... # define descriptor and sparsification
        number: [128, 1.0]

This `selection` defines a sequential produces that consists of two selectors. 
Input structures will be first selected based on the property and the selected 
structures will be further selected based on the descriptor.

For the most selections, a parameter `number` is required as it determines the number of 
selected structures. The first value is a fixed number and the second value is a percentage. 
If the input dataset have 500 structures, with `number: [512, 0.5]`, 250 structures will be 
selected (500*0.5=250 as 500 < 512). Then, with `number: [128, 1.0]`, 128 structures will be 
selected (250 > 128).

After a successful selection, there are several output files. The `selected_frames.xyz` 
contains the final structures. Output files start with a number that indicates their 
oder in the list of selections. Some files, which end with `-info`, stores basic information 
of selected structures. For example, 

.. code-block:: shell

    #  index    confid      step    natoms           ene          aene        maxfrc         score
           0        -1         0        43     -196.7322       -4.5752       16.1094        0.0994
           1        -1       100        43     -242.8428       -5.6475       48.5978        0.0203
          87        -1      8700        43     -271.3238       -6.3099       30.7878        0.0164
          88        -1      8800        43     -271.2264       -6.3076       47.6111        0.0175
          89        -1      8900        43     -284.6631       -6.6201       64.4184        0.0143
          90        -1      9000        43     -303.0153       -7.0469       60.4111        0.0147
          91        -1      9100        43     -311.1232       -7.2354       66.2150        0.0120
          92        -1      9200        43     -309.4916       -7.1975       60.4200        0.0091
          93        -1      9300        43     -312.9583       -7.2781      149.9330        0.0097
          94        -1      9400        43     -314.2778       -7.3088       29.8337        0.0089
          95        -1      9500        43     -315.8645       -7.3457       34.8396        0.0114
          96        -1      9600        43     -310.2994       -7.2163       24.1396        0.0073
          97        -1      9700        43     -313.9329       -7.3008       29.1520        0.0062
          98        -1      9800        43     -327.4579       -7.6153       14.7447        0.0074
          99        -1      9900        43     -330.4879       -7.6858       20.1336        0.0083
         100        -1     10000        43     -329.5945       -7.6650       34.6097        0.0055
    # random_seed None

The first columns are structure identifiers that come from explorations, for instance, 
the candidate ID (confid) and the dynamics step (step) in MD or minimisation. Other notations are `natoms` - number of atoms, `ene` - total energy, `aene` - 
average atomic energy (ene/natoms), `maxfrc` - maximum atomic force, `score` - selection 
score whose meaning depends on the sparsification method. Units are in `eV` nad `eV/Ang`.

There have some other output files by specific selection method. Find details in the following 
subsections.

.. warning::

    When run the same selection again, `gdp` will read the cached results (`-info.txt` files).
    However, it will not check whether the input structures are different from the last time. 
    Remove output files before selection if necessary.


Property
--------

`Select structures based on properties.` The property can be total energy, atomic forces, or 
any properties that can be stored in the **ase** `atoms.info`. The example below demonstrates 
that the selection based on `max_devi_f` that is the maximum deviation of force prediction by 
a committee of MLIPs.

After chosing the property, there are several sparsification methods to select structures.

- filter: 
  
    Select structures that have property within `range`. All valid structures will be 
    selected, which is not affected by the parameter `number`.

- sort: 

    Sort structures by property and select the first `number` of them. Set `reverse: True` 
    if structures with larger property values are of interest.

- hist: 

    Randomly select `number` structures based on probabilities by the histogram.
    For example, if 10 structures will be selected, dataset has 100 structures in
    bin 1 and 25 in bin 2, then roughly 8 will be from bin 1 and 2 from bin 2.

- boltz: 
 
    Randomly select `number` structures based on probabilities by the Boltzmann distribution. 
    This is useful when selecting structures based on energy-related properties. 
    The probabilty is computed as `exp(-p/kBT)` where `p` is the property value 
    and `kBT` is the custom parameter in eV.

.. code-block:: yaml
    :emphasize-lines: 7, 13

    selection:
      - method: property
        properties:
          max_devi_f:
            range: [0.05, null]
            nbins: 20
            sparsify: filter
      - method: property
        properties:
          max_devi_f:
            range: [0.05, 0.25]
            nbins: 20
            sparsify: hist
        number: [256, 1.0]


The first selection on property `max_devi_f` with `filter` will give an output file 
below

.. code-block:: yaml

    #Property max_devi_f
    # min 0.0304       max 17.9258
    # avg 0.7199       std 0.4960
    # histogram of 4914 points in the range (npoints: 5005)
          0.0500          3344
          0.9438          1547
          1.8376            11
          2.7314             2
          3.6252             4
          4.5189             3
          5.4127             1
          6.3065             0
          7.2003             0
          8.0941             0
          8.9879             0
          9.8817             0
         10.7755             0
         11.6693             0
         12.5631             0
         13.4568             1
         14.3506             0
         15.2444             0
         16.1382             0
         17.0320             1

There 4914 structure from 5005 have `max_devi_f` within [0.05,inf]. The rest 91 
structures have a `max_devi_f` smaller than 0.05.
        

Descriptor
----------

`Select structures based on descriptors.` 

Two sparsification methods are supported.

- cur:

    Run CUR decomposition to select the most representative structures. This method 
    computes a CUR score for every structure and `strategy` defines the selection 
    either performs a deterministic selection (`descent`), structures with the `number` largest scores, 
    or a random one (`stochastic`), structures with higher scores that have higher probability. 
    If `zeta` is larger than 0., the input descripters will be transformed as 
    `MATMUL(descriptors.T, descriptors)^zeta`.

- fps:

    The farthest point sampling strategy. `min_distance` can be set to adjust the 
    sparsity of selected structures in the feature (descriptor) space.

.. code-block:: yaml

    selection:
        - method: descriptor
        descriptor:
            name: soap
            species: ["H", "O", "Pt"]
            rcut : 6.0
            nmax : 12
            lmax : 8
            sigma : 0.3
            average : inner
            periodic : true
        sparsify:
            method: cur # fps
            zeta: -1
            strategy: descent
        number: [16, 1.0]

.. |dscribe| image:: ../../images/dscribe.png
    :width: 400

This selection will produce a picture to visualise the distribution of structures.

    |dscribe|

.. note:: 

    This requires the python package `dscribe` to be installed. Use `pip install` or 
    `conda install dscribe -c conda-forge`.

Graph
-----

...