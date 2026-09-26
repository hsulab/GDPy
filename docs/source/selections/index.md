# gdp select

This section gives more details how to run basic selections with different selectors
using a unified input file.

The related commands are

```shell
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
```

The selection configuration (`./selection.yaml`) is organised as

```yaml
selection:
  - method: property
    ... # define property and sparsification
    number: [512, 0.5]
  - method: descriptor
    ... # define descriptor and sparsification
    number: [128, 1.0]
```

This `selection` defines a sequential produces that consists of two selectors.
Input structures will be first selected based on the property and the selected
structures will be further selected based on the descriptor.

For the most selections, a parameter `number` is required as it determines the number of
selected structures. The first value is a fixed number and the second value is a percentage.
If the input dataset have 500 structures, with `number: [512, 0.5]`, 250 structures will be
selected (500\*0.5=250 as 500 < 512). Then, with `number: [128, 1.0]`, 128 structures will be
selected (250 > 128).

After a successful selection, there are several output files. The `selected_frames.xyz`
contains the final structures. Output files start with a number that indicates their
oder in the list of selections. Some files, which end with `-info`, stores basic information
of selected structures. For example,

```shell
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
```

The first columns are structure identifiers that come from explorations, for instance,
the candidate ID (confid) and the dynamics step (step) in MD or minimisation. Other notations are `natoms` - number of atoms, `ene` - total energy, `aene` -
average atomic energy (ene/natoms), `maxfrc` - maximum atomic force, `score` - selection
score whose meaning depends on the sparsification method. Units are in `eV` nad `eV/Ang`.

There have some other output files by specific selection method. Find details in the following
subsections.

:::{warning}
When run the same selection again, `gdp` will read the cached results (`-info.txt` files).
However, it will not check whether the input structures are different from the last time.
Remove output files before selection if necessary.
:::

## List of Selectors

See {doc}`methods` for property and descriptor selectors.
