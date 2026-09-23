# property

`Select structures based on properties.` The property can be total energy, atomic forces, or
any properties that can be stored in the **ase** `atoms.info`. The example below demonstrates
that the selection based on `max_devi_f` that is the maximum deviation of force prediction by
a committee of MLIPs.

After chosing the property, there are several sparsification methods to select structures.

- filter:

  > Select structures that have property within `range`. All valid structures will be
  > selected, which is not affected by the parameter `number`.

- sort:

  > Sort structures by property and select the first `number` of them. Set `reverse: True`
  > if structures with larger property values are of interest.

- hist:

  > Randomly select `number` structures based on probabilities by the histogram.
  > For example, if 10 structures will be selected, dataset has 100 structures in
  > bin 1 and 25 in bin 2, then roughly 8 will be from bin 1 and 2 from bin 2.

- boltz:

  > Randomly select `number` structures based on probabilities by the Boltzmann distribution.
  > This is useful when selecting structures based on energy-related properties.
  > The probabilty is computed as `exp(-p/kBT)` where `p` is the property value
  > and `kBT` is the custom parameter in eV.

```{code-block} yaml
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
```

The first selection on property `max_devi_f` with `filter` will give an output file
below

```yaml
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
```

There 4914 structure from 5005 have `max_devi_f` within [0.05,inf]. The rest 91
structures have a `max_devi_f` smaller than 0.05.
