(mc-cu111-oxidation-xreac)=

# Cu(111) oxidation with xreac

This short grand-canonical example attempts three atomic-O exchanges above a
three-layer Cu(111)-p(2×2) slab. The four Cu atoms in the bottom layer are fixed
in the input structure. With `random_seed: 2026`, the first insertion is
accepted and `mc.xyz` changes from Cu12 to Cu12O.

```{literalinclude} ../../../../../examples/monte_carlo/cu111-oxidation-xreac.yaml
:language: yaml
```

The ASE single-point executor uses the xreac backend and its bundled
`ffield.reax.CuOHCl.2010` force field. Install `gdpx[reax]`, then run from the
repository root:

```shell
gdp -d run-mc-cu111-oxidation \
    explore examples/monte_carlo/cu111-oxidation-xreac.yaml
cat run-mc-cu111-oxidation/opstat.txt
```

The run performs the initial energy evaluation and three trial evaluations. A
cold command-line run completed in about 20 seconds during validation; the MC
work itself took less than a second. Startup time depends on the environment,
but the example is sized to remain comfortably below one minute.

The oxygen chemical potential of -2.0 eV is chosen to make an accepted exchange
visible in this deterministic demonstration. It is not a calibrated oxygen
reservoir value and should not be used for physical coverage predictions.
