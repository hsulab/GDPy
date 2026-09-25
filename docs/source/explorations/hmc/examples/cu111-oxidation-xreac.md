(hmc-cu111-oxidation-xreac)=

# Cu(111) oxidation with xreac

This example applies HMC to the same three-layer Cu(111)-p(2×2) oxidation
system as the {doc}`MC example <../../mc/examples/cu111-oxidation-xreac>`. One
cycle runs 40 NVT MD steps followed by three grand-canonical atomic-O
exchanges. The four Cu atoms in the bottom layer are fixed in the initial,
MD, and MC runtimes.

```{literalinclude} ../../../../../examples/monte_carlo/hybrid-cu111-oxidation-xreac.yaml
:language: yaml
```

The ASE executors use the xreac backend with its bundled
`ffield.reax.CuOHCl.2010` force field. Install `gdpx[reax]`, then run from the
repository root:

```shell
gdp -d run-hmc-cu111-oxidation \
    explore examples/monte_carlo/hybrid-cu111-oxidation-xreac.yaml
cat run-hmc-cu111-oxidation/mcmoves.log
```

With `random_seed: 2026`, the first insertion is accepted and the cycle changes
Cu12 to Cu12O. During validation, the fixed layer remained stationary while the
mobile Cu layers moved, and a cold command-line run completed in about 20
seconds.

The oxygen chemical potential of -2.0 eV is chosen to expose an accepted move
in this deterministic demonstration. It is not a calibrated reservoir value.
