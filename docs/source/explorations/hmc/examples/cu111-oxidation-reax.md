(hmc-cu111-oxidation-reax)=

# Cu(111) oxidation with ReaxFF

This example alternates 25 fs of NVT molecular dynamics with five
grand-canonical oxygen moves on a two-layer Cu(111)-p(2×2) slab. Each MC move
inserts or removes one O atom in the 4.5 Å region above the surface. Ten cycles
produce 1000 MD steps and 50 attempted oxygen moves.

```{literalinclude} ../../../../../examples/monte_carlo/hybrid-cu111-oxidation-reax.yaml
:language: yaml
```

The input uses the xreac backend with ASE executors for initialization, MD, and
MC energies. Install `gdpx[reax]`; the input loads xreac's bundled
`ffield.reax.CuOHCl.2010`, which contains H, O, Cu, and Cl parameters. See the
{ref}`ReaxFF guide <potential-reax>` for backend requirements.

Run from the repository root:

```shell
gdp -d run-hmc-cu111-oxidation \
    explore examples/monte_carlo/hybrid-cu111-oxidation-reax.yaml
```

The atomic-O chemical potential of -4.95 eV is illustrative. Calibrate it to
the selected force field and reservoir reference before interpreting oxidation
coverage. The two-layer slab, small surface cell, short MD segments, and ten
cycles make this a workflow example rather than a converged surface study.
