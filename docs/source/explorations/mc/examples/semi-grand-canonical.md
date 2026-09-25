# Semi-grand-canonical: Cu/Ni identity changes

Start from Cu16Ni16 in the same fixed cell. Half the operator selections attempt
a displacement; the other half attempt a Cu ↔ Ni identity change. Total
$N=32$ stays fixed while the Cu/Ni counts fluctuate.

```{literalinclude} ../../../../../examples/monte_carlo/semi-grand-canonical.yaml
:language: yaml
```

`system.ensemble.chemical_potentials` maps each species to eV per atom. For an
A → B change the implemented rule uses
$\Delta E - (\mu_B-\mu_A)$ in place of $\Delta E$ in the Metropolis
exponent, together with the reverse/forward proposal-count ratio. Increasing
the Ni value relative to Cu therefore favours Ni. Only the difference matters:
`{Cu: 0.0, Ni: 0.2}` and `{Cu: 1.0, Ni: 1.2}` give the same
chemical contribution. The demonstration uses equal chemical potentials;
this does not impose equal concentrations.

## Run

From the repository root, use the shared {doc}`EMT single-point runtime <index>`:

```shell
gdp -d run-mc-semi-grand-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/semi-grand-canonical.yaml
python examples/monte_carlo/inspect_run.py run-mc-semi-grand-canonical
```

See the {ref}`Monte Carlo guide <monte-carlo>` for sampling limitations,
output interpretation, and restart instructions.
