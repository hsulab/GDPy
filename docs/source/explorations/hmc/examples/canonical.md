# Canonical: Cu displacements and MD

The initial Cu32 FCC supercell has a fixed 7.2 Å cubic cell. Langevin NVT MD
moves all atoms; each MC proposal displaces one Cu atom. Composition and cell
remain fixed throughout.

```{literalinclude} ../../../../../examples/monte_carlo/hybrid-canonical.yaml
:language: yaml
```

From the repository root:

```shell
gdp -d run-mc-hybrid-canonical explore examples/monte_carlo/hybrid-canonical.yaml
python examples/monte_carlo/inspect_run.py run-mc-hybrid-canonical
```

See the {ref}`Hybrid Monte Carlo guide <hybrid-monte-carlo>` for procedure
settings, runtime roles, sampling limitations, outputs, and restart behaviour.
