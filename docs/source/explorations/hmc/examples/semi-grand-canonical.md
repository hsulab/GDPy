# Semi-grand-canonical: Cu/Ni identity changes and MD

The initial Cu16Ni16 alloy uses the same cell. MD changes positions; the MC
block attempts Cu ↔ Ni identity changes while preserving the total of 32 atoms.
`chempots` is in eV per atom in `particles` order. Raising the Ni chemical
potential relative to Cu favours Ni; equal values do not fix the composition.

```{literalinclude} ../../../../../examples/monte_carlo/hybrid-semi-grand-canonical.yaml
:language: yaml
```

```shell
gdp -d run-mc-hybrid-semi-grand-canonical \
    explore examples/monte_carlo/hybrid-semi-grand-canonical.yaml
python examples/monte_carlo/inspect_run.py run-mc-hybrid-semi-grand-canonical
```

The current `swap_type` proposal-count limitation remains when identity
changes are combined with MD. This is a demonstration of variable-composition
MD/MC, not a validated semi-grand-canonical equilibrium calculation.

See the {ref}`Hybrid Monte Carlo guide <hybrid-monte-carlo>` for procedure
settings, runtime roles, sampling limitations, outputs, and restart behaviour.
