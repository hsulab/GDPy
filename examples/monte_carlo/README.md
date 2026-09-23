# EMT Monte Carlo examples

Run from the GDPy repository root after installing GDPy (`python -m pip install
-e .`). ASE's EMT requires no downloaded model or external executable.

```shell
gdp -d run-mc-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/canonical.yaml
gdp -d run-mc-semi-grand-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/semi-grand-canonical.yaml
gdp -d run-mc-grand-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/grand-canonical.yaml
```

| Input | System | Expected changes |
| --- | --- | --- |
| `canonical.yaml` | Periodic Cu32 | Positions; fixed composition and cell |
| `semi-grand-canonical.yaml` | Periodic Cu16Ni16 | Cu/Ni identities and positions; fixed 32 atoms and cell |
| `grand-canonical.yaml` | Cu8 plus one stationary Au in a periodic box | Cu insertion/removal; fixed Au count and cell |

`swap_type` changes an atom's
identity; `swap` would preserve species counts. All examples run 100 steps at
1200 K using single-point energies (`spc`), without minimisation. Chemical
potentials are in eV per atom and follow `particles` order. The grand-canonical
2.0 eV value is an illustrative EMT setting, not an experimental calibration.

These are workflow demos, not validated equilibrium studies. The current
`swap_type` species-first selection lacks a proposal-count correction, and
`exchange` lacks a correction for forced insertion at zero exchangeable count.
Disabling geometry checks and retries in the inputs does not fix those issues.
See the [example guides](../../docs/source/explorations/mc/examples/index.md)
for the individual ensembles and the [MC guide](../../docs/source/explorations/mc.md) for acceptance formulas,
parameter meanings, limitations, and restart details.

Initial structures are included. Regenerate them with:

```shell
python examples/monte_carlo/prepare.py
```

Inspect a run with:

```shell
python examples/monte_carlo/inspect_run.py run-mc-semi-grand-canonical
```

`mc.xyz` contains the initial state and every completed step, including repeated
states on rejection. `mc_attempts.xyz` contains evaluated trials and
`opstat.txt` reports acceptance. `dump_period` controls retained calculation
directories, not trajectory thinning. Repeat the same command and output
folder to resume; use a new output folder after changing inputs or runtime.
